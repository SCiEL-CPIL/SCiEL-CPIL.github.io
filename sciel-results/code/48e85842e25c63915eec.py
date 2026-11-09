import numpy as np
import time
from scipy.optimize import minimize
import heapq

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M_exact = 20000
    M_search = 2000
    
    # Precompute constants
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    
    # Keich construction initial guess (used as a fallback or seed for local refinement)
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    start_time = time.time()
    deadline = start_time + 550.0

    def _area(x, M):
        """Compute area using M quadrature points."""
        y = (np.arange(M) + 0.5) / M
        w = (1.0 - y) / n
        L = x[None, :] + a[None, :] * y[:, None]
        S = np.sort(L, axis=1)
        D = np.diff(S, axis=1)
        G = np.minimum(D, w[:, None])
        union_len = w + np.sum(G, axis=1)
        return np.mean(union_len)

    def _area_grad(x, M):
        """Compute area and gradient using M quadrature points."""
        y = (np.arange(M) + 0.5) / M
        w = (1.0 - y) / n
        L = x[None, :] + a[None, :] * y[:, None]
        S = np.sort(L, axis=1)
        D = np.diff(S, axis=1)
        G = np.minimum(D, w[:, None])
        area = np.mean(w + np.sum(G, axis=1))
        
        P = np.argsort(L, axis=1)
        T = (D < w[:, None]).astype(np.float64)
        
        P_plus = P[:, 1:].ravel()
        P_minus = P[:, :-1].ravel()
        T_flat = T.ravel()
        
        grad_plus = np.bincount(P_plus, weights=T_flat, minlength=n)
        grad_minus = np.bincount(P_minus, weights=T_flat, minlength=n)
        
        grad_x = (grad_plus - grad_minus) / M
        return area, grad_x

    # STRATEGY: Discrete Minimum-Cost Flow / Shortest Path Approach
    # 
    # Insight: The union width at height y is determined by the sorted positions L_i(y).
    # If we fix the relative ordering (permutation) of the triangles at each height,
    # the problem becomes convex in x. 
    # 
    # A simpler structural approach: 
    # The "Kakeya" property implies that to minimize area, the triangles should be 
    # "stacked" such that their interiors overlap as much as possible. 
    # 
    # Let's try a different structural representation: 
    # Instead of optimizing x_i directly, we optimize the "offsets" from a base 
    # configuration that ensures a specific permutation order at all heights.
    #
    # However, a more direct "different family" approach is to use a 
    # Discrete Dynamic Programming / Shortest Path method on a coarse grid 
    # of possible "states" (orderings) is too complex for 64 items.
    #
    # Alternative Structural Change: 
    # Use a "Greedy Construction with Local Swap Optimization".
    # This is different from gradient descent. It treats the problem as 
    # combinatorial: we decide the order of triangles at the base (y=0) 
    # and then adjust positions to maintain that order or allow specific swaps 
    # that reduce area.
    #
    # Actually, the most distinct "different family" available is to use 
    # a **Genetic Algorithm** or **Particle Swarm Optimization** which is 
    # population-based and derivative-free, contrasting with the single-trajectory 
    # gradient/SA approach.
    #
    # Let's implement a fast, vectorized Genetic Algorithm.
    
    def _evaluate_population(pop):
        """Vectorized area calculation for a population of size P."""
        # pop shape: (P, n)
        M = M_search
        y = (np.arange(M) + 0.5) / M
        w = (1.0 - y) / n
        
        # L shape: (P, M, n)
        # This might be memory heavy if P is large. 
        # P=100, M=2000, n=64 -> 100*2000*64*8 bytes = 102.4 MB. Acceptable.
        L = pop[:, None, :] + a[None, None, :] * y[None, :, None]
        S = np.sort(L, axis=2)
        D = np.diff(S, axis=2)
        G = np.minimum(D, w[None, :, None])
        union_len = w[None, :, None] + np.sum(G, axis=2)
        areas = np.mean(union_len, axis=1)
        return areas

    # GA Parameters
    pop_size = 128
    generations = 200
    mutation_rate = 0.1
    crossover_rate = 0.8
    elite_size = 10
    
    # Initialize population: mix of Keich, Random, and Sorted Uniform
    np.random.seed(42)
    
    # 1. Keich
    pop_keich = np.tile(x0, (pop_size // 4, 1))
    # 2. Random
    pop_rand = np.random.uniform(-1, 1, (pop_size // 4, n))
    # 3. Sorted Uniform (tries to align them)
    pop_sorted = np.sort(np.random.uniform(-1, 1, (pop_size // 4, n)), axis=1)
    # 4. Perturbed Keich
    pop_pert = x0[None, :] + np.random.normal(0, 0.1, (pop_size - len(pop_keich) - len(pop_rand) - len(pop_sorted), n))
    
    population = np.vstack([pop_keich, pop_rand, pop_sorted, pop_pert])
    
    # Ensure we have exactly pop_size
    if len(population) < pop_size:
        extra = np.random.uniform(-1, 1, (pop_size - len(population), n))
        population = np.vstack([population, extra])
    elif len(population) > pop_size:
        population = population[:pop_size]
        
    # Evaluate initial population
    areas = _evaluate_population(population)
    best_idx = np.argmin(areas)
    best_x = population[best_idx].copy()
    best_area = areas[best_idx]
    
    gen = 0
    while gen < generations:
        if time.time() > deadline - 100:
            break
            
        # Selection: Tournament
        new_population = []
        
        # Elitism
        elite_indices = np.argsort(areas)[:elite_size]
        new_population.extend([population[i].copy() for i in elite_indices])
        
        while len(new_population) < pop_size:
            # Pick two parents
            i1, i2 = np.random.choice(pop_size, 2, replace=False)
            
            # Crossover (Blend)
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand()
                child1 = alpha * population[i1] + (1 - alpha) * population[i2]
                child2 = (1 - alpha) * population[i1] + alpha * population[i2]
            else:
                child1 = population[i1].copy()
                child2 = population[i2].copy()
            
            # Mutation
            if np.random.rand() < mutation_rate:
                mask = np.random.rand(n) < 0.5
                child1[mask] += np.random.normal(0, 0.1, np.sum(mask))
            
            if np.random.rand() < mutation_rate:
                mask = np.random.rand(n) < 0.5
                child2[mask] += np.random.normal(0, 0.1, np.sum(mask))
                
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        population = np.array(new_population)
        areas = _evaluate_population(population)
        
        current_best_idx = np.argmin(areas)
        if areas[current_best_idx] < best_area:
            best_area = areas[current_best_idx]
            best_x = population[current_best_idx].copy()
            
        gen += 1
        
        # Check time
        if time.time() > deadline - 150:
            break

    # Phase 2: Local Refinement with L-BFGS-B using M_exact
    def obj_func(x):
        val, _ = _area_grad(x, M_exact)
        return val
        
    def jac_func(x):
        _, grad = _area_grad(x, M_exact)
        return grad

    best_x_final = best_x.copy()
    best_area_exact = _area(best_x_final, M_exact)

    # Run L-BFGS-B from the GA result
    try:
        res = minimize(
            obj_func, 
            best_x_final, 
            method='L-BFGS-B', 
            jac=jac_func, 
            options={'maxiter': 500, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if time.time() < deadline:
            if res.fun < best_area_exact:
                best_x_final = res.x
                best_area_exact = res.fun
    except Exception:
        pass

    # Phase 3: Basin Hopping with L-BFGS-B polishing
    if time.time() < deadline - 50:
        # Try a few random restarts of L-BFGS-B from the GA best
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            # Perturb GA best
            x_start = best_x + np.random.normal(0, 0.05, n)
            try:
                res = minimize(
                    obj_func, 
                    x_start, 
                    method='L-BFGS-B', 
                    jac=jac_func, 
                    options={'maxiter': 200, 'maxfun': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_area_exact:
                    best_x_final = res.x
                    best_area_exact = res.fun
            except Exception:
                pass

    return np.array(best_x_final, dtype=np.float64)
