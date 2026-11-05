import numpy as np
import time
import random

# Precompute for evaluation
N_COARSE = 8192
_THETA_COARSE = 2 * np.pi * np.arange(N_COARSE) / N_COARSE
_K = np.arange(24)
# exponents[k, j] = exp(i * k * theta_j)
# Shape (24, 8192)
_EXPO = np.exp(1j * np.outer(_K, _THETA_COARSE))

def _eval_norm(coeffs):
    """
    Fast sup norm evaluation.
    """
    # Coarse grid
    P = coeffs @ _EXPO
    mags = np.abs(P)
    
    # Find top 5 peaks
    top_idx = np.argsort(mags)[-5:]
    max_val = 0.0
    
    # Refine top peaks
    for i in top_idx:
        theta0 = _THETA_COARSE[i]
        a = theta0 - np.pi / N_COARSE
        b = theta0 + np.pi / N_COARSE
        
        # Golden section
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        # Helper to evaluate at t
        def f(t):
            z = np.exp(1j * t)
            # Iterative power calculation
            powers = np.empty(24, dtype=complex)
            powers[0] = 1.0
            for k in range(1, 24):
                powers[k] = powers[k-1] * z
            val = np.dot(coeffs, powers)
            return abs(val)
            
        fc = f(c)
        fd = f(d)
        
        for _ in range(30):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
        
        peak = max(fc, fd)
        if peak > max_val:
            max_val = peak
            
    return max_val

def _local_search(coeffs, time_budget=0.5):
    """
    Hill climbing.
    """
    current = coeffs.copy()
    current_val = _eval_norm(current)
    best = current.copy()
    best_val = current_val
    
    start_time = time.time()
    n = len(current)
    
    while time.time() - start_time < time_budget:
        improved = False
        # Try flipping each coefficient
        for i in range(n):
            # Flip
            current[i] *= -1
            new_val = _eval_norm(current)
            if new_val < current_val:
                current_val = new_val
                improved = True
                if current_val < best_val:
                    best = current.copy()
                    best_val = current_val
            else:
                # Revert
                current[i] *= -1
        if not improved:
            break
            
    return best, best_val

def construct():
    n = 24
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Initial Population
    pop = []
    
    # GRS Sequence
    g = [1]
    while len(g) < 32:
        g += [-x for x in g]
    g24 = np.array(g[:24], dtype=float)
    pop.append(g24)
    pop.append(-g24)
    pop.append(g24[::-1].copy())
    
    # Random
    random.seed(42)
    for _ in range(20):
        pop.append(np.array([1 if random.random() < 0.5 else -1 for _ in range(n)], dtype=float))
    
    # 2. Local Search on Initials
    for i in range(len(pop)):
        if time.time() - start_time > time_limit:
            break
        rem = time_limit - (time.time() - start_time)
        budget = min(0.5, rem / (len(pop) - i))
        pop[i], _ = _local_search(pop[i], budget)
    
    # Sort by fitness
    pop_fitness = [_eval_norm(c) for c in pop]
    
    best_coeffs = pop[0]
    best_score = pop_fitness[0]
    for i in range(1, len(pop)):
        if pop_fitness[i] < best_score:
            best_score = pop_fitness[i]
            best_coeffs = pop[i]
            
    # 3. Genetic Algorithm
    pop_size = len(pop)
    population = pop[:]
    fitnesses = pop_fitness[:]
    
    while time.time() - start_time < time_limit:
        # Selection: Tournament
        parents = []
        for _ in range(pop_size):
            idxs = random.sample(range(pop_size), 3)
            best_idx = min(idxs, key=lambda i: fitnesses[i])
            parents.append(population[best_idx])
            
        # Crossover and Mutation
        new_pop = []
        for i in range(0, pop_size, 2):
            p1 = parents[i]
            p2 = parents[i+1]
            # Uniform Crossover
            child = np.array([p1[j] if random.random() < 0.5 else p2[j] for j in range(n)], dtype=float)
            # Mutation
            if random.random() < 0.5:
                idx = random.randint(0, n-1)
                child[idx] *= -1
            
            new_pop.append(child)
            
        # Evaluate New Pop
        for i in range(len(new_pop)):
            if time.time() - start_time > time_limit:
                break
            # Local search on some of them
            if random.random() < 0.3:
                rem = time_limit - (time.time() - start_time)
                budget = min(0.2, rem / (len(new_pop) - i))
                new_pop[i], _ = _local_search(new_pop[i], budget)
                
        # Combine with old, select best
        combined = population + new_pop
        combined_fitness = fitnesses + [_eval_norm(c) for c in new_pop]
        
        # Keep best pop_size
        order = np.argsort(combined_fitness)
        population = [combined[i] for i in order[:pop_size]]
        fitnesses = [combined_fitness[i] for i in order[:pop_size]]
        
        if fitnesses[0] < best_score:
            best_score = fitnesses[0]
            best_coeffs = population[0]
            
    return [int(c) for c in best_coeffs]
