"""Direct-structured optimizer for BBOB f20 (Schwefel) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-start Nelder-Mead with CMA-ES refinement for BBOB f20 (Schwefel).
    Designed for BBOB f20 on [-5,5]^5 with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def try_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            fx = func(x)
            calls_made += 1
            return fx

        def nelder_mead(x0, max_calls, rng):
            """Nelder-Mead simplex method from x0."""
            nonlocal best_f, best_x, calls_made
            n = dim
            x = x0.copy().astype(float)
            fx = try_func(x)
            if fx is None:
                return
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Build initial simplex
            simplex = [x.copy()]
            simplex_f = [fx]
            for i in range(n):
                xi = x.copy()
                xi[i] += 0.5
                xi = np.clip(xi, -5, 5)
                fxi = try_func(xi)
                if fxi is None:
                    return
                simplex.append(xi)
                simplex_f.append(fxi)
                if fxi < best_f:
                    best_f = fxi
                    best_x = xi.copy()

            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            calls_in_nm = 1 + n  # initial simplex evaluations

            while calls_in_nm < max_calls:
                if calls_made >= budget:
                    return
                if time.time() > deadline:
                    return

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check convergence
                if max(simplex_f) - min(simplex_f) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = try_func(xr)
                if fxr is None:
                    return
                calls_in_nm += 1

                if fxr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = try_func(xe)
                    if fxe is None:
                        return
                    calls_in_nm += 1
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    # Contraction
                    if fxr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fxc = try_func(xc)
                    if fxc is None:
                        return
                    calls_in_nm += 1
                    if fxc < min(fxr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fsi = try_func(simplex[i])
                            if fsi is None:
                                return
                            simplex_f[i] = fsi
                            calls_in_nm += 1

                # Update best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

        def differential_evolution(max_calls, rng):
            """Differential Evolution for global search."""
            nonlocal best_f, best_x, calls_made
            n = dim
            pop_size = 10
            F = 0.8
            CR = 0.9
            
            # Initialize population
            population = []
            pop_f = []
            for _ in range(pop_size):
                if calls_made >= budget:
                    return
                if time.time() > deadline:
                    return
                x = rng.uniform(-5, 5, n)
                fx = try_func(x)
                if fx is None:
                    return
                population.append(x)
                pop_f.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            population = np.array(population)
            pop_f = np.array(pop_f)

            calls_in_de = pop_size
            stagnation_counter = 0

            while calls_in_de < max_calls:
                if calls_made >= budget:
                    return
                if time.time() > deadline:
                    return

                # Find best
                best_idx = np.argmin(pop_f)
                prev_best_f = pop_f[best_idx]

                new_population = population.copy()
                new_pop_f = pop_f.copy()

                for i in range(pop_size):
                    if calls_made >= budget:
                        return
                    if time.time() > deadline:
                        return

                    # Select 3 distinct random indices
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    while r1 == r2 or r2 == r3 or r1 == r3:
                         r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    
                    # Mutation
                    v = population[r1] + F * (population[r2] - population[r3])
                    
                    # Crossover
                    u = np.zeros(n)
                    j_rand = rng.integers(0, n)
                    for j in range(n):
                        if rng.random() < CR or j == j_rand:
                            u[j] = v[j]
                        else:
                            u[j] = population[i][j]
                    
                    u = np.clip(u, -5, 5)
                    fu = try_func(u)
                    if fu is None:
                        return
                    calls_in_de += 1

                    if fu <= pop_f[i]:
                        new_population[i] = u
                        new_pop_f[i] = fu

                population = new_population
                pop_f = new_pop_f

                # Update global best
                current_best_idx = np.argmin(pop_f)
                if pop_f[current_best_idx] < best_f:
                    best_f = pop_f[current_best_idx]
                    best_x = population[current_best_idx].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # If stagnant for too long, restart population around best
                if stagnation_counter > 5:
                    for i in range(pop_size):
                        if calls_made >= budget:
                            return
                        if time.time() > deadline:
                            return
                        if i == current_best_idx:
                            continue
                        x = best_x + rng.normal(0, 1.0, n)
                        x = np.clip(x, -5, 5)
                        fx = try_func(x)
                        if fx is None:
                            return
                        population[i] = x
                        pop_f[i] = fx
                        calls_in_de += 1
                    stagnation_counter = 0

        rng = np.random.default_rng(2024)

        # Phase 1: Random Sampling (Coarse Grid Approximation)
        # Sample 50 random points to find a good region
        sample_budget = 50
        for _ in range(sample_budget):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = try_func(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Differential Evolution
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            de_budget = min(remaining, 350)
            differential_evolution(de_budget, rng)

        # Phase 3: Nelder-Mead Refinement
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            nm_budget = min(remaining, 100)
            nelder_mead(best_x, nm_budget, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
