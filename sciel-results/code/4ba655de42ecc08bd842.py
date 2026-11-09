"""Differential Evolution with adaptive parameters (jDE) for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """jDE (Differential Evolution with adaptive parameters) for BBOB f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget or check_time():
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f = float(func(x_clipped))
                return f, x_clipped
            except Exception:
                return None, None

        def jde(x0, max_calls, rng):
            """
            Differential Evolution with adaptive F and CR (jDE).
            Returns (best_x, best_f, calls_used).
            """
            n = dim
            pop_size = 20  # lambda = 20 for good global diversity
            
            # Initialize population with diversity
            pop = np.zeros((pop_size, n))
            fvals = np.zeros(pop_size)
            calls_used = 0
            
            # Initial population: mix of random and centered around x0
            for i in range(pop_size):
                if calls_used >= max_calls or check_time():
                    break
                if i == 0:
                    x_trial = x0.copy()
                elif i < 5:
                    # Some points near x0
                    x_trial = x0 + rng.uniform(-1.0, 1.0, n)
                else:
                    # Rest are random in the box
                    x_trial = rng.uniform(-5.0, 5.0, n)
                
                f, x_eval = call_func(x_trial)
                if f is None:
                    break
                calls_used += 1
                pop[i] = x_eval
                fvals[i] = f
            
            if calls_used == 0:
                return x0, np.inf, 0
            
            # Ensure all population members have valid values
            valid_count = min(calls_used, pop_size)
            if valid_count < pop_size:
                # Fill remaining with duplicates of best
                best_idx = np.argmin(fvals[:valid_count])
                for i in range(valid_count, pop_size):
                    pop[i] = pop[best_idx]
                    fvals[i] = fvals[best_idx]
                valid_count = pop_size
            
            local_best_x = pop[np.argmin(fvals)].copy()
            local_best_f = min(fvals)
            
            # jDE parameters
            F = 0.5
            CR = 0.5
            F_sigma = 0.1
            CR_sigma = 0.1
            stagnation_counter = 0
            no_improve_threshold = 3 * pop_size
            
            while calls_used < max_calls and not check_time():
                # Sort population by fitness
                order = np.argsort(fvals)
                pop_sorted = pop[order].copy()
                fvals_sorted = fvals[order].copy()
                
                new_pop = pop_sorted.copy()
                new_fvals = fvals_sorted.copy()
                
                for i in range(pop_size):
                    if calls_used >= max_calls or check_time():
                        break
                    
                    # Adaptive parameters (jDE)
                    if rng.random() < 0.1:
                        F = 0.5 + rng.normal(0, F_sigma)
                        F = np.clip(F, 0.1, 0.9)
                    if rng.random() < 0.1:
                        CR = 0.5 + rng.normal(0, CR_sigma)
                        CR = np.clip(CR, 0.1, 0.9)
                    
                    # DE/rand/1 mutation
                    # Choose 3 distinct random individuals different from i
                    candidates = list(set(np.random.choice(pop_size, 3, replace=False).tolist()))
                    while i in candidates:
                        candidates = list(set(np.random.choice(pop_size, 3, replace=False).tolist()))
                    a, b, c = candidates
                    d = a if a != i else b
                    e = b if b != i else c
                    f_idx = c if c != i else a
                    
                    # Ensure we have 3 distinct indices different from i
                    indices = [j for j in range(pop_size) if j != i]
                    if len(indices) < 3:
                        break
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    
                    # Mutation: v = x_a + F * (x_b - x_c)
                    v = pop_sorted[a] + F * (pop_sorted[b] - pop_sorted[c])
                    v = np.clip(v, -5.0, 5.0)
                    
                    # Crossover
                    j_rand = rng.integers(0, n)
                    u = np.zeros(n)
                    for j in range(n):
                        if rng.random() <= CR or j == j_rand:
                            u[j] = v[j]
                        else:
                            u[j] = pop_sorted[i][j]
                    
                    # Evaluate trial vector
                    f_u, x_eval = call_func(u)
                    if f_u is None:
                        break
                    calls_used += 1
                    
                    # Selection
                    if f_u <= fvals_sorted[i]:
                        new_pop[i] = x_eval
                        new_fvals[i] = f_u
                        if f_u < local_best_f:
                            local_best_f = f_u
                            local_best_x = x_eval.copy()
                            stagnation_counter = 0
                        else:
                            stagnation_counter += 1
                    else:
                        stagnation_counter += 1
                
                pop = new_pop
                fvals = new_fvals
                
                # Check for stagnation
                if stagnation_counter > no_improve_threshold:
                    break
                
            return local_best_x, local_best_f, calls_used

        def lbfgs_refine(x0, max_calls):
            """
            L-BFGS-B local refinement using the function directly.
            Returns (best_x, best_f, calls_used).
            """
            nonlocal best_x, best_f, n_calls
            calls_used = 0
            local_best_x = x0.copy()
            local_best_f = best_f

            def objective(x):
                nonlocal calls_used
                if calls_used >= max_calls or check_time():
                    return best_f
                calls_used += 1
                n_calls += 1
                try:
                    x_clipped = np.clip(x, -5, 5)
                    f = float(func(x_clipped))
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_clipped.copy()
                    return f
                except Exception:
                    return best_f

            try:
                result = minimize(objective, x0, method='L-BFGS-B', bounds=[(-5, 5)] * dim,
                                  options={'maxiter': 50, 'maxfun': max_calls})
                if result.fun < local_best_f:
                    local_best_f = result.fun
                    local_best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            return local_best_x, local_best_f, calls_used

        rng = np.random.default_rng(42)

        # Phase 1: Initial diverse sampling
        initial_candidates = [
            np.zeros(dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-4, 4, dim),
        ]

        for x in initial_candidates:
            f, x_eval = call_func(x)
            if f is not None:
                if f < best_f:
                    best_f = f
                    best_x = x_eval.copy()

        if n_calls >= budget or check_time():
            return best_x, best_f

        # Phase 2: jDE with adaptive parameters
        current_x = best_x.copy()
        res_x, res_f, used = jde(current_x, budget - n_calls, rng)

        if res_f < best_f:
            best_f = res_f
            best_x = res_x.copy()

        # Phase 3: L-BFGS-B local refinement
        if n_calls < budget and not check_time():
            remaining = budget - n_calls
            if remaining >= 10:
                lbfgs_alloc = min(remaining, 50)
                _, _, _ = lbfgs_refine(best_x, lbfgs_alloc)

        return best_x, best_f
# EVOLVE-BLOCK-END
