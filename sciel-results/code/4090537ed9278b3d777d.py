import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        lb = -5.0
        ub = 5.0
        
        # State
        best_x = np.zeros(dim)
        best_f = np.inf
        eval_count = 0
        rng = np.random.default_rng(42)
        
        # DE Parameters
        # For dim=5, a population size of 10 is a good balance between exploration and exploitation.
        # CR (Crossover Rate) 0.9 is standard for high-dimensional or complex landscapes.
        # F (Scale Factor) 0.8 is aggressive enough for multimodal.
        pop_size = 10
        CR = 0.9
        F = 0.8
        
        # Initialize population uniformly in [-5, 5]
        # We reserve 1 eval for the first point to ensure we have a valid best_x immediately
        # to satisfy the "first call" requirement safely.
        
        # Create population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        
        # Evaluate initial population
        # We must be careful with the budget.
        # pop_size evaluations.
        f_vals = np.full(pop_size, np.inf)
        
        for i in range(pop_size):
            if eval_count >= budget or time.time() > deadline:
                break
            x = pop[i]
            f = func(x)
            f_vals[i] = f
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        # If we didn't even get one eval (unlikely but safe), return zero
        if eval_count == 0:
             x0 = np.zeros(dim)
             f0 = func(x0)
             return x0, f0

        # Main DE Loop
        # Each iteration consumes pop_size evaluations (generating pop_size offspring)
        # We stop when budget is exhausted or time limit is reached.
        
        while eval_count + pop_size <= budget and time.time() < deadline:
            # Generate offspring
            offspring = np.empty_like(pop)
            offspring_f = np.empty(pop_size)
            
            for i in range(pop_size):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                # Select 3 distinct random indices
                idxs = rng.choice(pop_size, 4, replace=False)
                a, b, c, j = idxs[0], idxs[1], idxs[2], idxs[3]
                
                # Differential Evolution mutation
                # v_i = x_a + F * (x_b - x_c)
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Clip to bounds (standard DE practice for bounded box)
                v = np.clip(v, lb, ub)
                
                # Binary Crossover
                # With probability CR, take from v, else from x_i
                # Ensure at least one gene is taken from v (standard DE rule)
                mask = rng.random(dim) < CR
                if not np.any(mask):
                    # Force one random position
                    mask[rng.integers(dim)] = True
                    
                trial_x = np.where(mask, v, pop[i])
                
                # Evaluate offspring
                f_trial = func(trial_x)
                eval_count += 1
                offspring_f[i] = f_trial
                offspring[i] = trial_x
                
                # Selection
                if f_trial < f_vals[i]:
                    pop[i] = trial_x
                    f_vals[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial_x.copy()
                        
            # If we broke out of the inner loop early due to budget/time,
            # we should break the outer loop too to avoid inconsistent state.
            if eval_count >= budget or time.time() > deadline:
                break

        # Optional: Final polish with Nelder-Mead if we have budget left and time
        # DE often gets stuck in local minima. A quick local search from the best
        # point found can squeeze out more performance.
        if eval_count < budget and time.time() < deadline:
            remaining = budget - eval_count
            # Nelder-Mead typically needs dim+1 simplex points and iterates.
            # With 500 budget, if DE used most of it, remaining might be small.
            # If remaining is significant (e.g., > 20), try it.
            if remaining > 20:
                try:
                    # Simple local search using Nelder-Mead
                    # We need to wrap func to count evals
                    nm_eval_count = 0
                    def nm_func(x):
                        nonlocal nm_eval_count, best_f, best_x, eval_count
                        if nm_eval_count >= remaining or time.time() > deadline:
                            return np.inf
                        nm_eval_count += 1
                        eval_count += 1
                        x_clipped = np.clip(x, lb, ub)
                        f_val = func(x_clipped)
                        if f_val < best_f:
                            best_f = f_val
                            best_x = x_clipped.copy()
                        return f_val
                        
                    # Initial simplex
                    x0 = best_x
                    simplex = [x0]
                    for i in range(dim):
                        xi = x0.copy()
                        # Small step
                        step = 0.1 if abs(x0[i]) > 1e-3 else 1.0
                        if x0[i] > 0:
                            xi[i] -= step
                        else:
                            xi[i] += step
                        simplex.append(xi)
                        
                    # Use scipy.optimize.minimize for robustness
                    import scipy.optimize
                    res = scipy.optimize.minimize(
                        nm_func, 
                        x0, 
                        method='Nelder-Mead',
                        options={'maxiter': remaining // 2, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                        
                except Exception:
                    pass

        # Ensure we return a valid point
        if np.isnan(best_f) or best_f == np.inf:
             best_x = np.zeros(dim)
             best_f = func(best_x)
             
        return best_x, best_f
