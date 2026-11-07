import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if self.calls >= budget:
                raise Exception("Budget Exceeded")
            self.calls += 1
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = float('inf')
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        rng = np.random.default_rng(12345)

        # 1. Initial Global Exploration (Latin Hypercube Sampling)
        # Use 30 evaluations to cover the space reasonably well.
        n_init = 30
        for i in range(n_init):
            x = np.zeros(dim)
            for d in range(dim):
                x[d] = rng.uniform(-5, 5)
            wrapped_func(x)
        
        # Also evaluate center
        x_center = np.zeros(dim)
        wrapped_func(x_center)
        
        if self.best_x is None:
            self.best_x = np.zeros(dim)
            self.best_f = wrapped_func(self.best_x)

        # 2. Nelder-Mead Implementation (from Inspiration 2)
        def run_nelder_mead(x0, max_calls, init_step=0.5):
            local_calls = 0
            local_best_x = x0.copy()
            local_best_f = wrapped_func(x0)
            local_calls += 1
            
            if local_calls >= max_calls:
                return local_best_x, local_best_f, local_calls

            # Parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink

            # Build simplex
            simplex = [x0.copy()]
            f_values = [local_best_f]
            
            # Other points
            for i in range(dim):
                x_new = x0.copy()
                step = init_step
                if i == 0:
                    x_new[i] += step
                else:
                    x_new[i] += step
                # Clip to box
                x_new = np.clip(x_new, -5, 5)
                f_new = wrapped_func(x_new)
                simplex.append(x_new)
                f_values.append(f_new)
                local_calls += 1
                
                if local_calls >= max_calls:
                    break
                if f_new < local_best_f:
                    local_best_f = f_new
                    local_best_x = x_new.copy()

            if local_calls >= max_calls:
                return local_best_x, local_best_f, local_calls

            # Nelder-Mead Main Loop
            while local_calls < max_calls:
                if time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Check convergence
                if np.std(f_values) < 1e-8:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst_idx = dim
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[worst_idx])
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)
                local_calls += 1
                if f_r < local_best_f:
                    local_best_f = f_r
                    local_best_x = x_r.copy()
                
                if local_calls >= max_calls:
                    break
                    
                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[worst_idx])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)
                    local_calls += 1
                    if f_e < local_best_f:
                        local_best_f = f_e
                        local_best_x = x_e.copy()
                    
                    if local_calls >= max_calls:
                        break
                    
                    if f_e < f_r:
                        simplex[worst_idx] = x_e
                        f_values[worst_idx] = f_e
                    else:
                        simplex[worst_idx] = x_r
                        f_values[worst_idx] = f_r
                elif f_r < f_values[-2]:
                    # Simple Reflection
                    simplex[worst_idx] = x_r
                    f_values[worst_idx] = f_r
                else:
                    # Contraction
                    # Outside contraction
                    x_c = centroid + rho * (simplex[worst_idx] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_func(x_c)
                    local_calls += 1
                    if f_c < local_best_f:
                        local_best_f = f_c
                        local_best_x = x_c.copy()
                    
                    if local_calls >= max_calls:
                        break
                    
                    if f_c < f_values[worst_idx]:
                        simplex[worst_idx] = x_c
                        f_values[worst_idx] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = wrapped_func(simplex[i])
                            local_calls += 1
                            if f_values[i] < local_best_f:
                                local_best_f = f_values[i]
                                local_best_x = simplex[i].copy()
                            if local_calls >= max_calls:
                                break
                        if local_calls >= max_calls:
                            break

            return local_best_x, local_best_f, local_calls

        # Strategy:
        # We have 500 calls. ~31 used for init. 469 remaining.
        # Run Nelder-Mead from best initial point.
        # Use most of the budget for the first run.
        # If stagnation is detected or budget remains, restart.
        
        remaining = budget - self.calls
        if remaining <= 0:
            return self.best_x, self.best_f

        # First run: 90% of remaining
        budget_run1 = int(remaining * 0.9)
        x_best, f_best, calls_used = run_nelder_mead(self.best_x.copy(), budget_run1)
        
        if f_best < self.best_f:
            self.best_f = f_best
            self.best_x = x_best

        remaining = budget - self.calls
        if remaining > 20 and time.time() < deadline - 10:
            # Check for stagnation: if improvement was minimal, try a new start
            # Or just refine if we improved.
            # Let's try a restart from a random point if we suspect we are stuck.
            # Heuristic: if the last run didn't improve much relative to the range, restart.
            # But we don't know the range. 
            # Let's just refine with a smaller sigma if we improved, or restart if not.
            
            # Simple approach: Run another Nelder-Mead from the current best with a smaller step to refine.
            # Or from a new random point if the first one was bad.
            # Since Discus is unimodal, refining is usually better if we are close.
            
            # Let's try a second run from the best point found.
            budget_run2 = remaining
            # Use smaller init_step for refinement
            x_best2, f_best2, calls_used2 = run_nelder_mead(self.best_x.copy(), budget_run2, init_step=0.2)
            
            if f_best2 < self.best_f:
                self.best_f = f_best2
                self.best_x = x_best2

        return self.best_x, self.best_f
