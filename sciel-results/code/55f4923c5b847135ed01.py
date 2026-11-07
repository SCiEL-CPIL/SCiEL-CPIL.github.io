import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        # 330s limit, use 320s for safety margin
        self.deadline = self.start_time + 320

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        eval_count = 0
        best_x = None
        best_f = np.inf
        # Keep track of top K candidates from initial sampling
        candidates = []

        def counted_func(x):
            nonlocal eval_count, best_f, best_x
            if time.time() > self.deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise RuntimeError("Budget exceeded")
            
            # Ensure x is a valid numpy array
            x = np.asarray(x, dtype=float)
            # Clip to bounds to ensure func receives valid input if it doesn't clip internally
            # The task says func clips, but we do it here for safety in our logic
            x_clipped = np.clip(x, -5.0, 5.0)
            
            eval_count += 1
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            
            # Add to candidates list for restarts
            candidates.append((fx, x_clipped.copy()))
            
            return fx

        # Phase 1: Global Search - Find good starting points
        # We don't need many points. 10-20 is sufficient to find the basin of attraction
        # for Rosenbrock in 5D, which has a large basin.
        n_global = 20
        for i in range(n_global):
            if time.time() > self.deadline or eval_count >= budget:
                break
            # Uniform sampling in the box
            x = rng.uniform(-5.0, 5.0, dim)
            try:
                counted_func(x)
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                continue

        if best_x is None:
            # Fallback if no valid evaluations were made
            return np.zeros(dim), np.inf

        # Sort candidates to get the best ones for restarts
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement with L-BFGS-B
        # L-BFGS-B is very effective for Rosenbrock.
        bounds = [(-5.0, 5.0)] * dim
        
        # We will try to optimize from the top 3 candidates found in Phase 1
        # This handles the case where the global minimum is in a different basin
        # or the first L-BFGS-B run converged to a local minimum (less likely for Rosenbrock, but possible).
        
        top_candidates = candidates[:3]
        
        for idx, (f_val, x_start) in enumerate(top_candidates):
            if time.time() > self.deadline or eval_count >= budget - 5:
                break
            
            remaining_budget = budget - eval_count
            if remaining_budget <= 10:
                break
                
            try:
                # Use a reasonable maxfun. L-BFGS-B usually converges in < 100 calls for 5D Rosenbrock.
                # But we allow up to the remaining budget to ensure we use all resources if it's stuck.
                res = scipy.optimize.minimize(
                    counted_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxfun': remaining_budget, 'maxiter': 500}
                )
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                continue
            
            # Update best if this run found something better
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x

        return best_x, best_f
