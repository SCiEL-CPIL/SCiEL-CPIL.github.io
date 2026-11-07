import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        # 330s limit, use 315s for safety margin
        self.deadline = self.start_time + 315

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = None
        best_f = np.inf
        candidates = []

        def counted_func(x):
            nonlocal eval_count, best_f, best_x
            if time.time() > self.deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise RuntimeError("Budget exceeded")
            
            x = np.asarray(x, dtype=float)
            x_clipped = np.clip(x, -5.0, 5.0)
            
            eval_count += 1
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            
            candidates.append((fx, x_clipped.copy()))
            
            return fx

        bounds = [(-5.0, 5.0)] * dim

        # Phase 1: Differential Evolution for global search
        # DE is a population-based global optimizer that handles rotated/non-convex landscapes well.
        # Use a moderate population and limited generations to stay within budget.
        de_budget = min(200, budget // 2)
        try:
            res_de = scipy.optimize.differential_evolution(
                counted_func,
                bounds=bounds,
                maxiter=20,
                popsize=15,
                tol=0.01,
                seed=42,
                polish=False,
                init='sobol'
            )
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = res_de.x
        except (TimeoutError, RuntimeError):
            pass
        except Exception:
            pass

        if best_x is None:
            return np.zeros(dim), np.inf

        # Sort candidates to get the best ones for L-BFGS-B restarts
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement with L-BFGS-B from top candidates
        top_candidates = candidates[:5]
        
        for idx, (f_val, x_start) in enumerate(top_candidates):
            if time.time() > self.deadline or eval_count >= budget - 5:
                break
            
            remaining_budget = budget - eval_count
            if remaining_budget <= 10:
                break
                
            try:
                res = scipy.optimize.minimize(
                    counted_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxfun': remaining_budget, 'maxiter': 500}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                continue

        return best_x, best_f
