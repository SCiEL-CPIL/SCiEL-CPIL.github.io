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
        
        # Use a fixed seed for reproducibility in the evaluator, 
        # but Sobol sequence is deterministic anyway.
        rng = np.random.default_rng(42)

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

        # Phase 1: Structured Global Search using Sobol Sequence
        # Sobol sequences provide low-discrepancy sampling, which is 
        # superior to uniform random for covering the search space efficiently.
        n_global = 10
        # Generate Sobol sequence points in [0, 1]^dim
        # scipy.stats.qmc.Sobol is available in SciPy >= 1.7
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        # We need to skip the first few points to get a better distribution? 
        # Or just use the next batch.
        # qmc.Sobol.next() returns a single point. 
        # It's easier to generate a batch.
        # Note: qmc.Sobol requires a power of 2 for the number of points in some versions, 
        # but `random` method works with any N. Let's use `random` for flexibility.
        sobol_points = sobol.random(n=n_global)
        
        # Map [0, 1]^dim to [-5, 5]^dim
        sobol_points = sobol_points * 10.0 - 5.0
        
        for i in range(n_global):
            if time.time() > self.deadline or eval_count >= budget:
                break
            x = sobol_points[i]
            try:
                counted_func(x)
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                continue

        if best_x is None:
            return np.zeros(dim), np.inf

        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Robust Local Refinement with Powell's Method
        # Powell's method is derivative-free and robust for navigating the 
        # narrow, curved valley of the Rosenbrock function.
        # We try the top 2 candidates.
        
        bounds = [(-5.0, 5.0)] * dim
        
        # We allocate a portion of the budget to Powell.
        # Powell can be slow to converge to high precision, but good for finding the basin.
        # We cap the function evaluations for Powell to save budget for L-BFGS-B polish.
        
        for idx in range(2):
            if idx >= len(candidates):
                break
            if time.time() > self.deadline or eval_count >= budget - 50:
                break
            
            f_val, x_start = candidates[idx]
            
            # If we already have a very good solution, skip further Powell runs
            if best_f < 1e-2:
                break
            
            try:
                # Powell's method
                # maxiter limits the number of iterations, maxfun limits function calls.
                # We give it a generous budget but not the whole remainder.
                remaining_for_powell = min(150, budget - eval_count - 50)
                if remaining_for_powell < 10:
                    break
                    
                res = scipy.optimize.minimize(
                    counted_func,
                    x_start,
                    method='Powell',
                    bounds=bounds,
                    options={'maxiter': 1000, 'maxfun': remaining_for_powell, 'xtol': 1e-8, 'ftol': 1e-8}
                )
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                continue
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x

        # Phase 3: High-Precision Polish with L-BFGS-B
        # Once a good basin is found, L-BFGS-B converges very quickly to the minimum.
        
        if time.time() > self.deadline or eval_count >= budget - 10:
            return best_x, best_f

        remaining_budget = budget - eval_count
        if remaining_budget <= 5:
            return best_x, best_f

        try:
            res = scipy.optimize.minimize(
                counted_func,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxfun': remaining_budget, 'maxiter': 500, 'ftol': 1e-15}
            )
        except (TimeoutError, RuntimeError):
            pass
        except Exception:
            pass
        
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x

        return best_x, best_f
