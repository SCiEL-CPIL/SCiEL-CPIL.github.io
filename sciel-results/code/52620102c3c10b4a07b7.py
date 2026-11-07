import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280  # 20s buffer before 300s kill
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        N = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise RuntimeError()
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        # Phase 1: Random Multi-Start Global Search
        # Use a small number of random starts to cover the space efficiently
        num_starts = 10
        start_points = []
        
        # Add origin as a safe start
        start_points.append(np.zeros(N))
        
        # Add random points in the likely region [-4, 4]
        for _ in range(num_starts - 1):
            x = rng.uniform(-4.0, 4.0, size=N)
            start_points.append(x)
        
        # Evaluate all start points first to sort them
        start_results = []
        try:
            for x in start_points:
                f_val = evaluate(x)
                start_results.append((f_val, x))
        except (TimeoutError, RuntimeError):
            pass

        if not start_results:
            return best_x, best_f

        # Sort by fitness, best first
        start_results.sort(key=lambda item: item[0])
        
        # Phase 2: Local Optimization with L-BFGS-B + Nelder-Mead refinement
        # For each top basin, run L-BFGS-B then Nelder-Mead to escape flat regions
        
        # We will try to refine the top 3 basins found in Phase 1.
        # Limiting to 3 allows deeper refinement per basin.
        top_basins = start_results[:3]
        
        for i, (f_init, x_init) in enumerate(top_basins):
            if calls_made >= budget or time.time() - start_time > time_limit:
                break
            
            # Step 2a: L-BFGS-B for fast gradient-based convergence
            # Allocate more iterations to the best initial point
            max_iter_lbf = 100 if i == 0 else 60
            try:
                res_lbf = minimize(
                    evaluate, 
                    x_init, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * N,
                    options={
                        'maxiter': max_iter_lbf,
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                        'maxls': 20
                    }
                )
                if res_lbf.fun < best_f:
                    best_f = res_lbf.fun
                    best_x = np.array(res_lbf.x)
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                pass

            # Step 2b: Nelder-Mead to escape shallow local minima / flat regions
            # Rosenbrock has a narrow valley; NM can help navigate this better than gradient methods
            if calls_made < budget and time.time() - start_time < time_limit:
                try:
                    res_nm = minimize(
                        evaluate, 
                        best_x, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': 500,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if res_nm.fun < best_f:
                        best_f = res_nm.fun
                        best_x = np.array(res_nm.x)
                except (TimeoutError, RuntimeError):
                    break
                except Exception:
                    pass

        # Phase 3: Final High-Precision Polish with L-BFGS-B
        # Use tight tolerances to squeeze out remaining error
        if calls_made < budget and time.time() - start_time > time_limit - 10:
            # Skip if very little time left to avoid incomplete polish
            pass
        elif calls_made < budget and time.time() - start_time < time_limit:
            try:
                res = minimize(
                    evaluate, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * N,
                    options={
                        'maxiter': 500,
                        'ftol': 1e-14,
                        'gtol': 1e-12,
                        'maxls': 50
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res_fun.x)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 4: One more Nelder-Mead polish if budget allows
        # Final NM pass to ensure we are at the bottom of the valley
        if calls_made < budget and time.time() - start_time < time_limit:
            try:
                res = minimize(
                    evaluate, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 300,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return best_x, best_f
