import numpy as np
import time
import itertools
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 310  # Safety margin before 330s kill
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        N = self.dim
        budget = self.budget

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

        def safe_minimize(fun, x0, method, options, bounds=None):
            """Wrapper for scipy minimize that respects budget and time."""
            def wrapper(x):
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

            try:
                if bounds:
                    res = minimize(wrapper, x0, method=method, bounds=bounds, options=options)
                else:
                    res = minimize(wrapper, x0, method=method, options=options)
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 1: Global Search using a structured grid + random sampling
        # Rosenbrock rotated is tricky. A coarse grid helps find the valley.
        # 3^5 = 243 points.
        
        # We will reserve some budget for local refinement.
        # Let's use 150 calls for global search, 350 for local.
        # Or adaptively: if we find a very good point early, we can stop global.
        
        # Generate candidate points
        # 1. Grid points: {-5, 0, 5}
        grid_points = []
        levels = [-5.0, 0.0, 5.0]
        for combo in itertools.product(levels, repeat=N):
            grid_points.append(np.array(combo))
        
        # 2. Add some random points to break symmetry or catch off-grid minima
        # But determinism is preferred for reproducibility, though random with fixed seed is fine.
        # Let's stick to deterministic first.
        
        # Sort grid points by a heuristic? No, we need to evaluate.
        # We will evaluate them in a specific order: center first, then corners.
        # Actually, sorting by distance from 0 might be good? Or just evaluate all.
        # Since 243 < 500, we can evaluate all grid points if budget allows.
        
        # Let's evaluate grid points.
        # To maximize early progress (AOCC), we should evaluate promising points first.
        # For Rosenbrock, the minimum is often near (1,1...1) in standard form.
        # With rotation, it's shifted.
        # Let's just evaluate all 243 grid points. It's a strong global sampler.
        
        grid_results = []
        for x in grid_points:
            try:
                f_val = evaluate(x)
                grid_results.append((f_val, x))
            except (TimeoutError, RuntimeError):
                break
        
        if calls_made >= budget or time.time() - start_time > time_limit:
            return best_x, best_f

        # Sort by fitness to find top basins
        grid_results.sort(key=lambda x: x[0])
        
        # Select top K basins for local refinement
        K = 5
        top_basins = [x for _, x in grid_results[:K]]
        
        # If we have budget left, do local refinement on these basins.
        # We also want to try refining the very best point found so far.
        
        # Phase 2: Local Refinement
        # Use Nelder-Mead for robustness, then L-BFGS-B for precision.
        
        # Budget remaining
        remaining_budget = budget - calls_made
        # Allocate roughly equally among top basins, but prioritize the best.
        # Let's give 100 calls to the best, 50 to the next 2, 25 to the rest.
        # Total 100+50+50+25+25 = 250. We have ~257 left.
        
        # We'll just loop through top basins and run a local optimizer with a cap.
        # We need to be careful not to exceed budget.
        
        # Helper to run local optimization with a call limit
        def run_local(x0, max_calls):
            nonlocal calls_made
            # We can't easily pass max_calls to scipy, so we rely on the wrapper raising RuntimeError.
            # But we want to stop earlier if we make progress?
            # For now, just run and let the wrapper handle budget.
            # We'll use Nelder-Mead first.
            
            # Initial simplex size
            # For Rosenbrock, the valley is narrow. Small steps are good.
            # But if we start far, we need to move.
            # Nelder-Mead adapts.
            
            options_nm = {
                'maxiter': 200,
                'xatol': 1e-8,
                'fatol': 1e-8,
                'adaptive': True
            }
            
            safe_minimize(func, x0, 'Nelder-Mead', options_nm)
            
            # If we still have budget and time, try L-BFGS-B for polish
            if calls_made < budget and time.time() - start_time < time_limit:
                options_lbfgs = {
                    'maxiter': 50,
                    'ftol': 1e-12,
                    'gtol': 1e-10
                }
                bounds = [(-5, 5)] * N
                safe_minimize(func, best_x, 'L-BFGS-B', options_lbfgs, bounds)

        # Run local refinement on top basins
        for i, x0 in enumerate(top_basins):
            if calls_made >= budget or time.time() - start_time > time_limit:
                break
            
            # We don't strictly limit calls per basin here, but the wrapper will stop us.
            # However, if one basin consumes all budget, others won't run.
            # We should ideally distribute.
            # Let's estimate calls per basin.
            # Remaining budget / remaining basins.
            rem_basins = len(top_basins) - i
            if rem_basins > 0:
                # Don't use all remaining budget on one basin.
                # But we can't easily enforce this in the wrapper without changing it.
                # Let's just run. If it's slow, it will hit time limit or budget.
                # To be safer, we can reduce maxiter in options if budget is tight.
                pass
            
            run_local(x0, None)
        
        # Final check
        return best_x, best_f
