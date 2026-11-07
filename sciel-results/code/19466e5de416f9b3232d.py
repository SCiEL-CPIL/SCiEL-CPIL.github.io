import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Leave a safety margin for the evaluator's overhead
        deadline = start_time + 290
        
        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return np.inf
            # Ensure x is a valid numpy array of floats
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            return func(x)

        bounds = [(-5.0, 5.0)] * dim
        
        # Strategy:
        # 1. Global Exploration with Latin Hypercube Sampling (LHS)
        # 2. Local Refinement with L-BFGS-B from the best candidates
        # 3. Final Polish if budget remains
        
        try:
            # Phase 1: Global Exploration
            # Use a moderate number of LHS points to cover the space well.
            # 5D is low, so 50 points give good coverage.
            n_lhs = 50
            sampler = LatinHypercube(d=dim, seed=42)
            lhs_points = sampler.random(n=n_lhs)
            # Map [0, 1] to [-4.5, 4.5] to stay safely inside [-5, 5] 
            # but cover most of the space where the minimum likely is.
            # The problem states minimizer is in [-4, 4]^5, so [-4.5, 4.5] is safe.
            lhs_points = lhs_points * 9.0 - 4.5
            
            lhs_results = []
            for x in lhs_points:
                if call_count[0] >= budget - 100: # Reserve budget for local search
                    break
                if time.time() > deadline - 50: # Time guard
                    break
                f_val = wrapped_func(x)
                if np.isfinite(f_val):
                    lhs_results.append((f_val, x.copy()))
                
            if not lhs_results:
                # Fallback if no valid points were evaluated
                best_x = np.zeros(dim)
                best_f = wrapped_func(best_x)
                return best_x, best_f

            # Sort by function value
            lhs_results.sort(key=lambda item: item[0])
            best_f, best_x = lhs_results[0]
            
            # Phase 2: Local Refinement
            # Pick the top few candidates to start local optimization.
            # This helps avoid local minima if the global one wasn't the absolute best LHS point.
            n_starts = min(3, len(lhs_results))
            starts = [p[1] for p in lhs_results[:n_starts]]
            
            # Allocate remaining budget for local search
            # L-BFGS-B is efficient, usually converging in < 50-100 evals for 5D Rosenbrock
            # once in the basin of attraction.
            
            for x0 in starts:
                if time.time() > deadline - 10:
                    break
                if call_count[0] >= budget - 20:
                    break
                
                try:
                    # Use L-BFGS-B for smooth constrained optimization
                    # maxiter controls the number of iterations, not function evals directly,
                    # but for 5D, it's a good proxy.
                    res = minimize(
                        wrapped_func, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={
                            'maxiter': 100,
                            'disp': False,
                            'maxls': 20 # Limit line search steps to prevent excessive evals
                        }
                    )
                    
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_f = res.fun
                        best_x = res.x.copy()
                        
                except Exception:
                    pass

            # Phase 3: Final Polish
            # If we have significant budget left and time, do one more high-precision pass
            # on the best point found so far.
            if call_count[0] < budget - 10 and time.time() < deadline - 5:
                try:
                    # Tighter tolerance for final polish
                    res_final = minimize(
                        wrapped_func, 
                        best_x, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={
                            'maxiter': 50,
                            'disp': False,
                            'ftol': 1e-12,
                            'gtol': 1e-8
                        }
                    )
                    if res_final.fun < best_f and np.isfinite(res_final.fun):
                        best_f = res_final.fun
                        best_x = res_final.x.copy()
                except Exception:
                    pass

        except Exception:
            # Critical failure fallback
            if best_x is None:
                best_x = np.zeros(dim)
                if call_count[0] < budget:
                    best_f = wrapped_func(best_x)
            # If we are out of time or budget, just return what we have

        # Ensure we return a valid array
        if best_x is None:
            best_x = np.zeros(dim)
        else:
            best_x = np.asarray(best_x, dtype=float)
            if best_x.shape != (dim,):
                best_x = best_x.reshape(dim)
            # Clip to bounds just in case
            best_x = np.clip(best_x, -5.0, 5.0)

        return best_x, best_f
