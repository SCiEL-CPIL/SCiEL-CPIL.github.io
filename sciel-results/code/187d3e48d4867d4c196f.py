"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Sobol quasi-random global search with Nelder-Mead local refinement for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 330s limit, leave 60s margin

        d = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        # Generate Sobol sequence for quasi-random coverage
        from scipy.spatial import QhullError
        from scipy.stats import qmc

        # Use Sobol sampler for uniform coverage of [-5, 5]^d
        # We need enough points to cover the budget
        # Sobol requires power of 2 samples
        sobol_sample_size = 512  # 2^9, reasonable for 5D
        sampler = qmc.Sobol(d=d, scramble=True, seed=42)
        sobol_points_unit = sampler.random(sobol_sample_size)
        # Map from [0,1]^d to [-5,5]^d
        sobol_points = sobol_points_unit * 10.0 - 5.0

        # Phase 1: Evaluate Sobol points for global coverage
        for x in sobol_points:
            if not time_ok():
                break
            x_clipped = np.clip(x, -5, 5)
            fx = call_func(x_clipped)
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: Nelder-Mead refinement around best point
        # Use multiple restarts with Nelder-Mead from different starting points
        nm_iterations = 0
        nm_max_iter = 200  # max iterations per NM run
        
        # Try Nelder-Mead from the best point found
        x0 = best_x.copy()
        
        # Also try a few other promising points from Sobol sequence
        # Sort Sobol points by function value (we only have best, so use proximity)
        # Instead, let's do NM from best_x and a few nearby points
        
        nm_starts = [best_x.copy()]
        
        # Add a few points near the best for diversity
        rng = np.random.default_rng(12345)
        for _ in range(3):
            perturbation = rng.normal(0, 0.5, d)
            x_candidate = np.clip(best_x + perturbation, -5, 5)
            nm_starts.append(x_candidate)
        
        for start_point in nm_starts:
            if not time_ok():
                break
            
            # Nelder-Mead with our call_func wrapper
            # We need to track calls and respect the budget
            def nm_func(x):
                if not time_ok():
                    return best_fx + 1000  # Return large value to stop
                x_clipped = np.clip(x, -5, 5)
                return call_func(x_clipped)
            
            try:
                result = minimize(nm_func, start_point, method='Nelder-Mead',
                                  options={'maxiter': nm_max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
                
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass
        
        # Phase 3: Final fine-tuning with very small steps
        if calls_made < budget - 10 and time.time() < deadline:
            # Do a few more Nelder-Mead runs with tighter tolerance
            for _ in range(2):
                if not time_ok():
                    break
                
                def nm_func_fine(x):
                    if not time_ok():
                        return best_fx + 1000
                    x_clipped = np.clip(x, -5, 5)
                    return call_func(x_clipped)
                
                try:
                    result = minimize(nm_func_fine, best_x, method='Nelder-Mead',
                                      options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12,
                                               'initial_simplex': np.array([best_x + 0.01 * np.eye(d).flatten()[:d*d]]).reshape(d, d+1) if d <= 10 else None})
                    
                    if result.fun < best_fx:
                        best_fx = result.fun
                        best_x = np.clip(result.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
