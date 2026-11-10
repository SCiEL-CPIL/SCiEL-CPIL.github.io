import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        budget = self.budget
        calls_left = [budget]

        def guarded(x):
            if calls_left[0] <= 0:
                return float('inf')
            calls_left[0] -= 1
            try:
                return float(func(x))
            except Exception:
                return float('inf')

        # Phase 1: Sobol scan to find good starting points
        n_scan = min(60, budget // 4)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            u = sampler.random(n_scan)
        except Exception:
            rng = np.random.default_rng(42)
            u = rng.random((n_scan, dim))

        pts = -5.0 + 10.0 * u
        pts = np.clip(pts, -5, 5)

        best_x = None
        best_fx = float('inf')
        for i in range(n_scan):
            if time.time() >= deadline or calls_left[0] <= 0:
                break
            x = pts[i]
            fx = guarded(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = guarded(best_x)

        # Phase 2: Use Powell's method (derivative-free, good for anisotropic functions)
        # Powell is particularly good for bent cigar because it adapts to the anisotropy
        # by searching along principal axes that it discovers.

        def objective(x):
            if calls_left[0] <= 0:
                return float('inf')
            calls_left[0] -= 1
            try:
                return float(func(np.clip(x, -5, 5)))
            except Exception:
                return float('inf')

        # Set up Powell with reasonable initial direction set
        # Use scaled identity to give it a head start
        x0 = best_x.copy()
        
        # Run Powell's method
        try:
            result = minimize(
                objective,
                x0,
                method='Powell',
                options={
                    'maxiter': 50,
                    'xtol': 1e-10,
                    'ftol': 1e-10,
                    'disp': False
                }
            )
            if result.x is not None and result.fun < best_fx:
                best_x = np.clip(result.x, -5, 5)
                best_fx = result.fun
        except Exception:
            pass

        # If we still have budget, do a final refinement with Nelder-Mead or another Powell run
        if calls_left[0] > 20 and time.time() < deadline - 5:
            try:
                result2 = minimize(
                    objective,
                    best_x.copy(),
                    method='Powell',
                    options={
                        'maxiter': 30,
                        'xtol': 1e-12,
                        'ftol': 1e-12,
                        'disp': False
                    }
                )
                if result2.x is not None and result2.fun < best_fx:
                    best_x = np.clip(result2.x, -5, 5)
                    best_fx = result2.fun
            except Exception:
                pass

        # Final local refinement using coordinate descent with adaptive steps
        if calls_left[0] > 10 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_fx
            steps = 0.1 * np.ones(dim)
            
            for _ in range(min(100, calls_left[0] // 2)):
                if time.time() >= deadline or calls_left[0] <= 0:
                    break
                improved = False
                for d in range(dim):
                    if calls_left[0] <= 0 or time.time() >= deadline:
                        break
                    
                    # Try positive
                    x_trial = x.copy()
                    x_trial[d] += steps[d]
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = objective(x_trial)
                    
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        improved = True
                        steps[d] *= 1.2
                    else:
                        # Try negative
                        x_trial = x.copy()
                        x_trial[d] -= steps[d]
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = objective(x_trial)
                        
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            improved = True
                            steps[d] *= 1.2
                        else:
                            steps[d] *= 0.7
                
                if not improved:
                    steps *= 0.8
                    if np.all(steps < 1e-8):
                        break
            
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx

        return best_x, best_fx
