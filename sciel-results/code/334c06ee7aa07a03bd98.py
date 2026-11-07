"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead based derivative-free optimizer with multi-start."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._calls = 0
        self._max_calls = budget

    def __call__(self, func):
        self._calls = 0
        self._max_calls = self.budget
        rng = np.random.default_rng(42)

        # Wrapper to track calls and enforce budget
        def wrapped_func(x):
            if self._calls >= self._max_calls:
                return float('inf')
            self._calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            if not np.isfinite(val):
                return float('inf')
            return float(val)

        best_x = None
        best_f = float('inf')

        # Strategy: Derivative-free Nelder-Mead with multiple restarts
        # Nelder-Mead is effective for low-dimensional problems and does not require gradients.
        
        # Phase 1: Global exploration with random starts
        n_starts = 8
        for i in range(n_starts):
            if self._calls >= self._max_calls:
                break
            
            x0 = rng.uniform(-5, 5, self.dim)
            
            try:
                # Nelder-Mead parameters
                # initial_simplex is default (unit simplex scaled by 0.05 or similar)
                # We can control the initial simplex size if needed, but default is often okay.
                # maxiter controls the number of iterations, each ~1 function call.
                # We allow up to 50 iterations per restart to spread the budget.
                res = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50, 
                        'xatol': 1e-8, 
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass
        
        # Phase 2: Local refinement around the best point
        if best_x is not None and self._calls < self._max_calls:
            n_refine = 5
            for i in range(n_refine):
                if self._calls >= self._max_calls:
                    break
                
                # Perturb the best point with moderate noise
                perturbation = 0.5 * rng.standard_normal(self.dim)
                x0 = np.clip(best_x + perturbation, -5.0, 5.0)
                
                try:
                    res = minimize(
                        wrapped_func,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 40, 
                            'xatol': 1e-10, 
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_f = res.fun
                        best_x = res.x.copy()
                        
                except Exception:
                    pass

        # Phase 3: Fine-tuning
        if best_x is not None and self._calls < self._max_calls:
            n_fine = 3
            for i in range(n_fine):
                if self._calls >= self._max_calls:
                    break
                
                # Very small perturbation
                perturbation = 0.1 * rng.standard_normal(self.dim)
                x0 = np.clip(best_x + perturbation, -5.0, 5.0)
                
                try:
                    res = minimize(
                        wrapped_func,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 30, 
                            'xatol': 1e-12, 
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_f = res.fun
                        best_x = res.x.copy()
                        
                except Exception:
                    pass
        
        # Final evaluation to ensure we have a valid point
        if best_x is None:
            # Fallback: just use a random point
            best_x = rng.uniform(-5, 5, self.dim)
            if self._calls < self._max_calls:
                self._calls += 1
                best_f = wrapped_func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
