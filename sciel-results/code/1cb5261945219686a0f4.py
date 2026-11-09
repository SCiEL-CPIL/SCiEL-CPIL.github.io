import numpy as np
import itertools
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Grid-Nelder-Mead Optimizer for BBOB f8 (Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Explicit deadline guard: 290s out of 300s limit
        deadline = time.time() + 290
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Wrapper to count calls and enforce budget/deadline
        def safe_func(x):
            nonlocal calls, best_f, best_x, deadline
            if time.time() > deadline or calls >= self.budget:
                raise TimeoutError("Budget or time limit exceeded")
            x = np.asarray(x, dtype=float)
            if x.shape != (self.dim,):
                raise ValueError("Wrong shape")
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Deterministic Grid Search
        # The minimizer is guaranteed to be in [-4, 4]^5.
        # We use a 3-level grid per dimension: -4, 0, 4.
        # Total points: 3^dim. For dim=5, this is 243 points.
        levels = [-4.0, 0.0, 4.0]
        grid_points = list(itertools.product(levels, repeat=self.dim))
        
        for p in grid_points:
            if time.time() > deadline or calls >= self.budget:
                break
            
            x = np.array(p, dtype=float)
            try:
                fx = safe_func(x)
            except (TimeoutError, ValueError):
                break
            except Exception:
                # If function evaluation fails for other reasons, stop
                break

        # Phase 2: Local Refinement with Nelder-Mead
        if best_x is not None and calls < self.budget and time.time() < deadline:
            x0 = best_x.copy()
            # Estimate max iterations: each NM iteration uses ~dim+1 evaluations
            # We have (budget - calls) left.
            remaining = self.budget - calls
            maxiter = max(1, remaining // (self.dim + 1))
            
            try:
                # Nelder-Mead is good for this problem
                res = minimize(
                    safe_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Phase 3: Final Polish with Coordinate Descent (if budget remains)
        if best_x is not None and calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            step = 0.1
            rng = np.random.default_rng(1)
            
            while calls < self.budget and time.time() < deadline:
                improved = False
                for i in range(self.dim):
                    # Try positive and negative step
                    for sign in [1, -1]:
                        y = x.copy()
                        y[i] += sign * step
                        y = np.clip(y, -5.0, 5.0)
                        try:
                            fy = safe_func(y)
                        except (TimeoutError, ValueError):
                            break
                        except Exception:
                            break
                        
                        if fy < fx:
                            x = y
                            fx = fy
                            improved = True
                            step *= 1.5 # Increase step if improving
                    if time.time() > deadline or calls >= self.budget:
                        break
                if not improved:
                    step *= 0.5 # Decrease step if no improvement
                    if step < 1e-4:
                        break
                else:
                    # Reset step to a moderate value if we improved significantly?
                    # Or keep increasing? Standard CD often resets.
                    # Let's just keep the logic simple: if improved, we might have overshot?
                    # Actually, increasing step when improving is risky.
                    # Let's stick to a simpler restart strategy or just stop if step is small.
                    pass

        # Fallback if no evaluations were made
        if best_x is None:
            best_x = np.zeros(self.dim)
            try:
                best_f = safe_func(best_x)
            except Exception:
                best_f = float('inf')

        return best_x, best_f
