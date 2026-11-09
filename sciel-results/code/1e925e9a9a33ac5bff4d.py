"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-Start Powell Optimizer with coarse-to-fine refinement and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 320.0  # 300s limit, 20s margin
        
        # Wrapper to count calls and enforce limits
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(x)

        n = self.dim
        best_x = np.zeros(n)
        best_fx = float('inf')
        
        # Phase 1: Coarse multi-start to identify the best basin
        # Use structured points (axes) + random to cover the space
        rng = np.random.default_rng(123)
        starts = [np.zeros(n)]
        # Axis-aligned points to help with bent cigar anisotropy
        for axis in range(n):
            for sign in [1, -1]:
                pt = np.zeros(n)
                pt[axis] = sign * 3.5
                starts.append(pt)
        # Random points
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, n))

        # Coarse pass: moderate tolerances, limited iterations to spread budget
        # We want to evaluate many starts to find the global basin
        coarse_max_iter = 50
        for x0 in starts:
            if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                break
            remaining = self.budget - self.calls
            if remaining < 10:
                break
            try:
                res = minimize(
                    eval_func,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': coarse_max_iter,
                        'ftol': 1e-8,
                        'xatol': 1e-4
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except StopIteration:
                break
            except Exception:
                continue

        # Phase 2: Fine refinement on the best basin found
        # Concentrate remaining budget on the best point with tighter tolerances
        if self.calls < self.budget and time.time() - self.start_time < self.time_limit - 10:
            try:
                # Use a larger maxiter to allow full convergence
                res = minimize(
                    eval_func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'xatol': 1e-8
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 3: Final high-precision polish
        if self.calls < self.budget and time.time() - self.start_time < self.time_limit - 5:
            remaining = self.budget - self.calls
            if remaining > 10:
                try:
                    res = minimize(
                        eval_func,
                        best_x,
                        method='Powell',
                        options={
                            'maxiter': 50,
                            'ftol': 1e-14,
                            'xatol': 1e-10
                        }
                    )
                    if res.fun < best_fx:
                        best_fx = res.fun
                        best_x = res.x.copy()
                except StopIteration:
                    pass
                except Exception:
                    pass

        # Ensure we return something valid
        if best_x is None or not np.all(np.isfinite(best_x)):
            best_x = np.zeros(n)
            if self.calls < self.budget:
                best_fx = eval_func(best_x)
            else:
                best_fx = 0.0
                
        return best_x, best_fx
# EVOLVE-BLOCK-END
