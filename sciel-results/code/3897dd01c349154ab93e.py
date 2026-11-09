import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Derivative-free Optimizer for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 290.0
        
        d = self.dim
        N = self.budget
        
        # Global best tracking
        best_x = np.zeros(d)
        best_fx = float('inf')
        
        def eval_func(x):
            nonlocal best_x, best_fx
            if self.calls >= N:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            
            self.calls += 1
            fx = func(x)
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # 1. Coarse Global Sampling to identify the best basin
        rng = np.random.default_rng(42)
        
        candidates = [np.zeros(d)]
        for i in range(5):
            p = rng.uniform(-4.0, 4.0, size=d)
            candidates.append(p)
        candidates.append(np.array([4.0] * d))
        candidates.append(np.array([-4.0] * d))
        
        for c in candidates:
            if self.calls >= N or time.time() - self.start_time > self.time_limit:
                break
            try:
                eval_func(c)
            except StopIteration:
                break
            except Exception:
                pass

        # 2. Focused Local Optimization using Nelder-Mead
        # Nelder-Mead is robust and doesn't require gradient estimation.
        # For Bent Cigar, it can handle the anisotropy reasonably well if initialized well.
        if self.calls < N and time.time() - self.start_time < self.time_limit:
            try:
                calls_at_start = self.calls
                def eval_func_nm(x):
                    if self.calls - calls_at_start >= N - calls_at_start:
                        raise StopIteration("NM budget exceeded")
                    return eval_func(x)
                
                # Initial simplex size: small enough to stay local, large enough to explore
                # Bent cigar is narrow, so we need to be careful not to step out of the valley
                # but also not so small that we get stuck.
                # A common heuristic is 0.05 * scale of the domain or similar.
                # Let's try a moderate initial simplex.
                # x0 is best_x.
                # We create a simplex by perturbing each dimension.
                x0 = best_x.copy()
                simplex = [x0]
                for i in range(d):
                    v = x0.copy()
                    # Step size: 0.1 seems reasonable for [-5,5] box
                    v[i] += 0.1
                    simplex.append(v)
                
                minimize(
                    eval_func_nm,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True,
                        'disp': False,
                        'initial_simplex': np.array(simplex)
                    }
                )
            except StopIteration:
                pass
            except Exception:
                pass

        # 3. Final Polish with Powell
        if self.calls < N and time.time() - self.start_time < self.time_limit:
            try:
                calls_at_start = self.calls
                def eval_func_polish(x):
                    if self.calls - calls_at_start >= 50:
                        raise StopIteration("Polish budget exceeded")
                    return eval_func(x)
                
                minimize(
                    eval_func_polish,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 50,
                        'ftol': 1e-14,
                        'xatol': 1e-10,
                        'disp': False
                    }
                )
            except StopIteration:
                pass
            except Exception:
                pass

        return best_x, best_fx
