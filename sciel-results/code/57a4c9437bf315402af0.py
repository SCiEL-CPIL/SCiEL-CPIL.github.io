"""Nelder-Mead based optimizer for BBOB f19 (Griewank-Rosenbrock).

Uses random initialization with Nelder-Mead local refinement to exploit
the local unimodality of the Rosenbrock core after escaping Griewank
oscillations.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0
        rng = np.random.default_rng(42)
        
        def elapsed():
            return time.time() - t0
        
        def sample(x):
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            return x, f
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Phase 1: Random exploration to find promising regions
        # Use a fraction of budget for random sampling
        n_random = min(50, self.budget // 10)
        for i in range(n_random):
            if elapsed() >= deadline or calls_made >= self.budget:
                break
            x = rng.uniform(-4, 4, self.dim)
            x, f = sample(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
            x = rng.uniform(-4, 4, self.dim)
            x, f = sample(x)
            calls_made += 1
            best_x = x.copy()
            best_f = f
        
        # Phase 2: Nelder-Mead local refinement from best point
        if elapsed() < deadline - 10:
            try:
                res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True, 'maxfev': max(1, self.budget - calls_made - 5)}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5, 5)
                calls_made += res.nfev
            except Exception:
                pass
        
        # Phase 3: Try additional Nelder-Mead from nearby points
        if elapsed() < deadline - 5:
            for trial in range(3):
                if elapsed() >= deadline - 2 or calls_made >= self.budget - 5:
                    break
                # Perturb best_x slightly
                perturbation = rng.uniform(-0.5, 0.5, self.dim)
                x_start = np.clip(best_x + perturbation, -5, 5)
                try:
                    res = minimize(
                        func,
                        x_start,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True, 'maxfev': max(1, (self.budget - calls_made - 5) // 3)}
                    )
                    if res.fun < best_f:
                        best_f = float(res.fun)
                        best_x = np.clip(res.x, -5, 5)
                    calls_made += res.nfev
                except Exception:
                    pass
        
        # Phase 4: Final L-BFGS-B polish
        if elapsed() < deadline - 2:
            try:
                res = minimize(
                    func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfev': max(1, self.budget - calls_made - 2)}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5, 5)
                calls_made += res.nfev
            except Exception:
                pass
        
        # Final check
        if elapsed() < deadline - 0.5 and calls_made < self.budget:
            try:
                _, f_final = sample(best_x)
                if f_final < best_f:
                    best_f = f_final
                calls_made += 1
            except Exception:
                pass
        
        return best_x
# EVOLVE-BLOCK-END
