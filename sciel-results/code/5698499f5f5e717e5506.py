# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Uses a hybrid approach: Nelder-Mead for initial exploration followed by
    L-BFGS-B for fine-tuning. Exploits the anisotropic nature of Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0  # safety margin
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        # Wrapper to count calls and handle any potential issues
        def wrapper(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls += 1
            try:
                val = float(func(x))
                return val
            except Exception:
                return float('inf')

        rng = np.random.default_rng(42)

        # Strategy: Use Nelder-Mead for robust global-ish search, then L-BFGS-B for polishing
        # Bent Cigar is highly anisotropic, so we need to be careful with step sizes
        
        # Phase 1: Nelder-Mead with multiple restarts
        n_restarts_nm = 5
        
        for i in range(n_restarts_nm):
            if calls >= budget - 50:
                break
            if time.time() > deadline:
                break
            
            # Generate start point
            if i == 0:
                x_start = np.zeros(dim)
            else:
                x_start = rng.uniform(-5, 5, dim)
            
            try:
                result = scipy.optimize.minimize(
                    wrapper,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': min(150, budget - calls - 10),
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 2: L-BFGS-B polishing from the best point found
        if calls < budget - 10 and time.time() < deadline:
            try:
                result = scipy.optimize.minimize(
                    wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfev': min(100, budget - calls - 5)}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: If we still have budget, do a few more L-BFGS-B from random points
        remaining_budget = budget - calls
        if remaining_budget > 30 and time.time() < deadline:
            n_extra = min(3, remaining_budget // 40)
            for i in range(n_extra):
                if calls >= budget - 10:
                    break
                if time.time() > deadline:
                    break
                
                x_start = rng.uniform(-5, 5, dim)
                try:
                    result = scipy.optimize.minimize(
                        wrapper,
                        x_start,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'maxfev': min(50, budget - calls - 5)}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
