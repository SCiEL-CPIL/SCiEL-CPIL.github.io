"""Rastrigin-aware optimizer: coordinate descent on integer lattice + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Coordinate descent exploiting Rastrigin's integer-lattice local minima."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit

        rng = np.random.default_rng(7)

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Local minima at integer coordinates (after hidden shift/rotation).
        # Strategy: coordinate descent probing integer lattice, then refine.

        # Phase 1: Probe integer lattice points in [-4, 4]^dim
        # But 9^5 = 59049 is too many. Use adaptive coordinate descent.
        
        # Start at origin
        counted_func(np.zeros(dim))
        
        # Coordinate descent: for each dim, try shifting by ±1, ±2, ±3, ±4
        # to find the best integer offset per coordinate
        steps = [1, 2, 3, 4]
        for iteration in range(3):  # Multiple passes
            if calls_made >= budget or time.time() > deadline - 60:
                break
            for d in range(dim):
                if calls_made >= budget or time.time() > deadline - 60:
                    break
                for step in steps:
                    for direction in [1, -1]:
                        if calls_made >= budget or time.time() > deadline - 60:
                            break
                        x_try = best_x.copy()
                        x_try[d] += direction * step
                        if np.all(x_try >= -5) and np.all(x_try <= 5):
                            fx = counted_func(x_try)
                            if fx >= best_f:
                                # No improvement, stop expanding this direction
                                break
        
        # Phase 2: Sub-integer refinement via coordinate descent with smaller steps
        # Try half-integer and quarter-integer offsets
        for sub_steps in [[0.5, 0.25, 0.125], [0.1, 0.05, 0.01]]:
            if calls_made >= budget or time.time() > deadline - 30:
                break
            for iteration in range(2):
                if calls_made >= budget or time.time() > deadline - 30:
                    break
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline - 30:
                        break
                    for step in sub_steps:
                        for direction in [1, -1]:
                            if calls_made >= budget or time.time() > deadline - 30:
                                break
                            x_try = best_x.copy()
                            x_try[d] += direction * step
                            if np.all(x_try >= -5) and np.all(x_try <= 5):
                                fx = counted_func(x_try)
                                if fx >= best_f:
                                    break
        
        # Phase 3: Nelder-Mead polish from best point
        if calls_made < budget - 20 and time.time() < deadline - 10:
            remaining = budget - calls_made
            maxfev = min(100, remaining)
            try:
                minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'maxfev': maxfev, 'xatol': 1e-10, 'fatol': 1e-10}
                )
            except Exception:
                pass
        
        # Phase 4: If budget remains, try a few random restarts with local search
        while calls_made < budget - 30 and time.time() < deadline - 15:
            x = rng.normal(0, 1.0, dim)
            x = np.clip(x, -4.5, 4.5)
            counted_func(x)
            
            # Quick coordinate refinement from this point
            for d in range(dim):
                if calls_made >= budget - 10:
                    break
                for step in [0.5, 0.25, 0.1]:
                    for direction in [1, -1]:
                        if calls_made >= budget - 5:
                            break
                        x_try = best_x.copy()
                        x_try[d] += direction * step
                        if np.all(x_try >= -5) and np.all(x_try <= 5):
                            counted_func(x_try)
        
        # Final: one more Nelder-Mead if we have budget
        if calls_made < budget - 10 and time.time() < deadline - 5:
            remaining = budget - calls_made
            maxfev = min(50, remaining)
            try:
                minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'maxfev': maxfev, 'xatol': 1e-10, 'fatol': 1e-10}
                )
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
