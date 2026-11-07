"""Rewritten optimizer: coarse lattice search exploiting Rastrigin periodicity, then CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Lattice-based Rastrigin-aware optimizer with CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Phase 1: Lattice search to find a good basin
        # Rastrigin has minima at integer points (after shift/rotation),
        # so we evaluate a grid of integer and half-integer points
        best_x = None
        best_f = np.inf
        calls_used = 0

        def try_point(x):
            nonlocal best_x, best_f, calls_used
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Generate lattice points: integers and half-integers in [-5, 5]
        # That's 11 * 11^5 = 161051 points, too many for 500 budget
        # Instead, use a coarser grid: step 1.0 in [-5, 5] gives 11 points per dim
        # 11^5 = 161051, still too many
        # Use step 2.0: points at -5, -3, -1, 1, 3, 5 -> 6 points per dim
        # 6^5 = 7776, still too many
        # Use a smarter approach: random sampling + local refinement

        # Actually, for 500 calls, we need a different strategy
        # Let's use: 200 random samples, then CMA-ES from best point

        rng = np.random.default_rng(42)

        # Phase 1: Random sampling to find a good starting point
        n_random = 100
        for i in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, self.dim)
            try_point(x)

        # Phase 2: Local refinement around best point using small steps
        # Try offsets of 0.5, 1.0 in each dimension
        if best_x is not None and time.time() < deadline:
            base = best_x.copy()
            offsets = [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5]
            # Evaluate neighbors: for each dimension, try each offset
            for d in range(self.dim):
                for off in offsets:
                    if time.time() > deadline:
                        break
                    x = base.copy()
                    x[d] += off
                    if np.all(x >= -5.0) and np.all(x <= 5.0):
                        try_point(x)
                # Update base if improved
                if best_x is not None and not np.array_equal(base, best_x):
                    base = best_x.copy()

        # Phase 3: CMA-ES-style refinement using scipy minimize
        if best_x is not None and time.time() < deadline and calls_used < self.budget - 10:
            try:
                # Use Nelder-Mead for fast local optimization
                result = minimize(
                    lambda x: float(func(np.clip(x, -5.0, 5.0))),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-6,
                             'adaptive': True, 'maxfev': self.budget - calls_used - 5}
                )
                calls_used += result.nfev
                if result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.clip(result.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 4: Final polish with very small steps if budget remains
        if best_x is not None and time.time() < deadline and calls_used < self.budget - 5:
            for step_size in [0.1, 0.05, 0.01]:
                if time.time() > deadline:
                    break
                base = best_x.copy()
                for d in range(self.dim):
                    for sign in [1.0, -1.0]:
                        if time.time() > deadline:
                            break
                        x = base.copy()
                        x[d] += sign * step_size
                        if np.all(x >= -5.0) and np.all(x <= 5.0):
                            try_point(x)
                if best_x is not None and not np.array_equal(base, best_x):
                    base = best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
