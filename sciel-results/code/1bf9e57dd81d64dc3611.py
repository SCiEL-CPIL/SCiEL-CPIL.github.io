"""Hybrid global-to-local optimizer for BBOB f9 (rotated Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-phase optimizer: coarse grid -> adaptive lattice -> L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s kill

        # --- helpers ---
        def time_left():
            return deadline - time.time()

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # --- state ---
        best_x = np.full(dim, np.nan)
        best_f = np.inf
        calls = 0

        def eval(x):
            nonlocal best_x, best_f, calls
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = clip(x)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: coarse 5^5 grid (3125 points) for early AOCC ---
        # Use a deterministic Latin-hypercube-style stratified sample instead of full grid
        # to keep it fast and cover the space well.
        rng = np.random.default_rng(42)
        # Stratified sampling: divide each dim into 5 bins, pick one point per bin combo
        # Full grid 5^5 = 3125 is too many; use a stratified sample of ~200 points
        n_strat = min(200, budget // 5)
        for _ in range(n_strat):
            if time_left() < 30 or calls >= budget - 50:
                break
            x = np.array([rng.uniform(-5 + i * 2, -3 + i * 2) for i in range(dim)])
            eval(x)

        # --- Phase 2: adaptive lattice refinement around best region ---
        # Identify the best hypercube from a 2^5 coarse grid, then refine
        # First do a 2^5 = 32 point coarse grid to find promising region
        if time_left() > 30 and calls < budget - 50:
            # 2^5 coarse grid
            for combo in range(2 ** dim):
                if time_left() < 30 or calls >= budget - 50:
                    break
                x = np.array([
                    -5.0 if (combo >> i) & 1 == 0 else 5.0
                    for i in range(dim)
                ])
                # Actually use midpoints of 2 divisions: [-5,0] and [0,5]
                x = np.array([
                    -2.5 if (combo >> i) & 1 == 0 else 2.5
                    for i in range(dim)
                ])
                eval(x)

        # --- Phase 3: adaptive local refinement via coordinate search + L-BFGS-B ---
        # Use remaining budget for local optimization from best point
        remaining = budget - calls
        if remaining > 10 and time_left() > 10:
            # Start from best_x and do L-BFGS-B
            x0 = best_x.copy()
            if not np.all(np.isfinite(x0)):
                x0 = np.zeros(dim)

            # Wrap func for scipy
            def scipy_func(x):
                return float(func(np.clip(x, -5, 5)))

            try:
                res = minimize(
                    scipy_func, x0, method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfun': remaining}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

            # If L-BFGS-B used few calls, do additional local search
            if time_left() > 10 and calls < budget - 5:
                # Coordinate descent refinement
                x = best_x.copy()
                fx = best_f
                step = 0.5
                for _ in range(20):
                    if time_left() < 5 or calls >= budget - 2:
                        break
                    improved = False
                    for i in range(dim):
                        if time_left() < 5 or calls >= budget - 2:
                            break
                        for direction in [1, -1]:
                            x_new = x.copy()
                            x_new[i] = np.clip(x[i] + direction * step, -5, 5)
                            fx_new = eval(x_new)
                            if fx_new < fx - 1e-10:
                                x, fx = x_new, fx_new
                                improved = True
                    if not improved:
                        step *= 0.5
                        if step < 1e-4:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
