"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts for BBOB f5 (linear slope)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = np.clip(x, -5, 5)
            calls_used += 1
            try:
                f = float(func(x))
            except Exception:
                return np.inf
            if not np.isfinite(f):
                return np.inf
            return f

        def nelder_mead(x0, max_iter):
            nonlocal best_x, best_f, calls_used
            # Initial simplex
            simplex = [x0.copy()]
            fvals = [safe_func(x0)]
            best_local = x0.copy()
            best_local_f = fvals[0]
            if best_local_f < best_f:
                best_f = best_local_f
                best_x = best_local.copy()

            delta = 2.0
            for i in range(dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
                fvals.append(safe_func(xi))
                if fvals[-1] < best_local_f:
                    best_local_f = fvals[-1]
                    best_local = xi.copy()
                    if best_local_f < best_f:
                        best_f = best_local_f
                        best_x = best_local.copy()

            # Nelder-Mead iterations
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break

                # Sort simplex by function value
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Check convergence
                if abs(fvals[0] - fvals[-1]) < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = fvals[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_func(x_r)

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = safe_func(x_c)
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = safe_func(simplex[i])

                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_local_f:
                    best_local_f = fvals[min_idx]
                    best_local = simplex[min_idx].copy()
                    if best_local_f < best_f:
                        best_f = best_local_f
                        best_x = best_local.copy()

            return best_local, best_local_f

        # Multi-start with Nelder-Mead
        # Start 1: origin
        x0 = np.zeros(dim)
        nelder_mead(x0, budget // 2)

        # If we have budget left, try a few more starts
        rng = np.random.default_rng(42)
        for restart in range(3):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Random start within [-4, 4]^dim
            x_start = rng.uniform(-4, 4, dim)
            # Use remaining budget divided among restarts
            remaining = budget - calls_used
            if remaining < 10:
                break
            max_iter = max(5, remaining // (3 - restart))
            nelder_mead(x_start, max_iter)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
