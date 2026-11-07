"""Nelder-Mead simplex with adaptive restarts for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Nelder-Mead simplex with adaptive restarts for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        deadline = time.time() + 280.0

        n_calls = [0]

        def safe_func(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Quick diverse starts to find a good basin early
        starts = [np.zeros(dim)]
        for scale in [1.5, 3.0, 4.0]:
            starts.append(np.full(dim, scale))
            starts.append(np.full(dim, -scale))
        for _ in range(8):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        for s in starts:
            if n_calls[0] >= budget or time.time() > deadline:
                break
            fx = safe_func(s)
            if fx < best_f:
                best_f = fx
                best_x = s.copy()

        # Phase 2: Nelder-Mead with restarts
        remaining = budget - n_calls[0]
        main_limit = int(n_calls[0] + remaining * 0.70)

        def nelder_mead(start_x, max_calls):
            nonlocal best_x, best_f
            n = dim
            # Initialize simplex: start_x plus n axis-aligned perturbations
            simplex = [start_x.copy()]
            for i in range(n):
                pt = start_x.copy()
                pt[i] += 0.5
                pt = np.clip(pt, -5.0, 5.0)
                simplex.append(pt)

            # Evaluate initial simplex
            fvals = []
            for i in range(n + 1):
                if n_calls[0] >= max_calls or time.time() > deadline:
                    fvals.append(np.inf)
                else:
                    fvals.append(safe_func(simplex[i]))

            # Track local best
            local_best_x = start_x.copy()
            local_best_f = min(fvals) if len(fvals) > 0 else np.inf
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            no_improve = 0
            stall_threshold = 30

            while n_calls[0] < max_calls and time.time() < deadline and no_improve < stall_threshold:
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update local best
                if fvals[0] < local_best_f:
                    local_best_f = fvals[0]
                    local_best_x = simplex[0].copy()
                    no_improve = 0
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()
                else:
                    no_improve += 1

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                centroid = np.clip(centroid, -5.0, 5.0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                if n_calls[0] < max_calls and time.time() < deadline:
                    fr = safe_func(xr)
                else:
                    fr = np.inf

                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    if n_calls[0] < max_calls and time.time() < deadline:
                        fe = safe_func(xe)
                    else:
                        fe = np.inf
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    if n_calls[0] < max_calls and time.time() < deadline:
                        fc = safe_func(xc)
                    else:
                        fc = np.inf
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            if n_calls[0] < max_calls and time.time() < deadline:
                                fvals[i] = safe_func(simplex[i])
                            else:
                                fvals[i] = np.inf

                # Re-sort and update
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if fvals[0] < local_best_f:
                    local_best_f = fvals[0]
                    local_best_x = simplex[0].copy()
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()

            return local_best_x, local_best_f

        # Run Nelder-Mead from the best start
        nm_x, nm_f = nelder_mead(best_x, main_limit)
        if nm_f < best_f:
            best_f = nm_f
            best_x = nm_x.copy()

        # Phase 3: Polish with smaller Nelder-Mead or fine (1+1)-ES
        if n_calls[0] < budget and time.time() < deadline:
            # Fine (1+1)-ES polish
            polish_sigma = 0.05
            polish_window = []
            polish_window_size = 6

            while n_calls[0] < budget and time.time() < deadline:
                candidate = best_x + polish_sigma * rng.standard_normal(dim)
                candidate = np.clip(candidate, -5.0, 5.0)
                fx = safe_func(candidate)

                if fx < best_f:
                    best_x = candidate.copy()
                    best_f = fx
                    polish_window.append(1)
                    polish_sigma *= 0.8
                else:
                    polish_window.append(0)
                    polish_sigma *= 0.95

                if len(polish_window) >= polish_window_size:
                    sr = sum(polish_window) / polish_window_size
                    if sr > 0.4:
                        polish_sigma *= 1.3
                    elif sr < 0.1:
                        polish_sigma *= 0.6
                    polish_window = polish_window[-polish_window_size // 2:]

                polish_sigma = max(polish_sigma, 1e-10)
                polish_sigma = min(polish_sigma, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
