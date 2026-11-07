"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-stage optimizer: global random sampling, Nelder-Mead local
    refinement, and adaptive (1+1)-ES polishing, with an explicit deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        DEADLINE = t_start + 290.0  # 10s safety margin before 300s wall-clock

        rng = np.random.default_rng(12345)
        calls = [0]

        def safe_call(x):
            if calls[0] >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                val = float(func(x))
            except Exception:
                return None
            calls[0] += 1
            if not np.isfinite(val):
                return None
            return val

        def time_ok():
            return time.time() < DEADLINE

        # ---- Phase 1: global random sampling ----
        n_sample = min(80, budget)
        xs = []
        fs = []
        for _ in range(n_sample):
            if not time_ok() or calls[0] >= budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = safe_call(x)
            if fx is None:
                continue
            xs.append(x)
            fs.append(fx)

        if not fs:
            # fallback single point
            x = np.zeros(dim)
            fx = safe_call(x)
            if fx is None:
                return None, None
            return x, fx

        fs = np.array(fs)
        xs = np.array(xs)
        best_idx = int(np.argmin(fs))
        x_best = xs[best_idx].copy()
        f_best = float(fs[best_idx])

        # ---- Phase 2: Nelder-Mead on the best few starting points ----
        # Rank and pick top-k distinct starting points
        order = np.argsort(fs)
        k = min(3, len(order))
        for oi in order[:k]:
            if not time_ok() or calls[0] >= budget:
                break
            x0 = xs[oi].copy()
            x0 = np.clip(x0, -5.0, 5.0)

            # Build simplex
            simplex = [x0.copy()]
            f0 = safe_call(x0)
            if f0 is None:
                continue
            simplex.append(np.array([f0]))  # placeholder, will rebuild
            # Rebuild properly
            simplex = [x0.copy()]
            fvals = [f0]
            for d in range(dim):
                step = x0.copy()
                if abs(step[d]) < 1.0:
                    step[d] += 0.5
                else:
                    step[d] -= 0.5
                step = np.clip(step, -5.0, 5.0)
                fstep = safe_call(step)
                if fstep is None:
                    break
                simplex.append(step)
                fvals.append(fstep)
            if len(simplex) < 2:
                continue

            # Nelder-Mead iterations
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            nm_iter = 0
            max_nm_iter = 60
            while nm_iter < max_nm_iter and time_ok() and calls[0] < budget:
                nm_iter += 1
                # sort
                order_s = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order_s]
                fvals = [fvals[i] for i in order_s]
                n = len(simplex)
                if n < 2:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # reflection
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5.0, 5.0)
                fr = safe_call(xr)
                if fr is None:
                    break

                if fr < fvals[0]:
                    # expansion
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5.0, 5.0)
                    fe = safe_call(xe)
                    if fe is None:
                        break
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
                    # contraction
                    xc = np.clip(centroid + rho * (simplex[-1] - centroid), -5.0, 5.0)
                    fc = safe_call(xc)
                    if fc is None:
                        break
                    if fc < fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # shrink
                        for i in range(1, n):
                            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5.0, 5.0)
                            fi = safe_call(simplex[i])
                            if fi is None:
                                break
                            fvals[i] = fi
                        else:
                            pass
                        if fi is None:
                            break

                # track global best
                local_best = min(fvals)
                if local_best < f_best:
                    bi = int(np.argmin(fvals))
                    f_best = local_best
                    x_best = simplex[bi].copy()

        # ---- Phase 3: adaptive (1+1)-ES polishing from best point ----
        x = x_best.copy()
        fx = f_best
        sigma = 1.0
        while calls[0] < budget and time_ok():
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = safe_call(y)
            if fy is None:
                break
            if fy < fx:
                x, fx = y, fy
                f_best = fx
                x_best = x.copy()
                sigma *= 1.3
            else:
                sigma *= 0.8
            if sigma < 1e-6:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
