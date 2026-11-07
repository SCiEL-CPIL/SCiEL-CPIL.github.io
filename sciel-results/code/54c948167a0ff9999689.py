"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Random-search exploration + Nelder-Mead polish, time-guarded."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        total = self.budget
        start = time.time()
        deadline = start + 285.0  # safety margin before the 330s kill

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def call(x):
            nonlocal calls, best_x, best_f
            calls += 1
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = np.resize(x, dim)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---------- Phase 1: random search (fast early progress) ----------
        rng = np.random.default_rng(12345)
        n_rand = min(200, total - 1)
        # deterministic grid points for reliable early coverage
        grid_vals = [-4.0, -2.0, 0.0, 2.0, 4.0]
        # evaluate a handful of structured points first for a quick baseline
        structured = [
            np.array([0.0] * dim),
            np.array([4.0] * dim),
            np.array([-4.0] * dim),
            np.array([2.0] * dim),
            np.array([-2.0] * dim),
        ]
        for p in structured:
            if calls >= total:
                break
            if time.time() > deadline:
                break
            call(p)

        # random samples, biased toward the interior where the min lives
        while calls < total - 1 and time.time() < deadline:
            x = rng.uniform(-4.5, 4.5, dim)
            call(x)

        # ---------- Phase 2: Nelder-Mead polish from the best point ----------
        remaining = total - calls
        if remaining >= 5 and time.time() < deadline:
            nm_budget = remaining
            nm_x = self._nelder_mead(func, best_x.copy(), best_f, dim, nm_budget, deadline)
            if nm_x is not None:
                # nm_x is the best simplex vertex found
                # ensure it is registered as best
                if nm_x[1] < best_f:
                    best_f = nm_x[1]
                    best_x = nm_x[0].copy()

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, dim, budget, deadline):
        """Deterministic Nelder-Mead. Returns (best_x, best_f) or None."""
        # build initial simplex
        simplex = [x0.copy()]
        values = [f0]
        for i in range(dim):
            v = x0.copy()
            step = 0.1 if x0[i] >= 0 else -0.1
            v[i] += step
            v = np.clip(v, -5.0, 5.0)
            fv = float(func(v))
            simplex.append(v)
            values.append(fv)
        n = dim + 1

        alpha, gamma, rho = 1.0, 2.0, 0.5
        min_iter = 0
        # track best
        best_i = int(np.argmin(values))
        best_x = simplex[best_i].copy()
        best_f = values[best_i]

        # budget accounting: we already used (dim+1) calls in simplex build
        used = n
        max_iters = max(1, budget - n)
        if max_iters < 1:
            return best_x, best_f

        for it in range(max_iters):
            if time.time() > deadline:
                break
            # sort simplex by value
            order = np.argsort(values)
            simplex = [simplex[i].copy() for i in order]
            values = [values[i] for i in order]
            best_i = 0
            worst_i = n - 1

            # stop if converged
            spread = values[-1] - values[0]
            if spread < 1e-12:
                break

            # centroid of all but worst
            centroid = np.mean(simplex[:n - 1], axis=0)

            # reflection
            xr = centroid + alpha * (centroid - simplex[worst_i])
            xr = np.clip(xr, -5.0, 5.0)
            fr = float(func(xr))
            used += 1

            if fr < values[0]:
                # expansion
                xe = centroid + gamma * (centroid - simplex[worst_i])
                xe = np.clip(xe, -5.0, 5.0)
                fe = float(func(xe))
                used += 1
                if fe < fr:
                    simplex[worst_i] = xe
                    values[worst_i] = fe
                else:
                    simplex[worst_i] = xr
                    values[worst_i] = fr
            elif fr < values[worst_i]:
                simplex[worst_i] = xr
                values[worst_i] = fr
            else:
                # contraction
                xc = centroid + rho * (simplex[worst_i] - centroid)
                xc = np.clip(xc, -5.0, 5.0)
                fc = float(func(xc))
                used += 1
                if fc < values[worst_i]:
                    simplex[worst_i] = xc
                    values[worst_i] = fc
                else:
                    # shrink
                    for i in range(1, n):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        values[i] = float(func(simplex[i]))
                        used += 1

            # update best
            bi = int(np.argmin(values))
            if values[bi] < best_f:
                best_f = values[bi]
                best_x = simplex[bi].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
