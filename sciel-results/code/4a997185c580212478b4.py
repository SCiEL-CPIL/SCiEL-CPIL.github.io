"""Improved BBOB f22 optimizer: Multi-Start (1+1)-ES + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-strategy optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        rng = np.random.default_rng(7)

        # Phase 1: Random exploration to find good basins
        n_random = min(60, max(15, self.budget // 8))
        random_points = []
        for i in range(n_random):
            if not check_time() or calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_func(x)
            random_points.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        random_points.sort(key=lambda t: t[0])

        # (1+1)-ES with 1/5 success rule
        def one_plus_one_es(x0, f0, max_steps=80):
            nonlocal best_x, best_f, calls_made

            if max_steps <= 0 or not check_time() or (self.budget - calls_made) <= 0:
                return

            dim = self.dim
            sigma = 0.5
            successes = 0
            trials = 0

            for _ in range(max_steps):
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                y = np.clip(x0 + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                trials += 1

                if fy < f0:
                    x0, f0 = y, fy
                    successes += 1
                    if f0 < best_f:
                        best_f = f0
                        best_x = x0.copy()

                # 1/5 success rule: adjust sigma
                if trials >= 5:
                    if successes / trials > 0.2:
                        sigma *= 1.25
                    else:
                        sigma *= 0.8
                    successes = 0
                    trials = 0

                sigma = np.clip(sigma, 1e-6, 10.0)

                if sigma < 1e-5:
                    break

        # Nelder-Mead local search
        def nelder_mead(x0, f0, max_steps=50):
            nonlocal best_x, best_f, calls_made

            if max_steps <= 0 or not check_time() or (self.budget - calls_made) <= 0:
                return

            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.05 * (1 if xi[i] < 0 else -1)
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex_f = [f0]
            for i in range(1, dim + 1):
                if not check_time() or (self.budget - calls_made) <= 0:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = simplex[i].copy()

            for step in range(max_steps):
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                if fxr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    if not check_time() or (self.budget - calls_made) <= 0:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_func(xc)
                    if not check_time() or (self.budget - calls_made) <= 0:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or (self.budget - calls_made) <= 0:
                                return
                            simplex_f[i] = safe_func(simplex[i])

                if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                    break

        # Phase 2: Multi-start (1+1)-ES from top random points
        top_k = min(5, len(random_points))
        for i in range(top_k):
            if not check_time() or (self.budget - calls_made) <= 0:
                break
            x_start = random_points[i][1].copy()
            f_start = random_points[i][0]
            one_plus_one_es(x_start, f_start, max_steps=80)

        # Phase 3: Nelder-Mead polish on current best
        remaining = self.budget - calls_made
        if remaining > 20 and best_x is not None and check_time():
            nelder_mead(best_x, best_f, max_steps=50)

        # Phase 4: Additional (1+1)-ES refinement if budget allows
        remaining = self.budget - calls_made
        if remaining > 30 and best_x is not None and check_time():
            one_plus_one_es(best_x, best_f, max_steps=60)

        # Phase 5: Final Nelder-Mead pass
        remaining = self.budget - calls_made
        if remaining > 15 and best_x is not None and check_time():
            nelder_mead(best_x, best_f, max_steps=30)

        return best_x, best_f
# EVOLVE-BLOCK-END
