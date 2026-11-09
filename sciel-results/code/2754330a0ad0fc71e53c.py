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
        # Safety margin: evaluator kills at 300s. We stop at 290s to be safe.
        time_limit = 290.0

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
        # f22 has 21 peaks. We need enough samples to hit them.
        # 80 random points is a good balance. Cost: 80 calls.
        n_random = min(80, max(20, self.budget // 6))
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

        if not random_points:
            return best_x, best_f

        random_points.sort(key=lambda t: t[0])

        # (1+1)-ES with 1/5 success rule
        def one_plus_one_es(x0, f0, max_steps=60):
            nonlocal best_x, best_f, calls_made

            if max_steps <= 0 or not check_time() or (self.budget - calls_made) <= 0:
                return

            dim = self.dim
            # Initial sigma: 0.5 is reasonable for [-5,5] box
            sigma = 0.5
            successes = 0
            trials = 0
            x_curr = x0.copy()
            f_curr = f0

            for _ in range(max_steps):
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                y = np.clip(x_curr + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                trials += 1

                if fy < f_curr:
                    x_curr, f_curr = y, fy
                    successes += 1
                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x_curr.copy()

                # 1/5 success rule: adjust sigma every 5 trials
                if trials >= 5:
                    if successes / trials > 0.2:
                        sigma *= 1.25
                    else:
                        sigma *= 0.8
                    successes = 0
                    trials = 0

                sigma = np.clip(sigma, 1e-6, 10.0)

                # If sigma is very small, we've converged locally
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

            # Initialize simplex
            simplex = [x0.copy()]
            # Perturb each dimension
            for i in range(dim):
                xi = x0.copy()
                # Use a small perturbation based on sign to stay in bounds
                perturbation = 0.05 * (1 if xi[i] < 0 else -1)
                xi[i] += perturbation
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

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                if fxr < simplex_f[0]:
                    # Expansion
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
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_func(xc)
                    if not check_time() or (self.budget - calls_made) <= 0:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or (self.budget - calls_made) <= 0:
                                return
                            simplex_f[i] = safe_func(simplex[i])

                # Convergence check
                if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                    break

        # Phase 2: Multi-start (1+1)-ES from top random points
        # We have 80 random points. Let's refine the top 5.
        # Each (1+1)-ES run takes ~60 calls. 5 * 60 = 300 calls.
        # Total so far: 80 + 300 = 380 calls. Remaining: 120 for NM.
        top_k = min(5, len(random_points))
        for i in range(top_k):
            if not check_time() or (self.budget - calls_made) <= 10:
                break
            x_start = random_points[i][1].copy()
            f_start = random_points[i][0]
            # Adjust max_steps based on remaining budget to avoid overshooting
            remaining = self.budget - calls_made
            steps = min(60, remaining // 2) # Reserve some for other starts
            if steps > 0:
                one_plus_one_es(x_start, f_start, max_steps=steps)

        # Phase 3: Nelder-Mead polish on current best
        remaining = self.budget - calls_made
        if remaining > 20 and best_x is not None and check_time():
            # Use most of the remaining budget for NM
            nm_steps = min(50, remaining - 5)
            nelder_mead(best_x, best_f, max_steps=nm_steps)

        # Phase 4: If we still have budget, do another quick (1+1)-ES pass
        remaining = self.budget - calls_made
        if remaining > 10 and best_x is not None and check_time():
            one_plus_one_es(best_x, best_f, max_steps=min(20, remaining - 2))

        # Phase 5: Final NM polish if budget remains
        remaining = self.budget - calls_made
        if remaining > 15 and best_x is not None and check_time():
            nelder_mead(best_x, best_f, max_steps=min(20, remaining - 2))

        return best_x, best_f
# EVOLVE-BLOCK-END
