"""Improved BBOB f22 optimizer: Multi-Start Nelder-Mead with final polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 285.0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(7)
        n = self.dim

        def lhs_sample(n_samples, dim, rng):
            samples = np.zeros((n_samples, dim))
            for i in range(dim):
                perm = rng.permutation(n_samples)
                u = (perm + 0.5) / n_samples
                samples[:, i] = -5 + 10 * u
            return samples

        def nelder_mead(x_start, f_start, max_steps=50, initial_delta=0.5):
            nonlocal calls_made, best_x, best_f
            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5
            self._prev_best_in_nm = np.inf

            simplex = [x_start.copy()]
            for i in range(dim):
                xi = x_start.copy()
                # Use a small positive delta to ensure distinct vertices
                # and stay within bounds if possible
                xi[i] += initial_delta
                xi = np.clip(xi, -5, 5)
                # Ensure distinctness if clipping caused issues
                if np.allclose(xi, x_start):
                    xi[i] = x_start[i] + initial_delta * 0.1
                    xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex_f = [f_start]
            for i in range(1, dim + 1):
                if not check_time() or calls_made >= self.budget:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)

            for step in range(max_steps):
                if not check_time() or calls_made >= self.budget:
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
                if not check_time() or calls_made >= self.budget:
                    break

                if fxr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    if not check_time() or calls_made >= self.budget:
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
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or calls_made >= self.budget:
                                return
                            simplex_f[i] = safe_func(simplex[i])

                # Adaptive convergence check: stop when progress is negligible
                current_spread = np.max(simplex_f) - np.min(simplex_f)
                if current_spread < 1e-12:
                    break
                # Also stop if we've made very little progress in recent steps
                if step > 5:
                    recent_progress = abs(simplex_f[0] - self._prev_best_in_nm) if hasattr(self, '_prev_best_in_nm') else np.inf
                    if recent_progress < 1e-10:
                        break
                if hasattr(self, '_prev_best_in_nm'):
                    del self._prev_best_in_nm
                self._prev_best_in_nm = simplex_f[0]

        # Phase 1: LHS Exploration with more points for better coverage
        n_lhs = 60
        points = []
        if calls_made + n_lhs <= self.budget:
            lhs_points = lhs_sample(n_lhs, n, rng)
            for x in lhs_points:
                if not check_time() or calls_made >= self.budget:
                    break
                fx = safe_func(x)
                points.append((fx, x.copy()))

        # Add a few random points for diversity
        n_random = 10
        if calls_made + n_random <= self.budget:
            for _ in range(n_random):
                if not check_time() or calls_made >= self.budget:
                    break
                x_rand = rng.uniform(-5, 5, size=n)
                fx = safe_func(x_rand)
                points.append((fx, x_rand.copy()))

        points.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead on top K with adaptive budget allocation
        top_k = 6
        remaining = self.budget - calls_made
        # Allocate more budget to earlier (better) starts
        if top_k > 0 and remaining > 0:
            # Give 50% of budget to the best start, split rest among others
            budget_best = int(remaining * 0.5)
            budget_others = remaining - budget_best
            nm_budget_per_start_other = max(10, budget_others // max(1, top_k - 1))
            
            # Run NM on the best point first with more budget
            x_start = points[0][1]
            f_start = points[0][0]
            nelder_mead(x_start, f_start, max_steps=budget_best, initial_delta=0.5)
            
            # Run NM on remaining top points
            for i in range(1, min(top_k, len(points))):
                if not check_time() or calls_made >= self.budget:
                    break
                x_start = points[i][1]
                f_start = points[i][0]
                nelder_mead(x_start, f_start, max_steps=nm_budget_per_start_other, initial_delta=0.5)
        else:
            nm_budget_per_start = max(15, remaining // (top_k + 1))
            for i in range(min(top_k, len(points))):
                if not check_time() or calls_made >= self.budget:
                    break
                x_start = points[i][1]
                f_start = points[i][0]
                nelder_mead(x_start, f_start, max_steps=nm_budget_per_start, initial_delta=0.5)

        # Phase 3: Final high-precision polish on the best point with very small delta
        if check_time() and calls_made < self.budget and best_f < np.inf:
            # Use a much smaller initial delta for fine-tuning
            nelder_mead(best_x.copy(), best_f, max_steps=40, initial_delta=0.05)
            
            # One more ultra-fine polish if budget allows
            if check_time() and calls_made < self.budget:
                nelder_mead(best_x.copy(), best_f, max_steps=20, initial_delta=0.01)

        return best_x, best_f

# EVOLVE-BLOCK-END
