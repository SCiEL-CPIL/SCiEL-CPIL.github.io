"""Combined optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, multi-start restarts,
    and a final Nelder-Mead polish pass on the best point found."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0  # 25s safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Multi-start (1+1)-ES phase: use ~70% of budget for exploration
        explore_budget = int(budget * 0.70)
        n_starts = 3

        for start_idx in range(n_starts):
            if calls_made >= explore_budget:
                break
            if time.time() > deadline:
                break

            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Don't start a new trajectory if we're about to run out of explore budget
            if calls_made > explore_budget - 30:
                break

            # (1+1)-ES with 1/5 success rule
            sigma = 2.0
            sigma_min = 1e-5
            sigma_max = 10.0
            stuck_count = 0
            max_stuck = 80

            while calls_made < explore_budget:
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    stuck_count = 0
                    sigma = min(sigma * 1.5, sigma_max)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.87, sigma_min)
                    stuck_count += 1

                if stuck_count >= max_stuck:
                    break

        # Ensure we have a valid best point
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # Final Nelder-Mead polish on the best point found
        # Use remaining budget for a high-precision local refinement
        polish_budget = budget - calls_made
        if polish_budget > 10 and time.time() < deadline - 5.0:
            best_x, best_f = self._nelder_mead_polish(
                func, best_x, best_f, dim, polish_budget, deadline, rng
            )

        return best_x, best_f

    def _nelder_mead_polish(self, func, x_best, f_best, dim, budget, deadline, rng):
        """Nelder-Mead simplex method for final high-precision polish."""
        # Initialize simplex: x_best + small perturbations along each axis
        alpha = 0.5  # initial simplex size
        simplex = [x_best.copy()]
        simplex_f = [f_best]

        for i in range(dim):
            pt = x_best.copy()
            pt[i] += alpha
            pt = np.clip(pt, -5, 5)
            simplex.append(pt)
            simplex_f.append(func(pt))

        # Track calls
        calls = dim + 1

        # Nelder-Mead parameters
        r = 1.0   # reflection
        e = 2.0   # expansion
        c = 0.5   # contraction
        s = 0.5   # shrink

        max_iter = budget - dim - 1

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = np.clip(centroid + r * (centroid - simplex[-1]), -5, 5)
            f_r = func(x_r)
            calls += 1

            if f_r < simplex_f[0]:
                # Expansion
                x_e = np.clip(centroid + e * (x_r - centroid), -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < simplex_f[-1]:
                    # Outside contraction
                    x_c = np.clip(centroid + c * (x_r - centroid), -5, 5)
                else:
                    # Inside contraction
                    x_c = np.clip(centroid - c * (centroid - simplex[-1]), -5, 5)
                f_c = func(x_c)
                calls += 1
                if f_c < min(f_r, simplex_f[-1]):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink: move all points toward the best
                    for i in range(1, dim + 1):
                        simplex[i] = np.clip(simplex[0] + s * (simplex[i] - simplex[0]), -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls += 1
                    if calls >= budget:
                        break

            # Update best
            min_idx = np.argmin(simplex_f)
            if simplex_f[min_idx] < f_best:
                f_best = simplex_f[min_idx]
                x_best = simplex[min_idx].copy()

            # Check for convergence
            spread = np.max(simplex_f) - np.min(simplex_f)
            if spread < 1e-10:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
