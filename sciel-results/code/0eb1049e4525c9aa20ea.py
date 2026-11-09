"""Gauss-Newton manifold optimizer for f13 Sharp ridge."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free Gauss-Newton manifold optimizer with Nelder-Mead hybrid."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)
        history_f = []
        history_x = []

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            if calls_made[0] >= budget:
                raise Exception("Budget exceeded")
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            calls_made[0] += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)

        def nelder_mead_step(simplex, f_values):
            """One Nelder-Mead step. Returns new simplex, f_values, and number of evaluations used."""
            nonlocal calls_made, deadline
            n_local = len(simplex[0])
            sorted_idx = np.argsort(f_values)
            simplex = [simplex[i].copy() for i in sorted_idx]
            f_values = [f_values[i] for i in sorted_idx]

            alpha, gamma, rho = 1.0, 2.0, 0.5
            x_worst = simplex[-1]
            centroid = np.mean(simplex[:-1], axis=0)
            evals_used = 0

            x_r = np.clip(centroid + alpha * (centroid - x_worst), -5, 5)
            f_r = wrapped_func(x_r)
            evals_used += 1
            if calls_made[0] >= budget:
                return simplex, f_values, evals_used

            if f_r < f_values[0]:
                x_e = np.clip(centroid + gamma * (centroid - x_worst), -5, 5)
                f_e = wrapped_func(x_e)
                evals_used += 1
                if calls_made[0] >= budget:
                    return simplex, f_values, evals_used
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                if f_r < f_values[-1]:
                    x_c = np.clip(centroid + rho * (x_r - centroid), -5, 5)
                else:
                    x_c = np.clip(centroid - rho * (x_r - centroid), -5, 5)
                f_c = wrapped_func(x_c)
                evals_used += 1
                if calls_made[0] >= budget:
                    return simplex, f_values, evals_used
                if f_c < min(f_r, f_values[-1]):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    sigma_shrink = 0.5
                    x_best = simplex[0]
                    for i in range(1, n_local + 1):
                        simplex[i] = np.clip(x_best + sigma_shrink * (simplex[i] - x_best), -5, 5)
                        f_values[i] = wrapped_func(simplex[i])
                        evals_used += 1
                        if calls_made[0] >= budget:
                            break
            return simplex, f_values, evals_used

        def finite_diff_jacobian(x0, eps=0.01):
            """Estimate Jacobian using central differences."""
            nonlocal calls_made, deadline
            J = np.zeros((n, n))
            for i in range(n):
                if time.time() > deadline or calls_made[0] >= budget:
                    break
                x_plus = x0.copy()
                x_minus = x0.copy()
                x_plus[i] = min(x0[i] + eps, 5.0)
                x_minus[i] = max(x0[i] - eps, -5.0)
                h = x_plus[i] - x_minus[i]
                if h < 1e-10:
                    h = 1e-8
                    x_plus[i] = min(x0[i] + h, 5.0)
                    x_minus[i] = max(x0[i] - h, -5.0)
                fp = wrapped_func(x_plus)
                fm = wrapped_func(x_minus)
                J[:, i] = (fp - fm) / (2 * h)
            return J

        def gauss_newton_manifold_search(x_start, max_iter=30):
            """Estimate low-dim manifold via SVD of Jacobian, search within it."""
            nonlocal calls_made, best_f, best_x, deadline
            n_local = n
            if calls_made[0] >= budget or time.time() > deadline:
                return

            # Initial simplex for NM
            simplex = [x_start.copy()]
            for i in range(n_local):
                pt = x_start.copy()
                step = 0.5
                if x_start[i] + step > 5:
                    pt[i] = x_start[i] - step
                elif x_start[i] - step < -5:
                    pt[i] = x_start[i] + step
                else:
                    pt[i] = x_start[i] + step
                simplex.append(pt)
            f_values = [wrapped_func(pt) for pt in simplex]

            if calls_made[0] >= budget:
                return

            stall = 0
            prev_best = best_f

            for it in range(max_iter):
                if calls_made[0] >= budget or time.time() > deadline:
                    break

                # Check improvement
                if best_f < prev_best - 1e-10:
                    prev_best = best_f
                    stall = 0
                else:
                    stall += 1
                    if stall > 8:
                        break

                # Compute Jacobian at best point
                J = finite_diff_jacobian(best_x)
                if calls_made[0] >= budget:
                    break

                # SVD to find manifold
                U, S, Vt = np.linalg.svd(J, full_matrices=False)
                # Keep top singular vectors (rank-revealing)
                rank = max(1, int(np.sum(S > S[0] * 1e-3))) if len(S) > 0 else 1
                V_manifold = Vt[:rank, :]  # Shape: (rank, n)

                # Project simplex to manifold
                x_best = best_x
                for i in range(len(simplex)):
                    # Project to manifold: x_best + V_manifold.T @ coords
                    delta = simplex[i] - x_best
                    coords = V_manifold @ delta
                    simplex[i] = x_best + V_manifold.T @ coords
                    simplex[i] = np.clip(simplex[i], -5, 5)

                # NM step on manifold-restricted simplex
                simplex, f_values, _ = nelder_mead_step(simplex, f_values)
                if calls_made[0] >= budget:
                    break

                # Re-expand simplex slightly if too small
                simplex_spread = max(np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, n + 1))
                if simplex_spread < 1e-4:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.05 * rng.standard_normal(n)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_values[i] = wrapped_func(simplex[i])
                        if calls_made[0] >= budget:
                            break

        # Multi-start
        start_points = [np.zeros(n)]
        for _ in range(3):
            start_points.append(rng.uniform(-3, 3, n))

        # Phase 1: Nelder-Mead with manifold restriction
        for sp in start_points:
            if calls_made[0] >= budget or time.time() > deadline:
                break
            gauss_newton_manifold_search(sp, max_iter=25)

        # Phase 2: (1+1)-ES polish with adaptive sigma
        if calls_made[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            success_count = 0
            total_trials = 0
            stall = 0

            while calls_made[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = wrapped_func(y)
                total_trials += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    stall = 0
                else:
                    stall += 1

                if total_trials % 10 == 0:
                    rate = success_count / total_trials
                    if rate > 0.2:
                        sigma *= 1.2
                    else:
                        sigma *= 0.85
                    sigma = np.clip(sigma, 1e-6, 3.0)
                    success_count = 0
                    total_trials = 0

                if stall >= 40:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
