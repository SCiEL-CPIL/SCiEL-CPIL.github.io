"""
Circle packing in unit square using Delaunay-based tight packing.
Centers are optimized; radii are derived from the local geometry
(half the minimum distance to any other center or to a boundary).
This enforces tight packing and reduces the effective search space.
"""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 115.0  # Safety margin from 120s limit (evaluator kills at 150s, but keep margin)


def construct():
    start_time = time.time()

    def compute_radii(x, y):
        """Compute radii from centers: min of half-distance to neighbors and boundary."""
        # Boundary distances
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        # Minimum distance to any other circle
        min_dist = np.min(dist, axis=1)

        # Radius is half the minimum of (distance to neighbor, 2*boundary distance)
        # Actually, for tight packing: r_i = min(0.5 * min_dist_to_neighbor, boundary_dist_i)
        # But we need to ensure r_i + r_j <= dist_ij for all pairs.
        # Using r_i = min(boundary_i, 0.5 * min_dist_i) is a good heuristic.
        r = np.minimum(r_bound, 0.5 * min_dist)
        # Ensure positive
        r = np.maximum(r, 1e-6)
        return r

    def check_validity(x, y, r):
        # Check boundary
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        if np.any(r < -1e-9):
            return False

        # Check non-overlap
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)

        if np.any(dist < min_dist - 1e-9):
            return False
        return True

    def objective_and_grad(params):
        """
        Objective: minimize -sum(radii) where radii are computed from centers.
        We use a smooth approximation.
        """
        x = params[:N]
        y = params[N:2*N]

        # Compute radii
        r = compute_radii(x, y)

        # For gradient, we need to be careful. Let's use finite differences
        # or derive analytically. The radius function is:
        # r_i = min(boundary_i, 0.5 * min_j dist_ij)
        # This is non-smooth. We'll use a smooth approximation.

        # Actually, let's use a different approach:
        # Define a smooth penalty that encourages large radii while maintaining feasibility.
        # We'll use the actual radii in the objective and use numerical gradients
        # via the optimizer's finite difference, but provide a smooth objective.

        # Better approach: use a smooth "effective radius" function.
        # For each circle, r_i = min(boundary_i, 0.5 * min_dist_i)
        # We can smooth the min using a soft-min.

        # Let's compute a smooth version:
        # r_i_smooth = -alpha * logsumexp(-distances/alpha) for the neighbor part
        # and soft-min with boundary.

        alpha = 0.05  # Softness parameter

        # Boundary part
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        # Neighbor part: for each i, min over j != i of 0.5 * dist_ij
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        # Soft min of 0.5 * dist
        # logsumexp trick: min_j a_j ≈ -alpha * log(sum_j exp(-a_j/alpha))
        # Here a_ij = 0.5 * dist_ij
        a = 0.5 * dist  # shape (N, N), diagonal is inf
        # For each row i, compute soft min over j
        # exp(-a/alpha) will be 0 for inf entries
        exp_neg_a = np.exp(-a / alpha)
        # Replace inf entries with 0
        exp_neg_a = np.where(np.isinf(a), 0.0, exp_neg_a)
        sum_exp = np.sum(exp_neg_a, axis=1)
        # Avoid log(0)
        sum_exp = np.maximum(sum_exp, 1e-300)
        r_neighbor = -alpha * np.log(sum_exp)

        # Soft min of r_bound and r_neighbor
        # min(a, b) ≈ -alpha * log(exp(-a/alpha) + exp(-b/alpha))
        exp_neg_rb = np.exp(-r_bound / alpha)
        exp_neg_rn = np.exp(-r_neighbor / alpha)
        r_smooth = -alpha * np.log(exp_neg_rb + exp_neg_rn)

        obj = -np.sum(r_smooth)
        return obj

    def make_initial_guess(seed):
        rng = np.random.RandomState(seed)
        # Start with a grid and perturb
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x[idx] = (i + 0.5) / side
                y[idx] = (j + 0.5) / side
                idx += 1

        # Perturb
        x += rng.uniform(-0.1, 0.1, N)
        y += rng.uniform(-0.1, 0.1, N)

        # Clip to valid range
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)

        return x, y

    def run_optimization(seed):
        x0, y0 = make_initial_guess(seed)
        params0 = np.concatenate([x0, y0])

        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))  # x
        for i in range(N):
            bounds.append((0.01, 0.99))  # y

        options = {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}

        try:
            res = minimize(objective_and_grad, params0, method='L-BFGS-B',
                           bounds=bounds, options=options)

            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = compute_radii(x_opt, y_opt)

            # Check validity
            if check_validity(x_opt, y_opt, r_opt):
                val = np.sum(r_opt)
                return x_opt, y_opt, r_opt, val
            else:
                # Try to repair by shrinking
                for scale in [0.99, 0.98, 0.95, 0.90]:
                    r_test = r_opt * scale
                    if check_validity(x_opt, y_opt, r_test):
                        return x_opt, y_opt, r_test, np.sum(r_test)
            return None
        except Exception:
            return None

    # Initial best: grid configuration
    side = int(np.ceil(np.sqrt(N)))
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            best_x[idx] = (i + 0.5) / side
            best_y[idx] = (j + 0.5) / side
            idx += 1
    best_r = compute_radii(best_x, best_y)
    best_val = np.sum(best_r)

    # Multi-start loop
    num_starts = 30
    for i in range(num_starts):
        if time.time() > DEADLINE:
            break

        res = run_optimization(i)
        if res is not None:
            x_opt, y_opt, r_opt, val = res
            if val > best_val:
                best_val = val
                best_x = x_opt
                best_y = y_opt
                best_r = r_opt

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.90, 10):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            # Fallback to grid
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
