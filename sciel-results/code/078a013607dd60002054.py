import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 20.0
    N = 16

    def get_ratio(pts):
        """Calculate d_min / d_max. Higher is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return d_min / d_max

    def get_ratio_sq(pts):
        """Calculate (d_min / d_max)^2. Higher is better. This is the actual metric R."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        """
        Objective for minimization: minimize -log(d_min) + log(d_max)
        This is equivalent to minimizing log(d_max/d_min) = -log(d_min/d_max)
        So minimizing this maximizes d_min/d_max.
        """
        pts = x.reshape(N, 2)
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min <= 1e-15 or d_max <= 1e-15:
            return 1e10
        return np.log(d_max) - np.log(d_min)

    def objective_grad(x):
        """
        Gradient of objective: minimize log(d_max) - log(d_min)
        """
        pts = x.reshape(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        dist = np.sqrt(np.maximum(dist_sq, 1e-30))
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(dist_sq, np.inf)

        d_min = np.min(dist)
        d_max = np.max(dist)

        if d_min <= 1e-15 or d_max <= 1e-15:
            return 1e10, np.zeros_like(x)

        obj_val = np.log(d_max) - np.log(d_min)

        # Find pairs achieving min and max
        # For gradient, we consider all pairs that are close to min/max
        # Use a soft approach: weight by proximity

        # Gradient of log(d_max): sum over pairs of (1/d_max) * (d_ij/d_max) * grad_d_ij
        # Actually, let's compute it properly.
        # d_ij = sqrt(dist_sq_ij)
        # d/dx_k log(d_ij) = (1/d_ij) * d_ij/dx_k = (diff_ik / d_ij) for point i, etc.

        # For log(d_max), the gradient is concentrated on the max pair(s)
        # For log(d_min), the gradient is concentrated on the min pair(s)

        # Use a soft-min/soft-max approach with a small epsilon
        eps = 0.01

        # Soft max weight: w_max_ij = exp(d_ij/eps) / sum(exp(d_ij/eps))
        # Soft min weight: w_min_ij = exp(-d_ij/eps) / sum(exp(-d_ij/eps))

        # To avoid overflow, subtract max
        d_flat = dist[~np.isinf(dist)]
        # Actually, let's just find the argmax and argmin pairs and use them

        # Find all pairs
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_pairs = dist[i_idx, j_idx]

        # Soft weights
        # For max:
        d_shifted = d_pairs - np.max(d_pairs)
        w_max = np.exp(d_shifted / eps)
        w_max /= np.sum(w_max)

        # For min:
        d_shifted_min = -d_pairs + np.max(d_pairs)
        w_min = np.exp(d_shifted_min / eps)
        w_min /= np.sum(w_min)

        grad = np.zeros_like(x)

        # Gradient contribution from log(d_max)
        # d/dx_k log(d_max) ≈ sum_ij w_max_ij * (1/d_max) * d(d_ij)/dx_k
        # d(d_ij)/dx_k for point i: (p_i - p_j) / d_ij
        # d(d_ij)/dx_k for point j: (p_j - p_i) / d_ij

        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            d_ij = dist[i, j]
            if d_ij < 1e-15:
                continue
            diff_ij = pts[i] - pts[j]
            # grad of d_ij w.r.t. pts[i] = diff_ij / d_ij
            # grad of d_ij w.r.t. pts[j] = -diff_ij / d_ij

            # Contribution to log(d_max)
            grad[2*i] += w_max[k] * (diff_ij[0] / d_ij) / d_max
            grad[2*i+1] += w_max[k] * (diff_ij[1] / d_ij) / d_max
            grad[2*j] += w_max[k] * (-diff_ij[0] / d_ij) / d_max
            grad[2*j+1] += w_max[k] * (-diff_ij[1] / d_ij) / d_max

            # Contribution to -log(d_min)
            grad[2*i] -= w_min[k] * (diff_ij[0] / d_ij) / d_min
            grad[2*i+1] -= w_min[k] * (diff_ij[1] / d_ij) / d_min
            grad[2*j] -= w_min[k] * (-diff_ij[0] / d_ij) / d_min
            grad[2*j+1] -= w_min[k] * (-diff_ij[1] / d_ij) / d_min

        return obj_val, grad

    def ring_init(n_inner, n_outer, r_inner, r_outer):
        """Points on two concentric rings."""
        pts = []
        if n_inner > 0:
            for i in range(n_inner):
                angle = 2 * np.pi * i / n_inner
                pts.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
        if n_outer > 0:
            for i in range(n_outer):
                angle = 2 * np.pi * i / n_outer + np.pi / n_outer
                pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
        return np.array(pts)

    def grid_init():
        """4x4 square grid."""
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        pts = np.array(pts, dtype=float)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def hex_init():
        """Hexagonal arrangement."""
        pts = []
        a = 1.0
        b = np.sqrt(3) / 2
        pts.append([0, 0])
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts.append([a * np.cos(angle), a * np.sin(angle)])
        ring2_pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * a + j * 0.5 * a
                y = j * b
                r = np.sqrt(x*x + y*y)
                if 1.5 * a - 0.1 < r < 2.5 * a + 0.1:
                    ring2_pts.append([x, y])
        ring2_pts = np.array(ring2_pts)
        if len(ring2_pts) >= 9:
            r_target = 2.0 * a
            dists_to_target = np.abs(np.linalg.norm(ring2_pts, axis=1) - r_target)
            idx = np.argsort(dists_to_target)[:9]
            ring2_pts = ring2_pts[idx]
        pts = np.array(pts)
        pts = np.vstack([pts, ring2_pts])
        pts = pts - np.mean(pts, axis=0)
        return pts

    def circle_init(n=16):
        """Points on a circle."""
        pts = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            pts.append([np.cos(angle), np.sin(angle)])
        return np.array(pts)

    def square_ring_init():
        """Points on boundary of a square with some interior."""
        pts = []
        for i in range(4):
            pts.append([-1 + 2*i/3, 1.0])
        for i in range(1, 4):
            pts.append([1.0, 1 - 2*i/3])
        for i in range(1, 4):
            pts.append([-1 + 2*i/3, -1.0])
        for i in range(1, 3):
            pts.append([-1.0, 1 - 2*i/3])
        pts.append([0, 0])
        pts.append([0.5, 0.5])
        pts.append([-0.5, 0.5])
        pts.append([0.5, -0.5])
        pts = np.array(pts[:16], dtype=float)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def generate_candidates():
        candidates = []
        candidates.append(ring_init(1, 15, 0, 1.0))
        candidates.append(ring_init(4, 12, 0.5, 1.0))
        candidates.append(ring_init(7, 9, 0.4, 1.0))
        candidates.append(ring_init(9, 7, 0.4, 1.0))
        candidates.append(ring_init(1, 3, 0, 0.5))
        for i, c in enumerate(candidates):
            c = c - np.mean(c, axis=0)
            max_r = np.max(np.linalg.norm(c, axis=1))
            if max_r > 0:
                c = c / max_r
            candidates[i] = c
        candidates.append(grid_init())
        candidates.append(hex_init())
        candidates.append(circle_init(16))
        candidates.append(square_ring_init())
        return candidates

    best_pts = None
    best_ratio = 0.0

    for c in generate_candidates():
        if len(c) != 16:
            continue
        ratio = get_ratio(c)
        if ratio > best_ratio:
            best_ratio = ratio
            best_pts = c.copy()

    # L-BFGS-B optimization with multiple restarts
    def lbfgsb_optimize(pts, time_budget=100.0):
        """Improve points using L-BFGS-B optimization."""
        pts = pts.copy()
        t_start = time.time()

        x0 = pts.flatten()
        best_x = x0.copy()
        best_obj = objective(x0)
        best_ratio = get_ratio(pts)

        # Try multiple random perturbations and optimize from each
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() - t_start > time_budget:
                break
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break

            if r == 0:
                x_start = x0.copy()
            else:
                # Random perturbation
                scale = 0.1
                x_start = x0 + np.random.randn(N * 2) * scale

            try:
                result = minimize(objective, x_start, jac=objective_grad,
                                  method='L-BFGS-B',
                                  options={'maxiter': 200, 'maxls': 50})
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_x = result.x
                    best_ratio = get_ratio(result.x.reshape(N, 2))
            except Exception:
                pass

        return best_x.reshape(N, 2), best_ratio

    # Run L-BFGS-B from best candidate
    time_elapsed = time.time() - start_time
    lbfgsb_budget = min(200.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if lbfgsb_budget > 10:
        lbfgsb_pts, lbfgsb_ratio = lbfgsb_optimize(best_pts, time_budget=lbfgsb_budget)
        if lbfgsb_ratio > best_ratio:
            best_pts = lbfgsb_pts
            best_ratio = lbfgsb_ratio

    # Simulated Annealing for broader search
    def simulated_annealing(pts, time_budget=150.0):
        """Improve points using simulated annealing."""
        pts = pts.copy()
        t_start = time.time()

        T = 0.05
        T_min = 1e-10
        cooling_rate = 0.9998

        current_ratio = get_ratio(pts)
        best_ratio = current_ratio
        best_pts = pts.copy()

        it = 0
        while time.time() - t_start < time_budget and T > T_min:
            it += 1

            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break

            idx = np.random.randint(N)
            scale = 0.05 * T / 0.05
            delta = np.random.randn(2) * scale

            old_pos = pts[idx].copy()
            pts[idx] += delta

            new_ratio = get_ratio(pts)

            if new_ratio > current_ratio:
                current_ratio = new_ratio
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_pts = pts.copy()
            else:
                delta_ratio = new_ratio - current_ratio
                accept_prob = np.exp(delta_ratio / (T * 0.01))
                if np.random.rand() < accept_prob:
                    current_ratio = new_ratio
                else:
                    pts[idx] = old_pos

            T *= cooling_rate

            if it % 500 == 0 and current_ratio == best_ratio:
                T = max(T, T_min * 10)

        return best_pts, best_ratio

    time_elapsed = time.time() - start_time
    sa_budget = min(200.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if sa_budget > 10:
        sa_pts, sa_ratio = simulated_annealing(best_pts, time_budget=sa_budget)
        if sa_ratio > best_ratio:
            best_pts = sa_pts
            best_ratio = sa_ratio

    # Final L-BFGS-B polish
    time_elapsed = time.time() - start_time
    if TIME_LIMIT - time_elapsed - SAFETY_MARGIN > 30:
        polish_budget = min(100.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
        polish_pts, polish_ratio = lbfgsb_optimize(best_pts, time_budget=polish_budget)
        if polish_ratio > best_ratio:
            best_pts = polish_pts
            best_ratio = polish_ratio

    # Ensure finite and correct shape
    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)) or final_pts.shape != (16, 2):
        final_pts = grid_init()

    return final_pts
