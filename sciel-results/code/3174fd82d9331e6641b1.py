"""Pack 300 equal circles in the unit circle using hexagonal lattice init + vectorized gradient descent."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 300
DEADLINE = 115.0  # safety margin before 150s kill


def _compute_score(points):
    """Compute the packing score (radius) for given points."""
    n = len(points)
    if n < 2:
        return 0.0
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_dist = np.min(d)
    norms = np.hypot(points[:, 0], points[:, 1])
    max_norm = np.max(norms)
    boundary_r = 1.0 - max_norm
    return min(min_dist / 2.0, boundary_r)


def _hex_lattice_init(n_points):
    """Generate hexagonal lattice packing inside the unit disk, return best subset."""
    best_points = None
    best_score = -1

    for scale in np.linspace(0.09, 0.14, 80):
        dx = scale
        dy = scale * np.sqrt(3) / 2

        points = []
        n_rows = int(2.0 / dy) + 3
        n_cols = int(2.0 / dx) + 3

        for j in range(-n_rows, n_rows + 1):
            for i in range(-n_cols, n_cols + 1):
                x = i * dx + (0.5 * dx if j % 2 != 0 else 0.0)
                y = j * dy
                if x * x + y * y <= 1.0:
                    points.append([x, y])

        points = np.array(points)
        if len(points) < n_points:
            continue

        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')[:n_points]
        pts = points[idx]

        est_r = _compute_score(pts)
        if est_r > best_score:
            best_score = est_r
            best_points = pts

    if best_points is None:
        best_points = np.random.randn(n_points, 2) * 0.5
        best_score = 0.0

    return best_points, best_score


def _vectorized_gradient(pts, alpha):
    """
    Compute gradient of -soft_min_distance + boundary_penalty.
    soft_min = min_d - (1/alpha)*log(sum(exp(-alpha*(d - min_d))))
    This approximates the minimum pairwise distance smoothly.
    """
    n = len(pts)
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)

    d_flat = d[d < np.inf]
    if len(d_flat) == 0:
        return np.zeros_like(pts)

    d_min = np.min(d_flat)
    # Soft min value
    log_sum = np.log(np.sum(np.exp(-alpha * (d_flat - d_min))))
    soft_min_val = d_min - log_sum / alpha

    # Weights: w_ij = exp(-alpha*(d_ij - d_min)) / sum_j exp(-alpha*(d_ij - d_min))
    exp_terms = np.exp(-alpha * (d - d_min))
    np.fill_diagonal(exp_terms, 0.0)
    w_sum = np.sum(exp_terms)
    if w_sum < 1e-30:
        return np.zeros_like(pts)

    # Gradient of soft_min w.r.t. pts[i]:
    # d(soft_min)/d(pts[i]) = sum_j w_ij * (pts[i] - pts[j]) / d_ij
    # where w_ij = exp_terms[i,j] / w_sum
    #
    # We need: grad[i] = sum_j (exp_terms[i,j] / w_sum) * (pts[i] - pts[j]) / d_ij
    #
    # Vectorized: for each i, sum over j of w_ij * (pts[i]-pts[j])/d_ij
    # = pts[i] * sum_j w_ij/d_ij - sum_j w_ij * pts[j] / d_ij

    # Build weight matrix: W[i,j] = exp_terms[i,j] / (w_sum * d_ij)
    d_safe = np.where(d > 1e-12, d, 1.0)
    W = exp_terms / (w_sum * d_safe)
    np.fill_diagonal(W, 0.0)

    # grad_pts[i] = sum_j W[i,j] * (pts[i] - pts[j])
    #             = pts[i] * rowsum(W) - W @ pts
    rowsum = W.sum(axis=1, keepdims=True)
    grad_soft = pts * rowsum - W @ pts

    # Objective = -soft_min + boundary_penalty
    # grad of -soft_min = -grad_soft
    grad_obj = -grad_soft

    # Boundary penalty: sum max(0, ||p_i|| - (1 - soft_min/2))^2
    # For simplicity, use a fixed radius estimate for boundary
    r_est = soft_min_val / 2.0
    norms = np.hypot(pts[:, 0], pts[:, 1])
    violation = np.maximum(0.0, norms - (1.0 - r_est))
    mask = violation > 0
    if np.any(mask):
        # d/dp_i of max(0, ||p_i|| - R)^2 = 2*violation * p_i / ||p_i||
        norms_safe = np.where(norms > 1e-12, norms, 1.0)
        grad_boundary = np.zeros_like(pts)
        grad_boundary[mask] = 2.0 * violation[mask, np.newaxis] * pts[mask] / norms_safe[mask, np.newaxis]
        grad_obj += 10.0 * grad_boundary

    return grad_obj


def _gradient_descent(pts, start_time, max_iter=3000, lr_init=0.001, alpha=30.0):
    """
    Gradient descent with adaptive learning rate to maximize the packing radius.
    Uses the smooth surrogate: maximize soft_min_distance subject to boundary.
    """
    n = len(pts)
    best_pts = pts.copy()
    best_r = _compute_score(pts)
    lr = lr_init
    no_improve = 0
    patience = 200

    for it in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5.0:
            break

        grad = _vectorized_gradient(pts, alpha)
        pts = pts - lr * grad

        # Project back into unit disk with margin
        norms = np.hypot(pts[:, 0], pts[:, 1])
        # Estimate current radius
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        cur_r = min(np.min(d) / 2.0, 1.0 - np.max(norms))
        max_norm = 1.0 - cur_r - 1e-6
        mask = norms > max_norm
        if np.any(mask):
            norms_safe = np.where(norms[mask] > 1e-12, norms[mask], 1.0)
            pts[mask] = pts[mask] * (max_norm / norms_safe)[:, np.newaxis]

        # Compute score periodically
        if it % 50 == 0 or it == max_iter - 1:
            r = _compute_score(pts)
            if r > best_r + 1e-8:
                best_r = r
                best_pts = pts.copy()
                no_improve = 0
                # Slightly increase lr if improving
                lr *= 1.05
            else:
                no_improve += 50
                if no_improve >= patience:
                    lr *= 0.7
                    no_improve = 0

        # Adaptive lr
        if it % 100 == 0:
            lr = max(lr * 0.95, 1e-6)

    return best_pts, best_r


def _lbfgs_refine(pts, start_time):
    """Use L-BFGS-B with a smooth objective for final polish."""
    from scipy.optimize import minimize

    alpha = 50.0

    def obj(x):
        p = x.reshape(-1, 2)
        d = cdist(p, p)
        np.fill_diagonal(d, np.inf)
        d_flat = d[d < np.inf]
        if len(d_flat) == 0:
            return 0.0
        d_min = np.min(d_flat)
        log_sum = np.log(np.sum(np.exp(-alpha * (d_flat - d_min))))
        soft_min = d_min - log_sum / alpha
        norms = np.hypot(p[:, 0], p[:, 1])
        bv = np.maximum(0.0, norms - (1.0 - soft_min / 2.0))
        return -soft_min + 10.0 * np.sum(bv ** 2)

    def grad(x):
        p = x.reshape(-1, 2)
        g = _vectorized_gradient(p, alpha)
        return g.flatten()

    x0 = pts.flatten()
    bounds = [(-1.0, 1.0)] * (2 * N)

    try:
        result = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 500}
        )
        if result.fun < obj(x0):
            cand = result.x.reshape(-1, 2)
            r = _compute_score(cand)
            return cand, r
    except Exception:
        pass

    return pts, _compute_score(pts)


def construct():
    start_time = time.time()

    # Step 1: Hexagonal lattice initialization
    points, init_r = _hex_lattice_init(N)
    best_pts = points.copy()
    best_r = init_r

    # Step 2: Gradient descent refinement (vectorized, fast)
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 30:
        gd_pts, gd_r = _gradient_descent(best_pts, start_time, max_iter=2000, lr_init=0.0005, alpha=25.0)
        if gd_r > best_r:
            best_pts = gd_pts
            best_r = gd_r

    # Step 3: Second gradient descent pass with finer settings
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 25:
        gd2_pts, gd2_r = _gradient_descent(best_pts, start_time, max_iter=1500, lr_init=0.0002, alpha=50.0)
        if gd2_r > best_r:
            best_pts = gd2_pts
            best_r = gd2_r

    # Step 4: L-BFGS-B polish
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 20:
        lb_pts, lb_r = _lbfgs_refine(best_pts, start_time)
        if lb_r > best_r:
            best_pts = lb_pts
            best_r = lb_r

    # Final: clamp to ensure validity
    norms = np.hypot(best_pts[:, 0], best_pts[:, 1])
    max_allowed = 1.0 - best_r
    mask = norms > max_allowed
    if np.any(mask):
        norms_safe = np.where(norms[mask] > 1e-12, norms[mask], 1.0)
        best_pts[mask] = best_pts[mask] * (max_allowed / norms_safe)[:, np.newaxis]

    return best_pts


if __name__ == "__main__":
    start = time.time()
    points = construct()
    elapsed = time.time() - start
    print(f"Shape: {points.shape}")
    r = _compute_score(points)
    print(f"Radius: {r}")
    print(f"Score: {r / 0.0531542991469}")
    print(f"Time: {elapsed:.2f}s")
