# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist


def _soft_min_max(points, beta):
    """Return (soft_min, soft_max) of pairwise distances via log-sum-exp."""
    d = pdist(points)
    # soft-min = - (1/beta) * logsumexp(-beta*d)
    neg = -beta * d
    m = neg.max()
    soft_min = -(m + np.log(np.sum(np.exp(neg - m)))) / beta
    # soft-max = (1/beta) * logsumexp(beta*d)
    pos = beta * d
    m2 = pos.max()
    soft_max = (m2 + np.log(np.sum(np.exp(pos - m2)))) / beta
    return soft_min, soft_max


def _objective(points, beta):
    sm, smax = _soft_min_max(points, beta)
    if smax <= 0:
        return 0.0
    return (sm / smax) ** 2


def _objective_grad(points, beta):
    """Gradient of (soft_min/soft_max)^2 w.r.t. points. Returns (obj, grad)."""
    n = points.shape[0]
    d = pdist(points)
    idx = np.triu_indices(n, k=1)

    # soft-min weights
    neg = -beta * d
    m = neg.max()
    e_neg = np.exp(neg - m)
    s_neg = e_neg.sum()
    w_min = e_neg / s_neg  # weights for soft-min

    # soft-max weights
    pos = beta * d
    m2 = pos.max()
    e_pos = np.exp(pos - m2)
    s_pos = e_pos.sum()
    w_max = e_pos / s_pos  # weights for soft-max

    soft_min = -(m + np.log(s_neg)) / beta
    soft_max = (m2 + np.log(s_pos)) / beta
    if soft_max <= 1e-12:
        return 0.0, np.zeros_like(points)

    ratio = soft_min / soft_max
    obj = ratio ** 2

    # d(obj)/d(soft_min) = 2 * ratio / soft_max
    # d(obj)/d(soft_max) = -2 * ratio * soft_min / soft_max**2
    d_obj_d_sm = 2.0 * ratio / soft_max
    d_obj_d_smax = -2.0 * ratio * soft_min / (soft_max ** 2)

    # d(soft_min)/d(d_k) = -w_min_k
    # d(soft_max)/d(d_k) = +w_max_k
    # So d(obj)/d(d_k) = d_obj_d_sm * (-w_min_k) + d_obj_d_smax * (w_max_k)
    grad_d = -d_obj_d_sm * w_min + d_obj_d_smax * w_max

    # Now backprop to points. For each pair (i,j), d_k = ||p_i - p_j||.
    # d(d_k)/d(p_i) = (p_i - p_j)/d_k, d(d_k)/d(p_j) = (p_j - p_i)/d_k
    grad_points = np.zeros_like(points)
    for k in range(len(d)):
        i, j = idx[0][k], idx[1][k]
        diff = points[i] - points[j]
        dk = d[k]
        if dk < 1e-12:
            continue
        g = grad_d[k] / dk
        grad_points[i] += g * diff
        grad_points[j] -= g * diff

    return obj, grad_points


def _line_search(points, grad, obj_cur, beta, max_iter=50):
    """Backtracking line search."""
    step = 1.0
    shrink = 0.5
    for _ in range(max_iter):
        new_points = points + step * grad
        new_obj, _ = _objective_grad(new_points, beta)
        if np.isfinite(new_obj) and new_obj > obj_cur + 1e-12:
            return new_points, new_obj, step
        step *= shrink
    return points, obj_cur, 0.0


def _fibonacci_sphere(n, scale=1.0):
    """Generate n points on a sphere using the golden spiral."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    points = np.zeros((n, 3))
    for i in range(n):
        z = 1.0 - (i / (n - 1)) * 2.0  # from 1 to -1
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        points[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return points * scale


def _critical_pair_perturb(points, step=0.05):
    """Perturb closest pair apart and farthest pair together."""
    d = pdist(points)
    idx = np.triu_indices(14, k=1)
    min_k = np.argmin(d)
    max_k = np.argmax(d)
    i, j = idx[0][min_k], idx[1][min_k]
    diff = points[i] - points[j]
    norm = np.linalg.norm(diff)
    if norm > 1e-12:
        direction = diff / norm
        points[i] += step * direction
        points[j] -= step * direction
    i2, j2 = idx[0][max_k], idx[1][max_k]
    diff2 = points[i2] - points[j2]
    norm2 = np.linalg.norm(diff2)
    if norm2 > 1e-12:
        direction2 = diff2 / norm2
        points[i2] -= step * direction2
        points[j2] += step * direction2
    return points


def _polish(points, beta, time_budget):
    """High-precision polish with SLSQP-like gradient ascent."""
    start = time.time()
    obj, grad = _objective_grad(points, beta)
    max_iter = 200
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        new_points, new_obj, step = _line_search(points, grad, obj, beta)
        if step == 0.0:
            break
        points = new_points
        obj = new_obj
        obj, grad = _objective_grad(points, beta)
    return points, obj


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.

    Returns
        points: np.ndarray of shape (14,3) containing the coordinates.
    """
    t0 = time.time()
    deadline = 620.0  # safety margin before 630s kill

    n = 14
    d = 3

    # Multiple initializations
    inits = []
    # Fibonacci sphere
    inits.append(_fibonacci_sphere(n, scale=1.0))
    # Random with different seeds
    for seed in [0, 1, 2, 3, 4]:
        rng = np.random.RandomState(seed)
        inits.append(rng.randn(n, d) * 1.0)

    best_points = None
    best_score = -1.0

    # Phase 1: Coarse optimization with low beta (smooth)
    betas_coarse = [5.0, 10.0, 20.0, 50.0]
    for beta in betas_coarse:
        if time.time() - t0 > deadline - 60:
            break
        for init in inits:
            if time.time() - t0 > deadline - 30:
                break
            points = init.copy()
            obj, grad = _objective_grad(points, beta)
            # Gradient ascent with line search
            for it in range(300):
                if time.time() - t0 > deadline - 30:
                    break
                new_points, new_obj, step = _line_search(points, grad, obj, beta)
                if step == 0.0:
                    break
                points = new_points
                obj = new_obj
                obj, grad = _objective_grad(points, beta)
            # Evaluate true objective
            d_arr = pdist(points)
            if d_arr.max() > 1e-12:
                score = (d_arr.min() / d_arr.max()) ** 2
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_points = points.copy()
            # Update init for next beta
            inits[0] = points.copy()

    # Phase 2: Critical pair perturbation + re-optimize
    if best_points is not None and time.time() - t0 < deadline - 40:
        for perturb_step in [0.1, 0.05, 0.02]:
            if time.time() - t0 > deadline - 20:
                break
            perturbed = _critical_pair_perturb(best_points.copy(), perturb_step)
            obj, grad = _objective_grad(perturbed, 50.0)
            for it in range(100):
                if time.time() - t0 > deadline - 20:
                    break
                new_points, new_obj, step = _line_search(perturbed, grad, obj, 50.0)
                if step == 0.0:
                    break
                perturbed = new_points
                obj = new_obj
                obj, grad = _objective_grad(perturbed, 50.0)
            d_arr = pdist(perturbed)
            if d_arr.max() > 1e-12:
                score = (d_arr.min() / d_arr.max()) ** 2
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_points = perturbed.copy()

    # Phase 3: High-precision polish with high beta
    if best_points is not None and time.time() - t0 < deadline - 10:
        for beta in [100.0, 200.0, 500.0, 1000.0]:
            if time.time() - t0 > deadline - 5:
                break
            remaining = deadline - 5 - (time.time() - t0)
            if remaining < 2:
                break
            polished, _ = _polish(best_points.copy(), beta, time_budget=remaining)
            d_arr = pdist(polished)
            if d_arr.max() > 1e-12:
                score = (d_arr.min() / d_arr.max()) ** 2
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_points = polished.copy()

    # Final validation
    if best_points is None:
        return np.random.randn(n, d)
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        return np.random.randn(n, d)
    return best_points


# EVOLVE-BLOCK-END
