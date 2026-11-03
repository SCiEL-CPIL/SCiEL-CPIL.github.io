"""Improved packing: ring-based initialization + smooth annealing with constraint push."""
import numpy as np
import time

N = 50
DEADLINE = 115.0  # seconds, safety margin before 120s kill


def _ring_init(n):
    """Generate n points on concentric rings, evenly spaced."""
    if n <= 0:
        return np.empty((0, 2))
    if n == 1:
        return np.array([[0.0, 0.0]])
    
    # Use a heuristic: place points on rings with roughly equal spacing
    # Start with a hexagonal-like distribution
    points = [np.array([0.0, 0.0])]
    
    # Try different ring configurations
    # For 50 points: 1 center + rings of 6, 12, 18, ... or similar
    # Let's use a more systematic approach
    
    # Place points on rings with increasing radius
    remaining = n - 1
    ring_idx = 1
    while remaining > 0:
        # How many points on this ring?
        # Approximate: circumference / desired spacing
        # Desired spacing ~ 0.25 for initial guess
        circumference = 2 * np.pi * ring_idx * 0.25
        count = max(1, int(circumference / 0.22))
        count = min(count, remaining)
        
        angles = 2 * np.pi * np.arange(count) / count + ring_idx * 0.1  # offset for hex-like
        r = ring_idx * 0.22
        for a in angles:
            points.append(np.array([r * np.cos(a), r * np.sin(a)]))
        remaining -= count
        ring_idx += 1
    
    points = np.array(points[:n])
    
    # Normalize to fit in unit circle
    max_r = np.max(np.hypot(points[:, 0], points[:, 1]))
    if max_r > 0.95:
        points *= 0.95 / max_r
    
    return points


def _score(points):
    """Compute the packing score: min of half-min-pairwise-dist and min-boundary-dist."""
    n = len(points)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distances
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(1.0 - radii)
    
    return min(min_pair, min_boundary)


def _smooth_objective(points, beta):
    """
    Smooth approximation of the negative score.
    We want to maximize r = min(min_pair/2, min_boundary).
    Equivalently, minimize -r.
    
    Use soft-min: -softmin(dists/2, boundary_dists, beta)
    where softmin(x, beta) = -log(sum(exp(-beta*x)))/beta
    """
    n = len(points)
    
    # Pairwise distances / 2
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    radii = np.hypot(points[:, 0], points[:, 1])
    boundary = 1.0 - radii
    
    # Combine all constraints
    all_dists = np.concatenate([dists[np.triu_indices(n, k=1)], boundary])
    
    # Soft min
    # To avoid overflow, subtract max
    m = np.max(all_dists)
    exp_terms = np.exp(-beta * (all_dists - m))
    softmin_val = m - np.log(np.sum(exp_terms)) / beta
    
    # We want to maximize softmin_val, so minimize -softmin_val
    return -softmin_val


def _gradient(points, beta):
    """
    Compute gradient of the smooth objective w.r.t. points.
    Using automatic differentiation via finite differences would be too slow.
    Instead, use analytical gradient of the soft-min.
    
    Actually, let's use a simpler approach: gradient descent with finite differences
    for the gradient, or use the analytical form.
    
    For softmin of all_dists where all_dists are functions of points,
    the gradient is a weighted sum of gradients of each dist.
    
    Let's compute it properly.
    """
    n = len(points)
    
    # Pairwise distances / 2
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    radii = np.hypot(points[:, 0], points[:, 1])
    boundary = 1.0 - radii
    
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    
    # Combine
    all_dists = np.concatenate([pair_dists, boundary])
    
    # Soft-min weights
    m = np.max(all_dists)
    exp_terms = np.exp(-beta * (all_dists - m))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp  # These are the softmax weights
    
    # Gradient of softmin w.r.t. each dist
    # d(softmin)/d(dist_k) = -beta * w_k * (dist_k - softmin_val)
    # But we need gradient w.r.t. points.
    
    # Actually, let's just use finite differences for simplicity and reliability
    # with a small step.
    
    eps = 1e-6
    grad = np.zeros_like(points)
    
    for i in range(n):
        for j in range(2):
            pts_plus = points.copy()
            pts_plus[i, j] += eps
            f_plus = _smooth_objective(pts_plus, beta)
            
            pts_minus = points.copy()
            pts_minus[i, j] -= eps
            f_minus = _smooth_objective(pts_minus, beta)
            
            grad[i, j] = (f_plus - f_minus) / (2 * eps)
    
    return grad


def _adam_step(points, grad, m, v, t, lr, b1, b2, eps):
    """Adam optimizer step."""
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * (grad ** 2)
    
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    
    points = points - lr * m_hat / (np.sqrt(v_hat) + eps)
    return points, m, v


def construct():
    start_time = time.time()
    
    # Initial points
    points = _ring_init(N).astype(np.float64)
    
    # Adam optimizer state
    m = np.zeros_like(points)
    v = np.zeros_like(points)
    t = 0
    lr = 0.01
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    
    best_points = points.copy()
    best_score = _score(points)
    
    # Multi-stage annealing: start with low beta (smooth), increase to high beta (sharp)
    betas = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    for beta in betas:
        # Check time
        if time.time() - start_time > DEADLINE:
            break
        
        # For each beta, do several Adam steps
        n_steps = 200
        for step in range(n_steps):
            if time.time() - start_time > DEADLINE:
                break
            
            t += 1
            grad = _gradient(points, beta)
            points, m, v = _adam_step(points, grad, m, v, t, lr, b1, b2, eps)
            
            # Project back to unit circle
            radii = np.hypot(points[:, 0], points[:, 1])
            for i in range(len(points)):
                if radii[i] > 0.999:
                    points[i] *= 0.999 / radii[i]
            
            # Track best
            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
        
        # Decay learning rate
        lr *= 0.5
    
    # Final polish: try to push points apart using constraint-based approach
    if time.time() - start_time < DEADLINE:
        # Simple local search: try to increase minimum distance
        for _ in range(50):
            if time.time() - start_time > DEADLINE:
                break
            
            # Find the most constrained pair
            diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = min_idx[0], min_idx[1]
            if i >= j:
                i, j = j, i
            
            # Push i and j apart
            direction = points[i] - points[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
                shift = 0.001
                points[i] += shift * direction
                points[j] -= shift * direction
            
            # Project to unit circle
            radii = np.hypot(points[:, 0], points[:, 1])
            for k in range(len(points)):
                if radii[k] > 0.999:
                    points[k] *= 0.999 / radii[k]
            
            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    # Ensure validity: no point outside unit circle
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    for i in range(len(best_points)):
        if radii[i] > 0.9999:
            best_points[i] *= 0.9999 / radii[i]
    
    return best_points
