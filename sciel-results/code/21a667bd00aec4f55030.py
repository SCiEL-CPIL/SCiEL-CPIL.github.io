"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_init(n, spacing=None):
    """Generate initial hexagonal lattice positions inside unit disk."""
    if spacing is None:
        spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= n:
            # Sort by radius to prefer central points, take first n
            return points[np.argsort(radius, kind="stable")[:n]]
        spacing *= 0.97


def _compute_radius(pts):
    """Compute the minimum radius for a given set of points."""
    n = len(pts)
    if n < 2:
        return np.min(1.0 - np.linalg.norm(pts, axis=1))
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
    
    return min(min_pair_dist / 2.0, min_boundary)


def _score(pts):
    """Score = the common radius."""
    return _compute_radius(pts)


def _soft_min(x, k):
    """Soft minimum using log-sum-exp trick for stability."""
    # x is array of values, k is sharpness
    if k <= 0:
        return np.min(x)
    # Use the stable log-sum-exp form
    m = np.min(x)
    # For soft min: min(x) ≈ -1/k * log(sum(exp(-k*x)))
    # = -1/k * log(sum(exp(-k*(x - m)))) + m
    # = m - 1/k * log(sum(exp(-k*(x - m))))
    shifted = x - m
    log_sum = np.log(np.sum(np.exp(-k * shifted)))
    return m - log_sum / k


def _optimize(pts, time_budget, initial_lr=0.01):
    """Optimize the packing using gradient ascent on the soft-min radius."""
    n = len(pts)
    pts = pts.copy()
    
    # Precompute which pairs are close (for efficiency)
    # We'll use a full pairwise approach but with soft-min
    
    def soft_min_radius(pts, k):
        """Compute soft-min radius and its gradient."""
        # Boundary distances: 1 - |c_i|
        norms = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - norms
        
        # Pairwise distances / 2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_radii = dists / 2.0
        
        # Soft min over all constraints
        # We have n boundary constraints and n*(n-1)/2 pair constraints
        # For efficiency, compute soft min separately and combine
        k_boundary = k
        k_pair = k
        
        # Soft min of boundary
        m_b = np.min(boundary)
        log_sum_b = np.log(np.sum(np.exp(-k_boundary * (boundary - m_b))))
        soft_boundary = m_b - log_sum_b / k_boundary
        
        # Gradient of soft min boundary w.r.t. pts
        # d(soft_min)/d(pts_i) = -1/k * sum_j exp(-k*(x_j - m)) * (-k * dx_j/d(pts_i))
        # For boundary: x_j = 1 - |c_j|, dx_j/d(c_i) = -c_j/|c_j| if i==j, 0 otherwise
        weights_b = np.exp(-k_boundary * (boundary - m_b))
        weights_b = weights_b / np.sum(weights_b)  # normalized
        # gradient_i = -weights_b[i] * (-c_i / |c_i|) = weights_b[i] * c_i / |c_i|
        # Wait, let's be more careful.
        # soft_min = m - 1/k * log(sum(exp(-k*(x_j - m))))
        # d(soft_min)/d(x_i) = -1/k * d/d(x_i) [log(sum(exp(-k*(x_j - m))))]
        # = -1/k * [sum_j exp(-k*(x_j-m)) * (-k) * d(x_j)/d(x_i)] / sum_j exp(-k*(x_j-m))
        # = sum_j w_j * d(x_j)/d(x_i) where w_j = exp(-k*(x_j-m))/sum(...)
        # For boundary: x_j = 1 - |c_j|, d(x_j)/d(c_i) = -c_j/|c_j| if i==j
        # So d(soft_min)/d(c_i) = w_i * (-c_i/|c_i|)
        # But we want to MAXIMIZE soft_min, so gradient ascent direction is this.
        # Actually, we want to increase the minimum, so we move in the direction of increasing soft_min.
        # d(soft_min)/d(c_i) = -w_i * c_i / |c_i|
        # So the gradient for c_i is -w_i * c_i / |c_i|
        
        grad_boundary = np.zeros_like(pts)
        for i in range(n):
            if norms[i] > 1e-10:
                grad_boundary[i] = -weights_b[i] * pts[i] / norms[i]
            else:
                # If at origin, push outward slightly
                grad_boundary[i] = -weights_b[i] * np.array([1.0, 0.0])
        
        # Soft min of pair radii
        # This is expensive for n=300. Let's use a sparse approach.
        # Only consider pairs that are close.
        # For now, use a threshold.
        threshold = 0.15  # only consider pairs within this distance
        close_mask = (dists < threshold) & (dists > 1e-10)
        
        # Get close pairs
        close_pairs = np.argwhere(close_mask)
        close_pairs = close_pairs[close_pairs[:, 0] < close_pairs[:, 1]]  # avoid duplicates
        
        if len(close_pairs) == 0:
            soft_pair = np.inf
            grad_pair = np.zeros_like(pts)
        else:
            pair_vals = pair_radii[close_pairs[:, 0], close_pairs[:, 1]]
            m_p = np.min(pair_vals)
            log_sum_p = np.log(np.sum(np.exp(-k_pair * (pair_vals - m_p))))
            soft_pair = m_p - log_sum_p / k_pair
            
            weights_p = np.exp(-k_pair * (pair_vals - m_p))
            weights_p = weights_p / np.sum(weights_p)
            
            grad_pair = np.zeros_like(pts)
            for idx, (i, j) in enumerate(close_pairs):
                w = weights_p[idx]
                if w < 1e-10:
                    continue
                diff_ij = pts[i] - pts[j]
                dist_ij = np.linalg.norm(diff_ij)
                if dist_ij < 1e-10:
                    continue
                # d(pair_radius)/d(c_i) = (c_i - c_j) / (2 * dist_ij)
                # d(soft_min)/d(c_i) = w * d(pair_radius)/d(c_i)
                grad_pair[i] += w * diff_ij / (2.0 * dist_ij)
                grad_pair[j] -= w * diff_ij / (2.0 * dist_ij)
        
        # Combine: we want soft min of all constraints
        # This is tricky. Let's just take the min of the two soft mins as an approximation
        # and use the gradient from the tighter constraint.
        if soft_boundary < soft_pair:
            return soft_boundary, grad_boundary
        else:
            return soft_pair, grad_pair
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    k = 50.0  # initial sharpness
    lr = initial_lr
    
    t0 = time.time()
    while time.time() - t0 < time_budget:
        soft_val, grad = soft_min_radius(pts, k)
        
        # Gradient ascent
        pts = pts + lr * grad
        
        # Keep points inside the unit disk (hard constraint)
        norms = np.linalg.norm(pts, axis=1)
        too_far = norms > 0.999
        if np.any(too_far):
            pts[too_far] *= 0.999 / norms[too_far]
        
        # Check if improved
        current_score = _score(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
        
        # Annealing: decrease learning rate, increase sharpness
        lr *= 0.995
        k *= 1.005
        
        # If lr is too small, break
        if lr < 1e-6:
            break
    
    return best_pts


def _validate_and_fix(pts):
    """Validate and fix any issues with the packing."""
    pts = np.array(pts, dtype=np.float64)
    
    # Check for non-finite values
    if not np.all(np.isfinite(pts)):
        pts[~np.isfinite(pts)] = 0.0
    
    # Check shape
    if pts.shape != (N, 2):
        # Try to fix
        if pts.size == N * 2:
            pts = pts.reshape(N, 2)
        else:
            # Generate a new valid packing
            pts = _hex_init(N)
    
    # Ensure points are inside the unit disk
    norms = np.linalg.norm(pts, axis=1)
    too_far = norms > 1.0
    if np.any(too_far):
        pts[too_far] *= 0.999 / norms[too_far]
    
    # Check for coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                # Separate them slightly
                pts[j] += np.array([1e-6, 1e-6])
    
    return pts


def construct():
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 145.0  # 5s safety margin before 150s limit
    
    # Generate initial hexagonal packing
    pts = _hex_init(N)
    
    # Validate
    pts = _validate_and_fix(pts)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Phase 1: Coarse optimization
    if time.time() < deadline - 30.0:
        pts = _optimize(pts, time_budget=25.0, initial_lr=0.02)
        pts = _validate_and_fix(pts)
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Phase 2: Fine optimization with higher sharpness
    if time.time() < deadline - 15.0:
        pts = _optimize(pts, time_budget=10.0, initial_lr=0.005)
        pts = _validate_and_fix(pts)
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Phase 3: Polish with very sharp soft-min
    if time.time() < deadline - 5.0:
        pts = _optimize(pts, time_budget=5.0, initial_lr=0.001)
        pts = _validate_and_fix(pts)
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
