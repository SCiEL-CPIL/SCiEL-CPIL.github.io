"""Packing 300 equal circles in the unit disk.
Strategy:
1. Generate initial positions using hexagonal lattice (multiple spacings).
2. Refine with gradient ascent on a soft-min surrogate of the radius objective.
3. Multi-restart: try several different initializations, keep the best.
4. Final per-point coordinate descent polish.
5. Deadline guard to ensure termination within 150s.
"""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def _hex_init(n, spacing=None, jitter=0.0, seed=42):
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
            pts = points[np.argsort(radius, kind="stable")[:n]].copy()
            if jitter > 0:
                rng = np.random.default_rng(seed)
                pts += rng.standard_normal((n, 2)) * jitter
                # Re-clamp inside disk
                norms = np.hypot(pts[:, 0], pts[:, 1])
                too_far = norms > 0.999
                if np.any(too_far):
                    pts[too_far] = pts[too_far] * (0.999 / norms[too_far])
            return pts
        spacing *= 0.97


def _compute_radius(pts):
    """Compute the minimum radius for a given set of points."""
    n = len(pts)
    if n < 2:
        return float(np.min(1.0 - np.linalg.norm(pts, axis=1)))
    
    # Squared distances for speed
    sq = (pts[:, 0] ** 2 + pts[:, 1] ** 2)[:, np.newaxis] + \
         (pts[np.newaxis, :, 0] ** 2 + pts[np.newaxis, :, 1] ** 2) - \
         2.0 * pts @ pts.T
    np.fill_diagonal(sq, np.inf)
    sq = np.maximum(sq, 0.0)
    
    min_pair_dist = float(np.sqrt(np.min(sq)))
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    min_boundary = float(np.min(1.0 - norms))
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_gradient(pts, k, threshold=0.12):
    """
    Compute soft-min radius and its gradient using log-sum-exp.
    Returns (soft_value, gradient_array).
    """
    n = len(pts)
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary = 1.0 - norms
    
    # Soft min of boundary constraints
    m_b = float(np.min(boundary))
    shifted_b = boundary - m_b
    exp_b = np.exp(-k * shifted_b)
    sum_exp_b = float(np.sum(exp_b))
    soft_boundary = m_b - np.log(sum_exp_b) / k
    weights_b = exp_b / sum_exp_b
    
    # Gradient of soft_boundary w.r.t. pts
    grad_boundary = np.zeros_like(pts)
    nonzero = norms > 1e-10
    grad_boundary[nonzero] = -weights_b[nonzero, np.newaxis] * pts[nonzero] / norms[nonzero, np.newaxis]
    
    # Pairwise distances
    sq = (pts[:, 0] ** 2 + pts[:, 1] ** 2)[:, np.newaxis] + \
         (pts[np.newaxis, :, 0] ** 2 + pts[np.newaxis, :, 1] ** 2) - \
         2.0 * pts @ pts.T
    np.fill_diagonal(sq, np.inf)
    sq = np.maximum(sq, 0.0)
    dists = np.sqrt(sq)
    pair_radii = dists / 2.0
    
    # Sparse: only consider pairs within threshold
    close_mask = (dists < threshold) & (dists > 1e-10)
    close_pairs = np.argwhere(close_mask)
    close_pairs = close_pairs[close_pairs[:, 0] < close_pairs[:, 1]]
    
    if len(close_pairs) == 0:
        return soft_boundary, grad_boundary
    else:
        pair_vals = pair_radii[close_pairs[:, 0], close_pairs[:, 1]]
        m_p = float(np.min(pair_vals))
        shifted_p = pair_vals - m_p
        exp_p = np.exp(-k * shifted_p)
        sum_exp_p = float(np.sum(exp_p))
        soft_pair = m_p - np.log(sum_exp_p) / k
        weights_p = exp_p / sum_exp_p
        
        grad_pair = np.zeros_like(pts)
        for idx in range(len(close_pairs)):
            w = weights_p[idx]
            if w < 1e-12:
                continue
            i, j = close_pairs[idx]
            diff_ij = pts[i] - pts[j]
            dist_ij = dists[i, j]
            if dist_ij < 1e-10:
                continue
            unit = diff_ij / dist_ij
            grad_pair[i] += w * unit / 2.0
            grad_pair[j] -= w * unit / 2.0
    
    # Combine: soft-min of the two soft-mins
    vals = np.array([soft_boundary, soft_pair])
    m = float(np.min(vals))
    shifted = vals - m
    exps = np.exp(-k * shifted)
    sum_exps = float(np.sum(exps))
    soft_combined = m - np.log(sum_exps) / k
    
    w_b = exps[0] / sum_exps
    w_p = exps[1] / sum_exps
    grad = w_b * grad_boundary + w_p * grad_pair
    
    return soft_combined, grad


def _optimize(pts, time_budget, initial_lr=0.004, initial_k=40.0):
    """Optimize packing using gradient ascent on soft-min radius."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _compute_radius(pts)
    
    k = initial_k
    lr = initial_lr
    patience = 0
    max_patience = 400
    
    t0 = time.time()
    step = 0
    while time.time() - t0 < time_budget:
        soft_val, grad = _soft_min_gradient(pts, k, threshold=0.12)
        
        pts = pts + lr * grad
        
        # Keep points inside the unit disk
        norms = np.sqrt(np.sum(pts ** 2, axis=1))
        too_far = norms > 0.999
        if np.any(too_far):
            pts[too_far] = pts[too_far] * (0.999 / norms[too_far])
        
        # Compute actual score every 5 steps
        if step % 5 == 0 or patience > 0:
            current_score = _compute_radius(pts)
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                patience = 0
            else:
                patience += 1
        
        step += 1
        if step % 80 == 0:
            lr = max(0.0003, lr * 0.93)
            k = min(300.0, k * 1.06)
        
        if patience > max_patience:
            break
    
    return best_pts, best_score


def _coordinate_descent_polish(pts, time_budget):
    """Per-point coordinate descent: move each point optimally."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _compute_radius(pts)
    
    t0 = time.time()
    lr = 0.002
    
    step = 0
    while time.time() - t0 < time_budget:
        for i in range(N):
            if time.time() - t0 > time_budget:
                break
            
            # Find nearest neighbors and boundary constraint
            diffs = pts - pts[i]
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            dists[i] = np.inf
            
            # Get the 5 nearest neighbors
            nn_idx = np.argsort(dists)[:5]
            
            force = np.zeros(2)
            # Repulsion from nearest neighbors
            for j in nn_idx:
                d = dists[j]
                if d < 1e-10:
                    continue
                # Weight by inverse distance (stronger for closer)
                w = 1.0 / (d + 0.01)
                force += w * (pts[i] - pts[j]) / d
            
            # Boundary repulsion
            norm_i = np.linalg.norm(pts[i])
            if norm_i > 1e-10:
                boundary_force = (1.0 - norm_i) * (-pts[i] / norm_i)
                force += 2.0 * boundary_force
            
            # Move in direction of force
            force_norm = np.linalg.norm(force)
            if force_norm > 1e-12:
                pts[i] += lr * force / force_norm
            
            # Clamp inside disk
            n_i = np.linalg.norm(pts[i])
            if n_i > 0.999:
                pts[i] = pts[i] * (0.999 / n_i)
        
        current_score = _compute_radius(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
        
        step += 1
        if step % 20 == 0:
            lr = max(0.0001, lr * 0.9)
    
    return best_pts, best_score


def _validate_and_fix(pos):
    """Ensure positions are valid: finite, inside unit disk, no coincident points."""
    pos = np.array(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    if not np.all(np.isfinite(pos)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    norms = np.sqrt(np.sum(pos ** 2, axis=1))
    mask = norms >= 1.0
    if np.any(mask):
        pos[mask] = pos[mask] * (0.999 / norms[mask])
    
    # Check for coincident points
    sq = (pos[:, 0] ** 2 + pos[:, 1] ** 2)[:, np.newaxis] + \
         (pos[np.newaxis, :, 0] ** 2 + pos[np.newaxis, :, 1] ** 2) - \
         2.0 * pos @ pos.T
    np.fill_diagonal(sq, np.inf)
    sq = np.maximum(sq, 0.0)
    min_sq = float(np.min(sq))
    if min_sq < 1e-8:
        rng = np.random.default_rng(42)
        dist = np.sqrt(sq)
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i, j] < 1e-4:
                    pos[i] += rng.standard_normal(2) * 1e-3
                    pos[j] -= rng.standard_normal(2) * 1e-3
                    n_i = np.linalg.norm(pos[i])
                    if n_i > 0.999:
                        pos[i] = pos[i] * (0.999 / n_i)
                    n_j = np.linalg.norm(pos[j])
                    if n_j > 0.999:
                        pos[j] = pos[j] * (0.999 / n_j)
    
    return pos


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s budget, leaving 10s margin
    
    best_positions = None
    best_score = 0.0
    
    # Multi-restart: try 3 different initializations
    configs = [
        {"spacing": None, "jitter": 0.0, "seed": 42},
        {"spacing": 0.058, "jitter": 0.005, "seed": 123},
        {"spacing": 0.062, "jitter": 0.01, "seed": 999},
    ]
    
    for ci, cfg in enumerate(configs):
        time_remaining = deadline - time.time()
        if time_remaining < 15.0:
            break
        
        # Allocate time: 55% to main optimization, 30% to polish, 15% buffer
        opt_budget = time_remaining * 0.55
        polish_budget = time_remaining * 0.30
        
        positions = _hex_init(N, spacing=cfg["spacing"], jitter=cfg["jitter"], seed=cfg["seed"])
        
        if opt_budget > 5.0:
            positions, score = _optimize(positions, time_budget=opt_budget)
        
        time_remaining = deadline - time.time()
        if time_remaining > 10.0 and polish_budget > 3.0:
            positions, score = _coordinate_descent_polish(positions, time_budget=polish_budget)
        
        score = _compute_radius(positions)
        if score > best_score:
            best_score = score
            best_positions = positions.copy()
    
    # If no valid result, fall back to hex init
    if best_positions is None:
        best_positions = _hex_init(N)
        best_score = _compute_radius(best_positions)
    
    best_positions = _validate_and_fix(best_positions)
    
    return best_positions
# EVOLVE-BLOCK-END
