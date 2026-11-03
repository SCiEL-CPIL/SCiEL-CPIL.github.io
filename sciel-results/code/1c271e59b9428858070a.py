import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0

    # Try a few different spacings and rotations
    for attempt in range(14):
        # Spacing estimation
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        spacing = base_spacing * (0.95 + 0.015 * attempt)

        k = int(1.0 / spacing) + 5
        rows = []
        # Use a fixed rotation per attempt to break symmetry
        rot_angle = rng.uniform(0, 2 * np.pi)
        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                # Apply rotation
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                r = np.hypot(x_rot, y_rot)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x_rot, y_rot, r))

        if len(rows) < n_points:
            continue

        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]

        # Add noise for attempts > 0
        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.04, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]

        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)

        if r > best_r:
            best_r = r
            best_pts = selected.copy()

    if best_pts is None:
        # Fallback
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]

    return best_pts

def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)

def _soft_min_logsumexp(values, T):
    """Stable soft minimum: -T * logsumexp(-values/T)"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    neg_vals = -values / T
    max_neg = np.max(neg_vals)
    if max_neg > 700:
        return values.min()
    exp_shifted = np.exp(neg_vals - max_neg)
    log_sum = np.log(np.sum(exp_shifted)) + max_neg
    return -T * log_sum

def _spectral_regularization_grad(pts, weight=0.1):
    """
    Computes gradient of the variance of Delaunay edge lengths.
    Minimizing this variance promotes a more uniform lattice structure,
    acting as a global regularizer to enforce rigidity and uniformity.
    """
    try:
        # Compute Delaunay triangulation
        tri = Delaunay(pts)
        edges = tri.edges
        if len(edges) == 0:
            return np.zeros_like(pts)
        
        # Compute edge lengths
        p1 = pts[edges[:, 0]]
        p2 = pts[edges[:, 1]]
        diffs = p1 - p2
        lengths_sq = np.sum(diffs**2, axis=1)
        lengths = np.sqrt(lengths_sq)
        
        # Variance of lengths
        mean_len = np.mean(lengths)
        # We want to minimize sum((l_i - mean)^2)
        # Gradient w.r.t pts:
        # d/dx sum (l_i - mean)^2 = 2 * sum (l_i - mean) * dl_i/dx
        # dl_i/dx = (x_i - x_j) / l_i  (for edge i connecting i, j)
        
        # Efficiently accumulate gradients
        grads = np.zeros_like(pts)
        # (l_i - mean) * (1/l_i) = (l_i - mean) / l_i = 1 - mean/l_i
        # Let w_i = 2 * (l_i - mean) / l_i
        w = 2.0 * (lengths - mean_len) / np.maximum(lengths, 1e-9)
        
        # For each edge (u, v), grad_u += w_i * (p_u - p_v), grad_v -= w_i * (p_u - p_v)
        # Note: diffs = p1 - p2 = p_u - p_v
        # So we add w_i * diffs to p_u and subtract w_i * diffs from p_v
        
        scaled_diffs = diffs * w[:, np.newaxis]
        
        # Add to u indices
        np.add.at(grads, edges[:, 0], scaled_diffs)
        # Subtract from v indices
        np.add.at(grads, edges[:, 1], -scaled_diffs)
        
        return grads * weight
    except Exception:
        return np.zeros_like(pts)

def _force_relax(pts, n_steps=2000, deadline=None):
    """Fast force-directed relaxation with Hertzian overlap^4 repulsion."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)

    # Estimate initial radius
    r_est = best_r

    lr = 0.02
    cooldown = 0.998

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        forces = np.zeros_like(pos)

        # Pairwise repulsion using Hertzian overlap^4 potential
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)

        # Target separation: 2 * r_est
        target = 2.0 * r_est
        mask = dists < target
        # Safe distances
        safe_dists = np.where(mask, dists, 1.0)

        # Hertzian overlap: (target - dist)^4 when dist < target
        excess = np.where(mask, target - dists, 0.0)
        overlap4 = excess ** 4

        # Force magnitude: derivative of (target-d)^4 / d = 4*(target-d)^3
        force_mag = np.where(mask, 4.0 * excess**3 / safe_dists, 0.0)

        # Unit vectors
        unit_vecs = diff / safe_dists[:, :, np.newaxis]

        # Sum forces
        forces += np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)

        # Boundary force: push inward if too close to wall
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.03
        for i in range(n):
            if norms[i] > 1.0 - wall_margin:
                excess_b = norms[i] - (1.0 - wall_margin)
                if excess_b > 0:
                    f_mag = 50.0 * excess_b**3
                    if norms[i] > 1e-10:
                        direction = -pos[i] / norms[i]
                    else:
                        direction = np.array([1.0, 0.0])
                    forces[i] += f_mag * direction

        # Update positions
        pos += lr * forces
        lr *= cooldown

        # Project to unit circle
        norms_new = np.linalg.norm(pos, axis=1)
        mask_out = norms_new > 1.0
        if np.any(mask_out):
            pos[mask_out] /= norms_new[mask_out, np.newaxis]

        # Check improvement
        current_r = _compute_radius(pos)
        if current_r > best_r:
            best_r = current_r
            best_pos = pos.copy()

    return best_pos

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization: Use the improved hex init from inspiration
    points = _hex_init(N, seed=42)
    
    # Initial force relaxation
    deadline_force = start_time + 10.0
    points = _force_relax(points, n_steps=2000, deadline=deadline_force)
    
    best_points = points.copy()
    best_score = _compute_radius(best_points)
    
    # Precompute indices for pairwise distances
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def _score(pts):
        """Compute the packing score."""
        d = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(d**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists[i_idx, j_idx]) / 2.0
        rad = np.linalg.norm(pts, axis=1)
        bound_min = np.min(1.0 - rad)
        return min(pair_min, bound_min)
    
    def _soft_min_weights(vals, k_soft):
        """Compute soft-min weights with numerical stability."""
        current_min = np.min(vals)
        shifted = vals - current_min
        
        # For numerical stability, only consider values within a reasonable range
        threshold = 50.0 / k_soft
        mask = shifted < threshold
        
        if np.any(mask):
            close_shifted = shifted[mask]
            # Use log-sum-exp for stability
            max_close = np.max(close_shifted)
            if max_close > 0:
                exp_terms = np.exp(-k_soft * (close_shifted - max_close))
            else:
                exp_terms = np.exp(-k_soft * close_shifted)
            
            sum_exp = np.sum(exp_terms)
            if sum_exp > 0:
                weights = np.zeros_like(vals)
                weights[mask] = exp_terms / sum_exp
            else:
                weights = np.ones_like(vals) / len(vals)
        else:
            weights = np.ones_like(vals) / len(vals)
        
        return weights, current_min

    # Optimization parameters
    alpha = 0.02
    cooling = 0.99995  # Slower cooling
    k_soft_start = 3.0
    k_soft_end = 150.0
    max_iter = 8000
    iter_count = 0
    stall_count = 0
    stall_threshold = 150
    
    # Adaptive k_soft
    k_soft = k_soft_start
    
    # FISTA parameters
    t_k = 1.0
    y = points.copy()
    spectral_weight = 0.05 # Start with small weight, increase if needed
    
    # Phase 1: Main optimization with FISTA and Spectral Regularization
    while time.time() - start_time < time_limit * 0.65 and iter_count < max_iter:
        # Compute pairwise distances efficiently
        diff = y[:, np.newaxis, :] - y[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Compute boundary distances
        rad = np.linalg.norm(y, axis=1)
        bound_dists = 1.0 - rad
        
        # All constraint values
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        if current_min > best_score:
            best_score = current_min
            best_points = y.copy()
            stall_count = 0
        else:
            stall_count += 1
        
        # Compute Soft-Min Weights
        weights, _ = _soft_min_weights(all_vals, k_soft)
        
        # Compute Gradients for Packing Constraint
        grads = np.zeros_like(y)
        
        # Pair contributions
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            vecs[mask_nonzero] = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (y / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        # Add Spectral Regularization Gradient
        spec_grad = _spectral_regularization_grad(y, spectral_weight)
        grads += spec_grad
        
        # Normalize gradient to prevent large steps
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm
        
        # FISTA Update
        # x_new = y - alpha * grad(y)
        # Project to feasible set
        x_new = y - alpha * grads
        rad_new = np.linalg.norm(x_new, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            x_new[mask] /= rad_new[mask, np.newaxis]
        
        # Momentum update
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
        y = x_new + ((t_k - 1.0) / t_next) * (x_new - y)
        
        # Project y back to feasible set as well to keep it valid
        rad_y = np.linalg.norm(y, axis=1)
        mask_y = rad_y > 1.0
        if np.any(mask_y):
            y[mask_y] /= rad_y[mask_y, np.newaxis]
        
        points = x_new
        t_k = t_next
        
        # Cool down
        alpha *= cooling
        
        # Increase k_soft over time with smoother schedule
        progress = iter_count / max_iter
        k_soft = k_soft_start + (k_soft_end - k_soft_start) * (progress ** 1.5)
        
        # Increase spectral weight slightly as we converge to enforce structure
        spectral_weight = 0.05 + 0.05 * progress
        
        # Random perturbation to escape local minima
        if stall_count > stall_threshold:
            noise_scale = alpha * 2.0
            y += np.random.normal(0, noise_scale, size=y.shape)
            rad_new = np.linalg.norm(y, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                y[mask] /= rad_new[mask, np.newaxis]
            points = y.copy()
            t_k = 1.0
            stall_count = 0
            alpha = max(alpha, 0.005)
        
        iter_count += 1

    # Phase 2: High-precision polish
    points = best_points.copy()
    polish_alpha = 0.0005
    polish_k = 300.0
    polish_time_limit = time_limit * 0.97
    
    while time.time() - start_time < polish_time_limit:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()
        
        # Very sharp soft-min for polish
        weights, _ = _soft_min_weights(all_vals, polish_k)
        
        grads = np.zeros_like(points)
        
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            vecs[mask_nonzero] = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        # Add small spectral regularization in polish too? Maybe not, focus on min distance.
        # Let's stick to pure min distance gradient for polish to maximize r.
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm
        
        points += polish_alpha * grads
        
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
        
        polish_alpha *= 0.9995
        polish_k *= 1.002

    return best_points
