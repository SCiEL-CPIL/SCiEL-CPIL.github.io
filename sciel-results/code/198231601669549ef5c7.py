"""Rewritten program for packing 50 equal circles in a unit circle."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import random


def _hex_lattice_init(n_points, radius=1.0, seed=42):
    """Initialize points on a hexagonal lattice clipped to a disk."""
    rng = np.random.default_rng(seed)
    # Use a fine hex lattice
    spacing = 0.15
    k = int(radius / spacing) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= radius * radius:
                points.append([x, y])
    points = np.array(points)
    if len(points) < n_points:
        # Pad with random points inside disk
        extra = rng.random((n_points - len(points), 2)) * 2 - 1
        radii = np.hypot(extra[:, 0], extra[:, 1])
        mask = radii < 1.0
        extra = extra[mask]
        if len(extra) < n_points - len(points):
            extra = np.vstack([extra, rng.random((n_points - len(points) - len(extra), 2)) * 0.5])
        points = np.vstack([points, extra])
    # Shuffle and take n_points
    idx = rng.permutation(len(points))[:n_points]
    return points[idx]


def _compute_score(pts):
    """Compute the packing score (minimum radius)."""
    n = len(pts)
    # Pairwise distances
    dx = pts[:, 0, np.newaxis] - pts[:, 0, np.newaxis].T
    dy = pts[:, 1, np.newaxis] - pts[:, 1, np.newaxis].T
    dists = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    
    # Boundary distances
    radii = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - radii)
    
    return min(min_pairwise, min_boundary)


def _soft_min_loss_and_grad(pts, T):
    """
    Compute soft-min loss and gradient.
    Loss = -soft_min(values) where values = [pairwise_dists/2, boundary_dists]
    We want to minimize this (i.e., maximize soft_min).
    """
    n = len(pts)
    # Pairwise distances
    dx = pts[:, 0, np.newaxis] - pts[:, 0, np.newaxis].T
    dy = pts[:, 1, np.newaxis] - pts[:, 1, np.newaxis].T
    dists = np.sqrt(dx * dx + dy * dy + 1e-12)
    np.fill_diagonal(dists, 1e10)
    
    # Boundary distances
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary_dists = 1.0 - radii
    
    # Get upper triangle indices
    iu = np.triu_indices_from(dists, k=1)
    pairwise_vals = dists[iu] / 2.0
    boundary_vals = boundary_dists
    
    all_vals = np.concatenate([pairwise_vals, boundary_vals])
    
    # Soft-min: -T * log(sum(exp(-v/T)))
    neg_vals = -all_vals / T
    neg_vals = np.clip(neg_vals, -100, 100)
    exp_vals = np.exp(neg_vals)
    sum_exp = np.sum(exp_vals)
    log_sum = np.log(sum_exp + 1e-12)
    soft_min = -T * log_sum
    loss = -soft_min
    
    # Weights for gradient
    weights = exp_vals / (sum_exp + 1e-12)
    
    # Gradient of loss w.r.t. all_vals
    # d(loss)/d(v_i) = -d(soft_min)/d(v_i) = -weights[i]
    grad_all = -weights
    
    # Propagate gradient to pts
    grad_pts = np.zeros_like(pts)
    
    # Pairwise part
    n_pairs = len(iu)
    pair_grads = grad_all[:n_pairs]
    
    for idx in range(n_pairs):
        i, j = iu[0][idx], iu[1][idx]
        d = dists[i, j]
        if d < 1e-6:
            continue
        diff = pts[i] - pts[j]
        # d(dist/2)/d(pts[i]) = -0.5 * (pts[i] - pts[j]) / dist
        # d(dist/2)/d(pts[j]) = 0.5 * (pts[i] - pts[j]) / dist
        grad_pts[i] += -0.5 * diff / d * pair_grads[idx]
        grad_pts[j] += 0.5 * diff / d * pair_grads[idx]
    
    # Boundary part
    n_boundary = len(boundary_vals)
    boundary_grads = grad_all[n_pairs:n_pairs + n_boundary]
    
    for i in range(n):
        r = radii[i]
        if r < 1e-6:
            continue
        # d(boundary_dist)/d(pts[i]) = -pts[i] / |p_i|
        grad_pts[i] += -pts[i] / r * boundary_grads[i]
    
    return loss, grad_pts


def _adam_optimize(pts_init, n_steps=2000, T_start=0.05, T_end=0.003, lr=0.005, 
                   beta1=0.9, beta2=0.999, eps=1e-8):
    """Run Adam optimization with temperature annealing."""
    pts = pts_init.copy()
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    
    for step in range(n_steps):
        progress = step / max(n_steps - 1, 1)
        # Exponential annealing of temperature
        T = T_start * (T_end / T_start) ** progress
        
        loss, grad_pts = _soft_min_loss_and_grad(pts, T)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad_pts
        v = beta2 * v + (1 - beta2) * grad_pts ** 2
        
        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))
        
        pts = pts - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Project into unit disk
        radii = np.hypot(pts[:, 0], pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            pts[mask] *= 0.999 / radii[mask, np.newaxis]
    
    return pts


def _simulated_annealing(pts_init, n_iterations=5000, T_start=0.05, T_end=0.001, 
                         alpha=0.9995, max_displacement=0.05):
    """Simple simulated annealing to escape local minima."""
    pts = pts_init.copy()
    current_score = _compute_score(pts)
    best_pts = pts.copy()
    best_score = current_score
    rng = np.random.default_rng(123)
    
    for it in range(n_iterations):
        T = T_start * (T_end / T_start) ** (it / max(n_iterations - 1, 1))
        
        # Random perturbation
        n_perturb = rng.integers(1, 4)
        idx = rng.integers(0, len(pts), size=n_perturb)
        perturb = rng.normal(0, max_displacement * T / T_start, size=(n_perturb, 2))
        new_pts = pts.copy()
        new_pts[idx] += perturb
        
        # Project into disk
        radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            new_pts[mask] *= 0.999 / radii[mask, np.newaxis]
        
        new_score = _compute_score(new_pts)
        
        # Accept if better or with probability exp((new_score - current_score) / T)
        if new_score >= current_score:
            pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        else:
            prob = np.exp((new_score - current_score) / max(T, 1e-10))
            if rng.random() < prob:
                pts = new_pts
                current_score = new_score
    
    return best_pts


def construct():
    """
    Place 50 equal, non-overlapping circles inside the unit circle.
    Returns an array of shape (N, 2) with the centres.
    """
    best_result = None
    best_score = -np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 4
    seeds = [42, 123, 456, 789]
    
    for restart in range(n_restarts):
        seed = seeds[restart]
        
        # Initialize with hex lattice
        pts_init = _hex_lattice_init(N, seed=seed)
        
        # Run Adam optimization
        pts_adam = _adam_optimize(pts_init, n_steps=1500, T_start=0.04, T_end=0.003, lr=0.004)
        score_adam = _compute_score(pts_adam)
        
        # Run simulated annealing from Adam result
        pts_sa = _simulated_annealing(pts_adam, n_iterations=3000, T_start=0.03, T_end=0.001, 
                                      max_displacement=0.03)
        score_sa = _compute_score(pts_sa)
        
        # Run Adam again from SA result for fine-tuning
        pts_final = _adam_optimize(pts_sa, n_steps=1000, T_start=0.02, T_end=0.002, lr=0.003)
        score_final = _compute_score(pts_final)
        
        # Take the best of the three
        candidates = [pts_adam, pts_sa, pts_final]
        scores = [score_adam, score_sa, score_final]
        best_idx = np.argmax(scores)
        best_pts = candidates[best_idx]
        best_s = scores[best_idx]
        
        if best_s > best_score:
            best_score = best_s
            best_result = best_pts.copy()
    
    if best_result is None:
        return _hex_lattice_init(N, seed=42)
    
    # Final safety: ensure all points are inside
    radii = np.hypot(best_result[:, 0], best_result[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        best_result[mask] *= 0.9999 / radii[mask, np.newaxis]
    
    return best_result
# EVOLVE-BLOCK-END
