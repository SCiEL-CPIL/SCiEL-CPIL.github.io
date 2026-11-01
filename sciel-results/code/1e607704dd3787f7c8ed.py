"""Improved packing of 50 equal spheres in a unit cube."""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 50
CUBE_HALF = 0.5


def _compute_radius(pts):
    """Compute the common radius for a given set of centers."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check boundaries
    max_coord = np.max(np.abs(pts), axis=1)
    if np.any(max_coord > CUBE_HALF + 1e-12):
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist <= 1e-12:
        return 0.0
    r_pair = min_dist / 2.0
    r_wall = CUBE_HALF - np.max(max_coord)
    return min(r_pair, r_wall)


def _soft_min(vals, beta=50.0):
    """Soft minimum approximation."""
    vals = np.asarray(vals)
    return -1.0 / beta * np.log(np.sum(np.exp(-beta * vals)))


def _objective_and_grad(pts, target_r, beta=50.0):
    """Objective: maximize min distance subject to wall constraints.
    We minimize -r where r = min(pair_dists/2, wall_margins).
    Using soft-min for smoothness."""
    pts = np.asarray(pts).reshape(N, 3)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall margins
    wall_margins = CUBE_HALF - np.abs(pts)  # shape (N, 3)
    wall_margins_flat = wall_margins.flatten()
    
    # All distances (pair and wall*2 to put on same scale as diameters)
    # Actually, let's work with radii directly
    # r_pair = dists/2, r_wall = wall_margins
    # We want to maximize min of these
    # Equivalent to minimizing -min
    
    # Collect all radius values
    pair_radii = dists / 2.0
    # Only upper triangle
    tri_idx = np.triu_indices(N, k=1)
    pair_radii_flat = pair_radii[tri_idx]
    
    # Combined: all radius candidates
    all_radii = np.concatenate([pair_radii_flat, wall_margins_flat])
    
    # Soft min
    r_soft = _soft_min(all_radii, beta)
    
    # Objective: minimize -r_soft
    obj = -r_soft
    
    # Gradient
    # d(-r_soft)/d(pts) = d r_soft / d(pts) with sign flip
    # r_soft = -1/beta * log(sum(exp(-beta * r_i)))
    # d r_soft / d r_i = -exp(-beta * r_i) / sum(exp(-beta * r_j))
    
    weights = np.exp(-beta * all_radii)
    weights = weights / np.sum(weights)
    
    # Gradient w.r.t. pts
    grad = np.zeros_like(pts)
    
    # Pair radii gradient: d(r_ij)/d(p_i) = (p_i - p_j)/dist_ij
    # But we need to be careful with the indexing
    # pair_radii_flat is indexed by tri_idx
    # For each pair (i,j), contribution to grad[i] += weights[k] * d(r_ij)/d(p_i)
    # d(r_ij)/d(p_i) = (p_i - p_j) / (2 * dist_ij) * 2 = (p_i - p_j) / dist_ij ... wait
    # r_ij = |p_i - p_j| / 2
    # d r_ij / d p_i = (p_i - p_j) / (2 * |p_i - p_j|) * ... no
    # d |p_i - p_j| / d p_i = (p_i - p_j) / |p_i - p_j|
    # d (|p_i - p_j|/2) / d p_i = (p_i - p_j) / (2 * |p_i - p_j|)
    
    # Actually let me reconsider. We're minimizing -r_soft.
    # grad of -r_soft w.r.t. pts = -grad of r_soft w.r.t. pts
    # grad of r_soft w.r.t. p_i involves sum over pairs (i,j) of weight_k * d r_ij/d p_i
    # plus wall terms
    
    # Let's compute gradient of r_soft
    # dr_soft/dr_i = -weights[i] (from soft-min formula)
    # Wait: r_soft = -1/beta * log(sum_j exp(-beta * r_j))
    # d r_soft / d r_k = -1/beta * (-beta * exp(-beta*r_k)) / sum_j exp(-beta*r_j) = exp(-beta*r_k)/sum = weights[k]
    
    # So dr_soft/d p_i = sum_k weights[k] * dr_k/d p_i
    
    # For pair (i,j) at index k in the flat array:
    # dr_k/d p_i = (p_i - p_j) / (2 * dist_ij) ... wait
    # r_k = dist_ij / 2
    # dr_k/d p_i = (1/2) * d(dist_ij)/d p_i = (1/2) * (p_i - p_j)/dist_ij
    
    # For wall term on coordinate (i, axis a):
    # r_k = CUBE_HALF - |p_i[a]|
    # dr_k/d p_i[a] = -sign(p_i[a])
    
    # So:
    # grad_r_soft[i] = sum_{j} weights[k_ij] * (p_i - p_j)/(2*dist_ij) - sum_{a} weights[wall_ia] * sign(p_i[a])
    # (the wall part: for each axis a, if p_i[a] > 0, sign=1, else -1)
    
    # Let's build the gradient
    for k, (i, j) in enumerate(zip(*tri_idx)):
        d = dists[i, j]
        if d < 1e-12:
            continue
        w = weights[k]
        grad[i] += w * (pts[i] - pts[j]) / (2 * d)
        grad[j] += w * (pts[j] - pts[i]) / (2 * d)
    
    # Wall terms
    wall_offset = len(pair_radii_flat)
    for i in range(N):
        for a in range(3):
            k = wall_offset + i * 3 + a
            w = weights[k]
            if pts[i, a] >= 0:
                grad[i, a] -= w  # d(-|x|)/dx = -sign(x) for x>=0
            else:
                grad[i, a] += w  # d(-|x|)/dx = +1 for x<0 ... wait
                # r_wall = CUBE_HALF - |p_i[a]|
                # d r_wall / d p_i[a] = -sign(p_i[a])
                # So contribution to grad is -w * (-sign) = w * sign
                # Hmm let me redo
                # dr_soft/d p_i[a] += weights[k] * d(r_wall)/d p_i[a] = weights[k] * (-sign(p_i[a]))
                # So grad_r_soft[i,a] += -weights[k] * sign(p_i[a])
    
    # I made an error above. Let me redo the wall part properly.
    grad_wall = np.zeros_like(pts)
    for i in range(N):
        for a in range(3):
            k = wall_offset + i * 3 + a
            w = weights[k]
            sgn = 1.0 if pts[i, a] >= 0 else -1.0
            grad_wall[i, a] += -w * sgn  # d r_soft / d p_i[a] from wall term
    
    grad_total = grad + grad_wall
    # Objective is -r_soft, so gradient of objective is -grad_total
    obj_grad = -grad_total
    
    return obj, obj_grad


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial placement: slightly perturbed grid
    rng = np.random.default_rng(42)
    m = 4  # 4x4x4 = 64 > 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()
    pts += rng.normal(0, 0.01, size=pts.shape)
    
    # Clip to cube
    pts = np.clip(pts, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Optimization loop
    beta_start = 10.0
    beta_end = 200.0
    n_iter_max = 200
    
    for iteration in range(n_iter_max):
        if time.time() > deadline - 10:
            break
        
        # Anneal beta
        t = iteration / max(n_iter_max - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        # Use L-BFGS-B with gradient
        def obj_func(p):
            p = p.reshape(N, 3)
            f, g = _objective_and_grad(p, best_r, beta)
            return f, g.ravel()
        
        # Convert to bounds
        bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
        
        try:
            res = minimize(obj_func, pts.ravel(), jac=True, bounds=bounds,
                          method='L-BFGS-B', options={'maxiter': 30, 'gtol': 1e-6})
            pts_new = res.x.reshape(N, 3)
        except Exception:
            pts_new = pts.copy()
        
        # Ensure validity
        pts_new = np.clip(pts_new, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        r_new = _compute_radius(pts_new)
        if r_new > best_r:
            best_r = r_new
            best_pts = pts_new.copy()
        
        pts = pts_new
    
    # Final polish with higher beta
    if time.time() < deadline - 5:
        beta = 500.0
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            try:
                res = minimize(obj_func, best_pts.ravel(), jac=True, bounds=bounds,
                              method='L-BFGS-B', options={'maxiter': 20, 'gtol': 1e-8})
                pts_new = res.x.reshape(N, 3)
                pts_new = np.clip(pts_new, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
                r_new = _compute_radius(pts_new)
                if r_new > best_r:
                    best_r = r_new
                    best_pts = pts_new.copy()
            except Exception:
                break
    
    return best_pts
