"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _init_hex(spacing):
    """Generate hexagonal lattice points within unit disk."""
    k = int(1.0 / spacing) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing / 2.0:
                points.append((x, y))
    return np.array(points) if points else np.empty((0, 2))

def _compute_radius(pts):
    """Compute the exact packing radius."""
    if len(pts) < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair_dist / 2.0, min_boundary)

def construct():
    deadline = time.time() + 115.0
    
    # 1. Initialization
    # Try a few spacings to get a good start
    best_pts = None
    best_r = 0.0
    for spacing in np.linspace(0.09, 0.13, 20):
        pts = _init_hex(spacing)
        if len(pts) < N:
            continue
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        candidate = pts[idx]
        r = _compute_radius(candidate)
        if r > best_r:
            best_r = r
            best_pts = candidate.copy()
            
    if best_pts is None:
        # Fallback
        spacing = 0.1
        pts = _init_hex(spacing)
        if len(pts) < N:
            # Force fit
            pts = np.random.randn(N, 2) * 0.8
        else:
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind='stable')[:N]
            best_pts = pts[idx]
        best_r = _compute_radius(best_pts)

    pts = best_pts.copy()
    current_r = best_r

    # 2. Gradient Ascent on Soft-Min
    # Minimize L = sum(exp(-eps * dist))
    # Gradient descent on L is equivalent to gradient ascent on SoftMin
    
    n_steps = 2000
    eps_start = 20.0
    eps_end = 200.0
    lr_start = 0.001
    lr_end = 0.00001
    
    for step in range(n_steps):
        if time.time() > deadline - 2.0:
            break
            
        # Anneal epsilon and learning rate
        progress = step / n_steps
        eps = eps_start * (eps_end / eps_start) ** progress
        lr = lr_start * (lr_end / lr_start) ** progress
        
        # Compute distances and weights
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Boundary distances
        radii = np.hypot(pts[:, 0], pts[:, 1])
        b_dists = 1.0 - radii
        
        # To avoid numerical issues with exp(-eps * dist) for large eps,
        # we can subtract the min dist before exp, but since we want to minimize sum(exp(-eps*d)),
        # large d gives small exp. Small d gives large exp.
        # If eps is large, only the smallest distances matter.
        
        # Weights for pairwise
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        
        # Calculate exp terms
        # To prevent underflow/overflow, we can normalize, but for gradient direction, 
        # scaling doesn't matter much as long as we are consistent.
        # However, if eps*d is very large, exp becomes 0. This is fine.
        # If eps*d is very small, exp is ~1.
        
        # Gradient calculation
        grad = np.zeros_like(pts)
        
        # Pairwise term: -eps * sum_{j!=i} exp(-eps * d_ij) * (c_i - c_j) / d_ij
        # We compute the force vector for each i
        # force_ij = exp(-eps * d_ij) * (c_i - c_j) / d_ij
        # Note: (c_i - c_j) / d_ij is the unit vector from j to i.
        # So we sum unit vectors weighted by exp(-eps * d_ij).
        
        # Vectorized computation of unit vectors
        # diff is c_i - c_j
        # unit_diff = diff / dists[..., np.newaxis]
        # Handle zero division
        mask = dists > 1e-12
        unit_diff = np.zeros_like(diff)
        unit_diff[mask] = diff[mask] / dists[mask, np.newaxis]
        
        weights_pair = np.exp(-eps * dists)
        np.fill_diagonal(weights_pair, 0.0)
        
        # Sum over j for each i
        # grad_pair[i] = sum_j weights_pair[i,j] * unit_diff[i,j]
        grad_pair = np.sum(weights_pair[:, :, np.newaxis] * unit_diff, axis=1)
        
        # Boundary term: eps * exp(-eps * b_i) * c_i / |c_i|
        # Force pushes outwards if b_i is small.
        # Wait, L = sum exp(-eps * b_i). 
        # dL/db_i = -eps * exp(-eps * b_i).
        # db_i/dc_i = -c_i / |c_i|.
        # dL/dc_i = dL/db_i * db_i/dc_i = (-eps * exp(-eps * b_i)) * (-c_i / |c_i|) = eps * exp(-eps * b_i) * c_i / |c_i|.
        # We want to minimize L, so we move in direction -dL/dc_i.
        # -dL/dc_i = -eps * exp(-eps * b_i) * c_i / |c_i|.
        # This pushes points INWARDS?
        # Let's check. If b_i is small (close to boundary), exp is large.
        # c_i / |c_i| is outward normal.
        # So -dL/dc_i is inward normal.
        # This makes sense: if you are close to the boundary, the "cost" is high, so you should move away from the boundary (inward).
        
        weights_bound = np.exp(-eps * b_dists)
        unit_radii = np.zeros_like(pts)
        mask_r = radii > 1e-12
        unit_radii[mask_r] = pts[mask_r] / radii[mask_r, np.newaxis]
        
        grad_bound = -eps * weights_bound[:, np.newaxis] * unit_radii
        
        # Total gradient of L
        grad_total = -eps * grad_pair + grad_bound
        
        # Update
        pts += lr * grad_total
        
        # Optional: Project back to unit disk if they go out (shouldn't happen if lr is small and force is inward)
        new_radii = np.hypot(pts[:, 0], pts[:, 1])
        out_mask = new_radii > 1.0
        if np.any(out_mask):
            pts[out_mask] = pts[out_mask] / new_radii[out_mask, np.newaxis]
            
        # Check improvement
        current_r = _compute_radius(pts)
        if current_r > best_r:
            best_r = current_r
            best_pts = pts.copy()
            
    return best_pts
# EVOLVE-BLOCK-END
