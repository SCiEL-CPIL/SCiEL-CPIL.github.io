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

def _soft_min_optimize(pts, deadline, eps_start=20.0, eps_end=500.0, lr_start=0.002, lr_end=0.00005, max_steps=3000):
    """Optimize using soft-min gradient ascent with annealing."""
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    pts = pts.copy()
    
    for step in range(max_steps):
        if time.time() > deadline - 3.0:
            break
        
        progress = step / max_steps
        eps = eps_start * (eps_end / eps_start) ** progress
        lr = lr_start * (lr_end / lr_start) ** progress
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        dists = np.sqrt(dists_sq)
        
        # Boundary distances
        radii = np.hypot(pts[:, 0], pts[:, 1])
        b_dists = 1.0 - radii
        
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        
        # Compute weights
        weights_pair = np.exp(-eps * dists)
        np.fill_diagonal(weights_pair, 0.0)
        
        # Unit vectors
        mask = dists > 1e-12
        unit_diff = np.zeros_like(diff)
        unit_diff[mask] = diff[mask] / dists[mask, np.newaxis]
        
        # Pairwise gradient: sum of weighted unit vectors (pushes apart)
        grad_pair = np.sum(weights_pair[:, :, np.newaxis] * unit_diff, axis=1)
        
        # Boundary gradient: push inward when close to boundary
        weights_bound = np.exp(-eps * b_dists)
        unit_radii = np.zeros_like(pts)
        mask_r = radii > 1e-12
        unit_radii[mask_r] = pts[mask_r] / radii[mask_r, np.newaxis]
        
        # The force from boundary should push inward (negative radial direction)
        grad_bound = -weights_bound[:, np.newaxis] * unit_radii
        
        # Total update direction (we want to maximize min distance, so follow positive gradient)
        grad_total = grad_pair + grad_bound
        
        # Update
        pts += lr * grad_total
        
        # Project back to unit disk
        new_radii = np.hypot(pts[:, 0], pts[:, 1])
        out_mask = new_radii > 1.0
        if np.any(out_mask):
            pts[out_mask] = pts[out_mask] / new_radii[out_mask, np.newaxis]
        
        # Check improvement
        current_r = _compute_radius(pts)
        if current_r > best_r:
            best_r = current_r
            best_pts = pts.copy()
    
    return best_pts, best_r

def _polish(pts, deadline):
    """Final high-precision polish pass with very small steps."""
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    pts = pts.copy()
    
    # Use a very large eps for near-exact min
    eps = 1000.0
    lr = 0.0001
    n_steps = 500
    
    for step in range(n_steps):
        if time.time() > deadline - 1.0:
            break
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        radii = np.hypot(pts[:, 0], pts[:, 1])
        b_dists = 1.0 - radii
        
        np.fill_diagonal(dists, np.inf)
        
        weights_pair = np.exp(-eps * dists)
        np.fill_diagonal(weights_pair, 0.0)
        
        mask = dists > 1e-12
        unit_diff = np.zeros_like(diff)
        unit_diff[mask] = diff[mask] / dists[mask, np.newaxis]
        
        grad_pair = np.sum(weights_pair[:, :, np.newaxis] * unit_diff, axis=1)
        
        weights_bound = np.exp(-eps * b_dists)
        unit_radii = np.zeros_like(pts)
        mask_r = radii > 1e-12
        unit_radii[mask_r] = pts[mask_r] / radii[mask_r, np.newaxis]
        
        grad_bound = -weights_bound[:, np.newaxis] * unit_radii
        
        grad_total = grad_pair + grad_bound
        pts += lr * grad_total
        
        new_radii = np.hypot(pts[:, 0], pts[:, 1])
        out_mask = new_radii > 1.0
        if np.any(out_mask):
            pts[out_mask] = pts[out_mask] / new_radii[out_mask, np.newaxis]
        
        current_r = _compute_radius(pts)
        if current_r > best_r:
            best_r = current_r
            best_pts = pts.copy()
    
    return best_pts, best_r

def construct():
    deadline = time.time() + 145.0
    
    # 1. Initialization - try multiple hexagonal spacings
    best_pts = None
    best_r = 0.0
    for spacing in np.linspace(0.085, 0.125, 15):
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
        # Fallback: random initialization
        best_pts = np.random.randn(N, 2) * 0.7
        best_r = _compute_radius(best_pts)

    # 2. Main optimization
    best_pts, best_r = _soft_min_optimize(best_pts, deadline)
    
    # 3. Polish pass
    if time.time() < deadline - 5.0:
        best_pts, best_r = _polish(best_pts, deadline)
    
    return best_pts
# EVOLVE-BLOCK-END
