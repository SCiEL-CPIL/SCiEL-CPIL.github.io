"""Circle packing optimizer for 300 equal circles in a unit disk."""
import numpy as np
import time

N = 300


def _hex_lattice(n=300):
    """Generate a hexagonal lattice arrangement inside the unit disk."""
    points = []
    # Generate hex lattice points
    spacing = 0.06
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    # Sort by distance from center (closest first)
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists)
    points = points[idx[:n]]
    return points


def _compute_radius(pts):
    """Compute the maximum valid radius for the given centers."""
    n = len(pts)
    # Container constraint
    r_cont = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    
    # Pairwise constraint (use efficient method for large N)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    
    return min(r_cont, r_pair)


def _local_optimize(pts, max_iters=2000, lr_init=0.005, lr_min=0.0001):
    """Local optimization using gradient-based method."""
    pts = pts.copy()
    n = len(pts)
    
    # Precompute neighbor list for efficiency
    # Use a simple approach: compute all pairwise forces each iteration
    # but use vectorized operations
    
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    for it in range(max_iters):
        # Compute forces
        # Force from container: push inward if too close to boundary
        dists_center = np.hypot(pts[:, 0], pts[:, 1])
        f_cont = np.zeros_like(pts)
        # For points close to boundary, push inward
        mask = dists_center > 0.5
        if np.any(mask):
            # Gradient of (1 - |c|) is -c/|c|
            f_cont[mask] = -pts[mask] / dists_center[mask, np.newaxis]
        
        # Force from pairwise repulsion
        # Use a soft-min approximation for the minimum distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        # Force: each pair pushes apart proportional to 1/d^2 (soft)
        # Use a temperature-based soft minimum
        beta = 50.0  # sharpness of soft-min
        # For each point, compute net repulsive force
        f_pair = np.zeros_like(pts)
        
        # Efficient: for each i, sum over j != i of (pts[i]-pts[j]) / dist^2
        # But this is O(n^2), which for n=300 is 90000 ops - fine
        inv_d2 = 1.0 / (dists ** 2)
        np.fill_diagonal(inv_d2, 0.0)
        
        # Force on i from j: (pts[i] - pts[j]) / dist_ij^2
        # f_pair[i] = sum_j (pts[i] - pts[j]) / dist_ij^2
        # = n*pts[i] - sum_j pts[j] / dist_ij^2
        # But we need to weight by 1/dist^2
        # f_pair[i] = sum_j (pts[i] - pts[j]) * inv_d2[i,j]
        # = pts[i] * sum_j inv_d2[i,j] - sum_j pts[j] * inv_d2[i,j]
        
        # Vectorized:
        weights = inv_d2  # shape (n, n)
        f_pair = pts * np.sum(weights, axis=1)[:, np.newaxis] - (weights @ pts)
        
        # Combine forces
        force = f_cont * 0.5 + f_pair * 0.5
        
        # Adaptive learning rate
        lr = lr_init * (lr_min / lr_init) ** (it / max_iters)
        
        # Update
        pts_new = pts + lr * force
        
        # Project to valid region (simple: clip to unit disk)
        dists_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
        mask_out = dists_new > 0.999
        if np.any(mask_out):
            pts_new[mask_out] *= 0.999 / dists_new[mask_out, np.newaxis]
        
        # Check improvement
        r_new = _compute_radius(pts_new)
        if r_new > best_r:
            best_r = r_new
            best_pts = pts_new.copy()
            pts = pts_new
        else:
            # Accept with some probability (simulated annealing)
            if it < max_iters // 2 and np.random.random() < 0.1:
                pts = pts_new
            else:
                # Decay learning rate faster
                lr *= 0.9
    
    return best_pts, best_r


def _anneal(pts, temp_steps=10, temp_init=0.03, temp_min=2e-5):
    """Simulated annealing refinement."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    for step in range(temp_steps):
        temp = temp_init * (temp_min / temp_init) ** (step / max(temp_steps - 1, 1))
        n_iter = 500
        
        for it in range(n_iter):
            # Random perturbation
            perturbation = np.random.randn(len(pts), 2) * temp
            pts_new = pts + perturbation
            
            # Keep inside disk
            dists = np.hypot(pts_new[:, 0], pts_new[:, 1])
            mask = dists > 0.999
            if np.any(mask):
                pts_new[mask] *= 0.999 / dists[mask, np.newaxis]
            
            r_new = _compute_radius(pts_new)
            delta = r_new - best_r
            
            # Accept if better or with probability exp(delta/temp)
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                pts = pts_new
                if r_new > best_r:
                    best_r = r_new
                    best_pts = pts_new.copy()
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    # Initial hexagonal lattice
    pts = _hex_lattice(N)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Local optimization
    pts, r = _local_optimize(pts, max_iters=1500)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()
    
    # Check time
    if time.time() > deadline - 30:
        return best_pts
    
    # Annealing
    pts, r = _anneal(pts, temp_steps=8, temp_init=0.02, temp_min=1e-5)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()
    
    # Check time
    if time.time() > deadline - 20:
        return best_pts
    
    # Final polish: low-temperature local optimization
    pts, r = _local_optimize(pts, max_iters=1000, lr_init=0.001, lr_min=0.00005)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()
    
    # Ultra-low temperature polish
    if time.time() < deadline - 10:
        pts, r = _anneal(pts, temp_steps=3, temp_init=0.005, temp_min=1e-6)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Final local polish
    if time.time() < deadline - 5:
        pts, r = _local_optimize(pts, max_iters=500, lr_init=0.0005, lr_min=0.00001)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    return best_pts
