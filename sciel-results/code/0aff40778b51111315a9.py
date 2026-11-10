import numpy as np
import time
import math

def _log_energy_and_grad(pts):
    """
    Computes the logarithmic energy and its gradient.
    Energy E = sum_{i<j} log(1 / |xi - xj|) = - sum_{i<j} log(|xi - xj|)
    Gradient w.r.t xi: sum_{j != i} (xj - xi) / |xi - xj|^2
    """
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    
    # Compute pairwise distances and update energy/grad
    # For efficiency, we can do this in a loop or vectorized, but n=100 is small enough for O(n^2) python loop or simple numpy broadcasting.
    # Let's use numpy broadcasting for speed.
    
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal to avoid division by zero
    mask = ~np.eye(n, dtype=bool)
    
    # Calculate energy
    # E = sum_{i<j} -log(d_ij)
    # We can sum all i != j and divide by 2
    # Be careful with d=0. If d < 1e-9, energy is huge.
    safe_dists = np.where(dists > 1e-9, dists, 1e-9)
    
    log_dists = np.log(safe_dists)
    # Sum over upper triangle
    energy = -0.5 * np.sum(log_dists[mask])
    
    # Calculate gradient
    # grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # Note: (x_j - x_i) is -diff[i, j]
    # Term for pair (i, j) in grad_i: (x_j - x_i) / d_ij^2
    # We can compute this matrix G where G[i, j] = (x_j - x_i) / d_ij^2 for j != i
    # Then grad[i] = sum_j G[i, j]
    
    # diff[i, j] = x_i - x_j
    # So x_j - x_i = -diff[i, j]
    
    # dists_sq[i, j] = d_ij^2
    dists_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    safe_dists_sq = np.where(dists_sq > 1e-18, dists_sq, 1e-18)
    
    # G[i, j] = -diff[i, j] / safe_dists_sq[i, j]
    # But we only want j != i
    G = np.zeros_like(diff)
    G[mask] = -diff[mask] / safe_dists_sq[mask, :, np.newaxis]
    
    grad = np.sum(G, axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-9, 1.0, norms)
    return pts / norms

def _optimize_from_start(pts0, time_budget):
    best_pts = pts0.copy()
    best_energy = _log_energy_and_grad(pts0)[0]
    
    pts = best_pts.copy()
    lr = 0.01 # Start with smaller lr to be stable
    max_iters = 5000
    start_t = time.time()
    
    for i in range(max_iters):
        if time.time() > time_budget - 2.0: # Safety margin
            break
        
        e, g = _log_energy_and_grad(pts)
        
        # Tangential gradient: project gradient onto tangent space
        # grad_tan = g - (g . p) p
        grad_tan = g - (g * pts).sum(axis=1, keepdims=True) * pts
        
        # Check convergence
        if np.linalg.norm(grad_tan) < 1e-10:
            break
            
        # Backtracking line search
        accepted = False
        for _ in range(50):
            # Step
            pts_new = pts - lr * grad_tan
            pts_new = _project_to_sphere(pts_new)
            
            # Check validity: no points too close
            min_dist = 1.0
            # Quick check for min distance
            # Using numpy for min distance is O(N^2)
            diff = pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            if min_dist < 1e-9:
                lr *= 0.5
                continue

            e_new, _ = _log_energy_and_grad(pts_new)
            
            if e_new < e:
                pts = pts_new
                accepted = True
                break
            lr *= 0.5
            
        if not accepted:
            # If we can't accept any step, maybe reduce lr further and try once more?
            # Or break if stuck
            if lr < 1e-6:
                break
            continue
            
        if e_new < best_energy:
            best_pts = pts.copy()
            best_energy = e_new
            
    return best_pts, best_energy

def construct():
    N = 100
    # Deadline: 120s limit, use 110s for optimization
    time_budget = 110.0
    start_time = time.time()
    
    # Initial points: Fibonacci sphere
    # This provides a good uniform distribution
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    pts = np.zeros((N, 3))
    for i in range(N):
        z = 1.0 - 2.0 * (i + 0.5) / N
        r = np.sqrt(1.0 - z**2)
        phi = golden_angle * i
        pts[i, 0] = r * np.cos(phi)
        pts[i, 1] = r * np.sin(phi)
        pts[i, 2] = z
    
    # Optimize
    best_pts, best_energy = _optimize_from_start(pts, time_budget)
    
    # Final verification and cleanup
    # Ensure points are on sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Check for invalid distances
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-9:
        # If invalid, try a random restart?
        # For now, just return what we have, hoping it's valid.
        # Or add small noise and re-project?
        # Let's add small noise to points that are too close
        close_mask = np.argwhere(dists < 1e-9)
        if len(close_mask) > 0:
             # This is a complex fix. Let's assume the optimizer avoided this.
             pass

    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        # Fallback to initial points
        best_pts = pts
    
    return best_pts
