import time
import numpy as np
from scipy.spatial.distance import pdist, squareform

N = 50
# 150s kill limit. We use 130s to be safe.
_DEADLINE = time.time() + 130.0

def _score(pts):
    """
    Compute the packing radius for N points in a unit circle.
    Returns 0 if invalid.
    """
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Distance from origin
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0):
        return 0.0
        
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
        
    if N < 2:
        return r_cont
        
    # Pairwise distances
    # Using scipy for speed
    d = pdist(pts, 'euclidean')
    r_pair = 0.5 * np.min(d)
    
    return min(r_cont, r_pair)

def _hex_lattice_init(rotations=5):
    """
    Generate multiple hexagonal lattice initializations with different rotations.
    """
    inits = []
    # Standard hex lattice spacing
    # For N=50, radius ~0.125, diameter 0.25.
    # Hex lattice spacing s = 2*r.
    # Let's try a range of spacings and rotations.
    
    base_spacing = 0.252 # Slightly larger than 2*0.1258
    for rot in np.linspace(0, np.pi/6, rotations):
        pts = []
        # Generate points in a hex grid
        # Range needs to cover the unit circle
        k = int(1.0 / base_spacing) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                # Hex coordinates
                x = (i + 0.5 * (j % 2)) * base_spacing
                y = j * base_spacing * np.sqrt(3.0) / 2.0
                
                # Rotate
                xr = x * np.cos(rot) - y * np.sin(rot)
                yr = x * np.sin(rot) + y * np.cos(rot)
                
                r = np.hypot(xr, yr)
                # Check if inside (with margin for radius)
                if r < 1.0 - base_spacing / 2.0 - 1e-5:
                    pts.append([xr, yr])
        
        pts = np.array(pts)
        if len(pts) >= N:
            # Select N points closest to center? Or random subset?
            # Closest to center usually gives a more compact core, but boundary is important.
            # Let's select the N points that maximize the minimum distance to others? 
            # That's expensive. 
            # Simple heuristic: Randomly select N from the valid set.
            idx = np.random.choice(len(pts), N, replace=False)
            inits.append(pts[idx])
            
    # Also add a simple grid init
    inits.append(_grid_init())
    
    # Add a random init
    inits.append(_random_init())
    
    return inits

def _grid_init():
    """Simple square grid init."""
    pts = []
    spacing = 0.26
    k = int(1.0 / spacing) + 2
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = i * spacing
            y = j * spacing
            if np.hypot(x, y) < 1.0 - spacing/2 - 1e-5:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        idx = np.random.choice(len(pts), N, replace=False)
        return pts[idx]
    return _random_init()

def _random_init():
    """Random points in unit disk."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _optimize(pts0, max_time=10.0):
    """
    Optimize a single configuration using vectorized repulsion.
    """
    pts = pts0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    if best_score <= 0:
        return best_pts, best_score

    step_size = 0.05
    momentum = 0.0
    v = np.zeros_like(pts)
    stuck_count = 0
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        # d_ij = pts[i] - pts[j]
        # Vectorized pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        
        # Mask diagonal
        np.fill_diagonal(dists, 1.0)
        
        # Force magnitude: 1/d^2 or similar
        # We want to push apart if dist < 2*r_est
        # Let's use a strong repulsion for all pairs to enforce separation
        # F = (1/d^2) * (d_vec/d)
        
        # Avoid division by zero (already handled by fill_diagonal)
        inv_d2 = 1.0 / (dists * dists)
        
        # Force vector
        # f_ij = inv_d2[i,j] * diff[i,j] / dists[i,j]
        # Sum over j for each i
        
        # Note: diff[i,j] = pts[i] - pts[j]
        # Force on i is sum_j (pts[i] - pts[j]) / dist^3
        # This is equivalent to - sum_j (pts[j] - pts[i]) / dist^3
        
        # Let's compute sum of (diff / dist^3)
        # diff shape (N, N, 2)
        # dists shape (N, N)
        
        # dists_cubed = dists**3
        # weights = 1.0 / dists_cubed
        # np.fill_diagonal(weights, 0)
        # forces = np.sum(diff * weights[..., np.newaxis], axis=1)
        
        # This is O(N^2) memory and compute. N=50 is small (2500 pairs).
        # Let's do it.
        
        weights = 1.0 / (dists * dists * dists)
        np.fill_diagonal(weights, 0.0)
        
        forces = np.sum(diff * weights[..., np.newaxis], axis=1)
        
        # Boundary constraint: Push inward if too close to edge
        dists_origin = np.linalg.norm(pts, axis=1)
        # If dist > 1 - r_est, push inward
        # We don't know r_est exactly, but we can use a soft penalty
        # Force = - (dist - (1-r_est)) * dir
        # Let's use a simple penalty: if dist > 0.9, push back
        # Actually, the score function handles the boundary. 
        # But to help optimization, we can add a force towards center for points near edge.
        # Or, simply project back if they go out.
        
        # Update velocity with momentum
        v = 0.5 * v + 0.5 * forces
        pts_new = pts + step_size * v
        
        # Project back to unit circle
        new_dists = np.linalg.norm(pts_new, axis=1)
        mask = new_dists > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / new_dists[mask, np.newaxis]
            
        # Score
        new_score = _score(pts_new)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = pts_new.copy()
            stuck_count = 0
            # Increase step size slightly?
            step_size = min(0.1, step_size * 1.01)
        else:
            stuck_count += 1
            # Decrease step size
            step_size *= 0.95
            # If stuck for too long, restart with perturbation
            if stuck_count > 50:
                # Perturb current best
                pts = best_pts + np.random.normal(0, 0.05, (N, 2))
                # Project back
                d = np.linalg.norm(pts, axis=1)
                m = d > 1.0
                if np.any(m):
                    pts[m] = pts[m] / d[m, np.newaxis]
                v = np.zeros_like(pts)
                stuck_count = 0
                step_size = 0.05
                
        pts = pts_new
        
    return best_pts, best_score

def construct():
    start_time = time.time()
    
    # Generate initial configurations
    inits = _hex_lattice_init()
    
    best_pts_global = None
    best_score_global = 0.0
    
    # Time budget for each init
    # Total time 130s. 
    # Let's try to run as many as possible.
    # Each optimization run is limited to remaining time / num_inits?
    # Or just run each with a fixed time slice?
    # Since N=50 is small, optimization is fast.
    # Let's give each init 5 seconds, and run as many as fit.
    
    time_per_init = 5.0
    num_inits = len(inits)
    
    for i, init_pts in enumerate(inits):
        remaining = _DEADLINE - time.time()
        if remaining < 2.0:
            break
            
        t_alloc = min(time_per_init, remaining - 1.0)
        
        try:
            opt_pts, opt_score = _optimize(init_pts, max_time=t_alloc)
            if opt_score > best_score_global:
                best_score_global = opt_score
                best_pts_global = opt_pts.copy()
        except Exception:
            continue
            
    # If we still have time, do some random restarts on the best solution found
    remaining = _DEADLINE - time.time()
    if remaining > 5.0 and best_pts_global is not None:
        # Fine-tune the best solution
        while time.time() < _DEADLINE - 2.0:
            # Perturb slightly
            perturb_pts = best_pts_global + np.random.normal(0, 0.01, (N, 2))
            d = np.linalg.norm(perturb_pts, axis=1)
            m = d > 1.0
            if np.any(m):
                perturb_pts[m] = perturb_pts[m] / d[m, np.newaxis]
                
            opt_pts, opt_score = _optimize(perturb_pts, max_time=2.0)
            if opt_score > best_score_global:
                best_score_global = opt_score
                best_pts_global = opt_pts.copy()
            else:
                break # Stop if no improvement in fine-tuning loop to save time? 
                # Or continue? Let's continue but with small steps.
                # Actually, if it didn't improve, the perturbation was bad.
                # Let's just do a few more tries.
                
    if best_pts_global is None:
        # Fallback
        return _random_init()
        
    return best_pts_global
