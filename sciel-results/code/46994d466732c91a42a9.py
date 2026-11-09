import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

def _compute_radius(pts):
    """Compute the packing radius for a set of points."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Boundary constraint
    dists_center = np.linalg.norm(pts, axis=1)
    boundary_min = np.min(1.0 - dists_center)
    if boundary_min <= 0:
        return 0.0
    
    # Pairwise distances
    # Using cdist is often faster and more stable for large N
    dist = cdist(pts, pts)
    # Only consider i < j
    iu = np.triu_indices(N, k=1)
    pair_min = np.min(dist[iu])
    
    r = min(boundary_min, pair_min / 2.0)
    return float(r)

def _initial_hex_grid():
    """Generate a hexagonal grid initial guess."""
    # Start with a slightly smaller radius to ensure it fits, then expand
    r = 0.12
    pts = []
    row_h = r * np.sqrt(3)
    max_y = 1.0
    num_rows = int(max_y / row_h) + 2
    for j in range(-num_rows, num_rows + 1):
        y = j * row_h
        offset = r if j % 2 != 0 else 0
        max_x = np.sqrt(max(0, 1.0 - y*y))
        num_cols = int(max_x / (2*r)) + 2
        for i in range(-num_cols, num_cols + 1):
            x = i * 2*r + offset
            if x*x + y*y <= 1.0:
                pts.append([x, y])
    pts = np.array(pts)
    
    if len(pts) < N:
        # Fill with random points if not enough
        while len(pts) < N:
            p = np.random.randn(2) * 0.5
            pts = np.vstack([pts, p])
    else:
        # Select N closest to center to form a compact cluster
        d = np.linalg.norm(pts, axis=1)
        idx = np.argsort(d)
        pts = pts[idx[:N]]
        
    return pts

def _dynamics_relax(pts, r, steps=1000, dt=0.001, damping=0.9):
    """
    Relax the configuration using a simple dynamics simulation.
    Forces:
    1. Repulsion between overlapping circles.
    2. Inward force if circles are too close to the boundary.
    """
    pts = pts.copy()
    v = np.zeros_like(pts)
    
    # Precompute indices for i < j
    iu = np.triu_indices(N, k=1)
    
    for step in range(steps):
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        # d_ij is distance between i and j
        # We want to push apart if d_ij < 2*r
        # Vectorized calculation of differences
        diff = pts[iu[0]] - pts[iu[1]]
        dist = np.linalg.norm(diff, axis=1)
        
        # Overlap depth
        overlap = 2*r - dist
        
        # Only apply force if overlapping
        mask = overlap > 0
        if np.any(mask):
            # Force magnitude proportional to overlap
            # Normalize diff to get direction
            # Avoid division by zero
            safe_dist = np.where(dist > 1e-9, dist, 1.0)
            dir_vec = diff[mask] / safe_dist[mask][:, np.newaxis]
            force_mag = overlap[mask]
            
            # Accumulate forces
            # For each pair (i, j) in mask:
            # force on i is +dir * mag
            # force on j is -dir * mag
            idx_i = iu[0][mask]
            idx_j = iu[1][mask]
            
            # Use np.add.at for accumulation
            np.add.at(forces, idx_i, dir_vec * force_mag[:, np.newaxis])
            np.add.at(forces, idx_j, -dir_vec * force_mag[:, np.newaxis])
            
        # Boundary forces
        # If |p_i| > 1 - r, push inward
        dist_center = np.linalg.norm(pts, axis=1)
        boundary_overlap = dist_center - (1.0 - r)
        b_mask = boundary_overlap > 0
        if np.any(b_mask):
            # Direction is inward: -p_i / |p_i|
            safe_dist_c = np.where(dist_center > 1e-9, dist_center, 1.0)
            dir_c = -pts[b_mask] / safe_dist_c[b_mask][:, np.newaxis]
            force_mag_c = boundary_overlap[b_mask]
            np.add.at(forces, np.where(b_mask)[0], dir_c * force_mag_c[:, np.newaxis])
            
        # Update velocities and positions (Velocity Verlet or Euler)
        # Using simple Euler with damping for stability
        v = v * damping + forces * dt
        pts = pts + v * dt
        
        # Optional: Project back to valid region if it goes crazy?
        # Not strictly necessary if forces are strong enough, but good for safety
        # If any point goes outside 1.0, clamp it? No, let the force pull it back.
        
    return pts

def _polish(pts, r_start):
    """
    Local optimization to maximize r starting from pts and r_start.
    Uses SLSQP with constraints.
    """
    # Flatten pts
    x0 = np.concatenate([pts.flatten(), [r_start]])
    
    def objective(x):
        # We want to maximize r, so minimize -r
        # But we need to ensure constraints are satisfied.
        # SLSQP handles constraints.
        return -x[2*N]
        
    def constraints_func(x):
        pts_x = x[:2*N].reshape(N, 2)
        r_x = x[2*N]
        
        # Pairwise constraints: d_ij - 2r >= 0
        diff = pts_x[:, np.newaxis, :] - pts_x[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constr = dist[iu] - 2*r_x
        
        # Boundary constraints: 1 - |p_i| - r >= 0
        dist_center = np.linalg.norm(pts_x, axis=1)
        boundary_constr = 1.0 - dist_center - r_x
        
        return np.concatenate([pair_constr, boundary_constr])
        
    bounds = [(-1.0, 1.0)] * (2*N) + [(r_start*0.9, 0.2)]
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraints_func},
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        if res.success or res.x[2*N] > r_start:
            pts_res = res.x[:2*N].reshape(N, 2)
            r_res = res.x[2*N]
            # Verify
            valid_r = _compute_radius(pts_res)
            if valid_r > r_start:
                return pts_res, valid_r
    except Exception:
        pass
        
    return pts, r_start

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # 1. Initialization
    pts = _initial_hex_grid()
    # Initial radius estimate
    r = _compute_radius(pts)
    best_r = r
    best_pts = pts.copy()
    
    # 2. Dynamics-based expansion
    # We will try to increase r in small steps.
    # If relaxation fails, we stop or try a different strategy.
    
    current_r = r
    step_size = 0.0005
    max_steps = 2000 # Max iterations of expansion
    
    for i in range(max_steps):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Try to increase r
        trial_r = current_r + step_size
        
        # Relax configuration at trial_r
        # Use fewer steps if we are close to the limit, more if we are early
        relax_steps = 500 if i < 50 else 200
        
        pts_relaxed = _dynamics_relax(pts, trial_r, steps=relax_steps)
        
        # Check if the relaxed configuration is valid for trial_r
        # i.e., _compute_radius(pts_relaxed) >= trial_r
        actual_r = _compute_radius(pts_relaxed)
        
        if actual_r >= trial_r - 1e-5:
            # Success! Accept new radius and positions
            current_r = trial_r
            pts = pts_relaxed
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        else:
            # Failure. Maybe step size is too big.
            # Try a smaller step? Or stop?
            # For now, let's try a smaller step a few times
            step_size *= 0.5
            if step_size < 1e-6:
                break
                
    # 3. Polish
    if time.time() - start_time < time_limit - 5:
        pts_polished, r_polished = _polish(best_pts, best_r)
        if r_polished > best_r:
            best_r = r_polished
            best_pts = pts_polished
            
    # Final check and projection
    # Ensure points are strictly inside
    for i in range(N):
        d = np.linalg.norm(best_pts[i])
        if d > 1.0 - best_r:
            if d > 1e-9:
                best_pts[i] = best_pts[i] * ((1.0 - best_r) / d)
                
    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _compute_radius(pts)
    print(f"Radius: {r}")
