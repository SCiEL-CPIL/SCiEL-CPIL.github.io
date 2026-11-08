import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 21

# EVOLVE-BLOCK-START

def _compute_inradii(centers, W, H):
    """
    Compute the maximum radius for each circle given centers in [0,W]x[0,H].
    Uses Voronoi tessellation to find the inradius of each cell.
    """
    n = len(centers)
    radii = np.zeros(n)
    
    # Create points for Voronoi (include boundary points to handle edges)
    # Add far-away points to ensure all cells are bounded
    margin = max(W, H) * 10
    extra_points = np.array([
        [-margin, -margin], [W + margin, -margin],
        [-margin, H + margin], [W + margin, H + margin],
        [-margin, H/2], [W + margin, H/2],
        [W/2, -margin], [W/2, H + margin],
    ])
    
    all_points = np.vstack([centers, extra_points])
    
    try:
        vor = Voronoi(all_points)
    except Exception:
        # Fallback: use distance to nearest neighbor and boundary
        for i in range(n):
            cx, cy = centers[i]
            # Distance to boundary
            dist_bound = min(cx, W - cx, cy, H - cy)
            # Distance to nearest other center
            min_dist = dist_bound
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    d = math.sqrt(dx*dx + dy*dy) / 2
                    min_dist = min(min_dist, d)
            radii[i] = max(1e-6, min_dist)
        return radii
    
    for i in range(n):
        cx, cy = centers[i]
        # Distance to boundary
        dist_bound = min(cx, W - cx, cy, H - cy)
        
        # Find the Voronoi cell for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0:
            radii[i] = max(1e-6, dist_bound)
            continue
        
        # Find the inradius: minimum distance from (cx,cy) to any edge of the cell
        min_dist = dist_bound
        for v1_idx, v2_idx in zip(region[:-1], region[1:]):
            if v1_idx == -1 or v2_idx == -1:
                continue
            v1 = vor.vertices[v1_idx]
            v2 = vor.vertices[v2_idx]
            
            # Distance from point to line segment v1-v2
            dx = v2[0] - v1[0]
            dy = v2[1] - v1[1]
            length_sq = dx*dx + dy*dy
            
            if length_sq < 1e-12:
                continue
            
            t = ((cx - v1[0])*dx + (cy - v1[1])*dy) / length_sq
            t = max(0, min(1, t))
            proj_x = v1[0] + t*dx
            proj_y = v1[1] + t*dy
            
            dist = math.sqrt((cx - proj_x)**2 + (cy - proj_y)**2)
            min_dist = min(min_dist, dist)
        
        radii[i] = max(1e-6, min_dist)
    
    return radii

def _objective_smooth(centers, W, H, temperature=0.01):
    """
    Smooth objective: maximize sum of radii using soft-min for inradii.
    Returns negative sum for minimization.
    """
    n = len(centers)
    radii = _compute_inradii(centers, W, H)
    
    # Add small penalty for boundary violations (shouldn't happen with clipping)
    penalty = 0.0
    for i in range(n):
        cx, cy = centers[i]
        r = radii[i]
        if cx < r or cx > W - r or cy < r or cy > H - r:
            penalty += 100.0
    
    return -np.sum(radii) + penalty

def _objective_exact(centers, W, H):
    """
    Exact objective: sum of radii if valid, else 0.
    """
    radii = _compute_inradii(centers, W, H)
    
    # Verify no overlaps
    for i in range(N):
        for j in range(i+1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist*min_dist - 1e-9:
                return 0.0
    
    # Verify boundaries
    for i in range(N):
        cx, cy = centers[i]
        r = radii[i]
        if cx < r - 1e-9 or cx > W - r + 1e-9 or cy < r - 1e-9 or cy > H - r + 1e-9:
            return 0.0
    
    return np.sum(radii)

def _grid_init_centers(W, H, n_cols, n_rows):
    """
    Initialize center positions in a grid pattern.
    """
    centers = np.zeros((N, 2))
    
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= N:
                break
            cx = (col + 0.5) * (W / n_cols)
            cy = (row + 0.5) * (H / n_rows)
            centers[idx] = [cx, cy]
            idx += 1
        if idx >= N:
            break
            
    return centers

def _run_center_optimization(centers0, W, H, deadline, max_iter=500):
    """
    Optimize center positions to maximize sum of inradii.
    """
    start_time = time.time()
    
    def objective(v):
        if time.time() > deadline - 2:
            return 1e10
        c = v.reshape(N, 2)
        # Clip to valid range
        c[:, 0] = np.clip(c[:, 0], 0.01, W - 0.01)
        c[:, 1] = np.clip(c[:, 1], 0.01, H - 0.01)
        return _objective_smooth(c, W, H, temperature=0.005)
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.01, W - 0.01))
        bounds.append((0.01, H - 0.01))
    
    options = {
        'maxiter': max_iter,
        'ftol': 1e-10,
        'gtol': 1e-10,
        'disp': False
    }
    
    try:
        res = minimize(
            objective,
            centers0.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        
        centers_opt = res.x.reshape(N, 2)
        centers_opt[:, 0] = np.clip(centers_opt[:, 0], 0.01, W - 0.01)
        centers_opt[:, 1] = np.clip(centers_opt[:, 1], 0.01, H - 0.01)
        
        val = _objective_exact(centers_opt, W, H)
        
        if val > 0:
            return centers_opt, val
        else:
            return centers0, _objective_exact(centers0, W, H)
            
    except Exception:
        return centers0, _objective_exact(centers0, W, H)

def _polish_centers(centers, W, H, deadline):
    """
    High-precision polish pass with tighter tolerances.
    """
    start_time = time.time()
    
    def objective(v):
        if time.time() > deadline - 1:
            return 1e10
        c = v.reshape(N, 2)
        c[:, 0] = np.clip(c[:, 0], 0.005, W - 0.005)
        c[:, 1] = np.clip(c[:, 1], 0.005, H - 0.005)
        return _objective_smooth(c, W, H, temperature=0.001)
    
    bounds = []
    for i in range(N):
        bounds.append((max(0.005, centers[i, 0] - 0.02), min(W - 0.005, centers[i, 0] + 0.02)))
        bounds.append((max(0.005, centers[i, 1] - 0.02), min(H - 0.005, centers[i, 1] + 0.02)))
    
    options = {
        'maxiter': 2000,
        'ftol': 1e-14,
        'gtol': 1e-14,
        'disp': False
    }
    
    try:
        res = minimize(
            objective,
            centers.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        
        centers_opt = res.x.reshape(N, 2)
        centers_opt[:, 0] = np.clip(centers_opt[:, 0], 0.005, W - 0.005)
        centers_opt[:, 1] = np.clip(centers_opt[:, 1], 0.005, H - 0.005)
        
        val = _objective_exact(centers_opt, W, H)
        
        if val > 0:
            return centers_opt, val
        else:
            return centers, _objective_exact(centers, W, H)
            
    except Exception:
        return centers, _objective_exact(centers, W, H)

def _repulsion_refine(centers, W, H, deadline, n_steps=500):
    """
    Refine using a repulsion-based force model.
    """
    start_time = time.time()
    centers = centers.copy()
    
    # Learn rate
    lr = 0.01
    base_lr = 0.01
    
    for step in range(n_steps):
        if time.time() > deadline - 3:
            break
        
        # Decay learning rate
        lr = base_lr * (1 - step / n_steps) + 0.0001
        
        forces = np.zeros_like(centers)
        
        for i in range(N):
            cx, cy = centers[i]
            r_i = _compute_inradii(centers, W, H)[i]
            
            # Force from boundary
            if cx < r_i * 1.1:
                forces[i, 0] += (r_i * 1.1 - cx) * 10
            elif cx > W - r_i * 1.1:
                forces[i, 0] -= (cx - (W - r_i * 1.1)) * 10
            if cy < r_i * 1.1:
                forces[i, 1] += (r_i * 1.1 - cy) * 10
            elif cy > H - r_i * 1.1:
                forces[i, 1] -= (cy - (H - r_i * 1.1)) * 10
            
            # Force from other circles
            for j in range(N):
                if i == j:
                    continue
                dx = cx - centers[j, 0]
                dy = cy - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1e-6:
                    dist = 1e-6
                    dx = 1e-6
                    dy = 0
                
                r_j = _compute_inradii(centers, W, H)[j]
                min_dist = r_i + r_j
                
                if dist < min_dist * 1.2:
                    # Repulsive force
                    force_mag = (min_dist * 1.2 - dist) * 50
                    forces[i, 0] += force_mag * dx / dist
                    forces[i, 1] += force_mag * dy / dist
        
        # Update positions
        centers += lr * forces
        centers[:, 0] = np.clip(centers[:, 0], 0.01, W - 0.01)
        centers[:, 1] = np.clip(centers[:, 1], 0.01, H - 0.01)
    
    val = _objective_exact(centers, W, H)
    return centers, val

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    best_val = 0.0
    best_centers = np.zeros((N, 2))
    best_W = 1.0
    best_H = 1.0
    
    # Try different aspect ratios
    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    # Grid configurations (cols, rows)
    grid_configs = [(3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (3, 8), (8, 3), (4, 5), (5, 4), (2, 11), (11, 2)]
    
    # First pass: find a good aspect ratio and grid
    for W0 in aspect_ratios:
        if time.time() > deadline - 25:
            break
        H0 = 2.0 - W0
        
        if H0 <= 0.1:
            continue
            
        for n_cols, n_rows in grid_configs:
            if time.time() > deadline - 25:
                break
                
            # Check if grid fits
            if n_cols * n_rows < N:
                continue
                
            centers0 = _grid_init_centers(W0, H0, n_cols, n_rows)
            
            # Add small noise for diversity
            noise_scale = 0.02
            centers0 += np.random.uniform(-noise_scale, noise_scale, (N, 2))
            centers0[:, 0] = np.clip(centers0[:, 0], 0.01, W0 - 0.01)
            centers0[:, 1] = np.clip(centers0[:, 1], 0.01, H0 - 0.01)
            
            centers_opt, val = _run_center_optimization(centers0, W0, H0, deadline, max_iter=300)
            
            if val > best_val:
                best_val = val
                best_centers = centers_opt.copy()
                best_W = W0
                best_H = H0
    
    # Second pass: refine with repulsion
    if time.time() < deadline - 15 and best_val > 0:
        centers_rep, val_rep = _repulsion_refine(best_centers, best_W, best_H, deadline, n_steps=200)
        if val_rep > best_val:
            best_val = val_rep
            best_centers = centers_rep.copy()
    
    # Third pass: more optimization restarts on best aspect ratio
    if time.time() < deadline - 10 and best_val > 0:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
                
            # Choose grid based on aspect ratio
            if best_W > best_H:
                n_cols, n_rows = (6, 4) if best_W > 1.2 else (5, 5)
            else:
                n_cols, n_rows = (4, 6) if best_H > 1.2 else (5, 5)
                
            centers0 = _grid_init_centers(best_W, best_H, n_cols, n_rows)
            noise_scale = 0.03
            centers0 += np.random.uniform(-noise_scale, noise_scale, (N, 2))
            centers0[:, 0] = np.clip(centers0[:, 0], 0.01, best_W - 0.01)
            centers0[:, 1] = np.clip(centers0[:, 1], 0.01, best_H - 0.01)
            
            centers_opt, val = _run_center_optimization(centers0, best_W, best_H, deadline, max_iter=400)
            
            if val > best_val:
                best_val = val
                best_centers = centers_opt.copy()
    
    # Final polish pass
    if time.time() < deadline - 5 and best_val > 0:
        centers_pol, val_pol = _polish_centers(best_centers, best_W, best_H, deadline)
        if val_pol > best_val:
            best_val = val_pol
            best_centers = centers_pol
    
    # Fallback
    if best_val == 0.0:
        W = 1.0
        H = 1.0
        side = 5
        for i in range(N):
            row = i // side
            col = i % side
            best_centers[i] = [(col + 0.5) / side, (row + 0.5) / side]
        best_W = W
        best_H = H
    
    # Compute final radii
    radii = _compute_inradii(best_centers, best_W, best_H)
    
    # Verify and adjust if needed
    for i in range(N):
        cx, cy = best_centers[i]
        r = radii[i]
        # Ensure circle fits in boundary
        r = min(r, cx, best_W - cx, cy, best_H - cy)
        radii[i] = max(1e-6, r)
    
    # Final validation
    final_val = _objective_exact(best_centers, best_W, best_H)
    if final_val == 0:
        # Try to fix by slightly reducing radii
        for i in range(N):
            radii[i] *= 0.99
        final_val = _objective_exact(best_centers, best_W, best_H)
    
    # Return in required format: (x, y, r)
    result = np.column_stack([best_centers[:, 0], best_centers[:, 1], radii])
    
    return result

# EVOLVE-BLOCK-END
