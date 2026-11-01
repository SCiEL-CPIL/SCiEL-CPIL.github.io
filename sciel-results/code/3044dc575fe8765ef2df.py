"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the exact packing radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distances
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    min_bound = np.min(np.minimum(dist_x, dist_y))
    
    return min(min_pair, min_bound)

def _raw_min_obj(x, n):
    """Objective for Nelder-Mead: minimize negative radius."""
    centers = x.reshape(n, 2)
    r = _compute_radius(centers)
    return -r

def _soft_min_obj(x, n, beta):
    """Soft-min objective for L-BFGS-B polishing."""
    centers = x.reshape(n, 2)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Boundary distances
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    bound_vals = np.minimum(dist_x, dist_y)
    
    # Combine all values
    all_vals = np.concatenate([pair_vals[~np.eye(n, dtype=bool)], bound_vals])
    
    # Soft min
    m = np.max(all_vals)
    exp_terms = np.exp(-beta * (all_vals - m))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = m - log_sum / beta
    
    return -soft_min_val

def _init_grid(n):
    """7x7 grid + best spot for 50th."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_49 = np.array([(x, y) for y in ticks for x in ticks])
    
    if n == 49:
        return grid_49
    
    # Find best spot for the 50th point
    sample_res = 200
    xs = np.linspace(-0.5, 0.5, sample_res)
    ys = np.linspace(-0.5, 0.5, sample_res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    
    diff = pts[:, np.newaxis, :] - grid_49[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    min_dists = np.min(dists, axis=1)
    
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    d_b = np.minimum(d_x, d_y)
    
    clearance = np.minimum(min_dists, d_b)
    idx = np.argmax(clearance)
    new_pt = pts[idx]
    
    centers = np.vstack([grid_49, new_pt])
    return centers[:n]

def _init_hex(n):
    """Hexagonal lattice initialization."""
    # Generate a hexagonal grid covering the square
    # Radius r approx 0.07
    # Spacing s = 2r = 0.14
    s = 0.14
    dx = s
    dy = s * np.sqrt(3) / 2
    
    centers = []
    # Generate points in a larger area and clip
    x_range = np.arange(-0.6, 0.6, dx/2)
    y_range = np.arange(-0.6, 0.6, dy)
    
    for j, y in enumerate(y_range):
        offset = 0 if j % 2 == 0 else dx/2
        for x in x_range:
            centers.append((x + offset, y))
            
    centers = np.array(centers)
    # Filter to inside square
    mask = (centers[:, 0] >= -0.5) & (centers[:, 0] <= 0.5) & (centers[:, 1] >= -0.5) & (centers[:, 1] <= 0.5)
    centers = centers[mask]
    
    # If we have more than n, take the first n (or random sample)
    if len(centers) > n:
        # Sort by distance from center to pick central ones? Or just take first n.
        # Let's take a random subset to break symmetry
        indices = np.random.choice(len(centers), n, replace=False)
        centers = centers[indices]
    elif len(centers) < n:
        # Pad with random points
        pad = np.random.uniform(-0.4, 0.4, (n - len(centers), 2))
        centers = np.vstack([centers, pad])
        
    return centers

def _init_random(n):
    """Random initialization with slight relaxation."""
    np.random.seed(42)
    centers = np.random.uniform(-0.4, 0.4, (n, 2))
    return centers

def _relax_forces(centers, n, iterations=100):
    """Simple force-directed relaxation to remove overlaps."""
    centers = centers.copy()
    r_target = _compute_radius(centers)
    
    for _ in range(iterations):
        forces = np.zeros_like(centers)
        # Pairwise repulsion
        for i in range(n):
            for j in range(i+1, n):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 2 * r_target:
                    # Push apart
                    f = (2 * r_target - dist) * (d / dist)
                    forces[i] += f
                    forces[j] -= f
        
        # Wall repulsion
        for i in range(n):
            # x walls
            if centers[i][0] < -0.5 + r_target:
                forces[i][0] += ( -0.5 + r_target - centers[i][0] )
            elif centers[i][0] > 0.5 - r_target:
                forces[i][0] += ( 0.5 - r_target - centers[i][0] )
            
            # y walls
            if centers[i][1] < -0.5 + r_target:
                forces[i][1] += ( -0.5 + r_target - centers[i][1] )
            elif centers[i][1] > 0.5 - r_target:
                forces[i][1] += ( 0.5 - r_target - centers[i][1] )
                
        centers += 0.1 * forces
        # Clip to square
        centers = np.clip(centers, -0.5, 0.5)
        
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_centers = None
    best_r = 0.0
    
    inits = [
        _init_grid(N),
        _init_hex(N),
        _init_random(N)
    ]
    
    # Add jittered versions
    np.random.seed(123)
    inits.append(_init_grid(N) + np.random.normal(0, 0.02, (N, 2)))
    inits.append(_init_hex(N) + np.random.normal(0, 0.02, (N, 2)))
    
    for init_centers in inits:
        if time.time() > deadline - 10.0:
            break
            
        centers = init_centers.copy()
        # Relax first
        centers = _relax_forces(centers, N, iterations=50)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
        
        # Nelder-Mead optimization
        if time.time() > deadline - 5.0:
            continue
            
        try:
            x0 = centers.flatten()
            res = minimize(
                _raw_min_obj,
                x0,
                args=(N,),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            refined_centers = res.x.reshape(N, 2)
            refined_r = _compute_radius(refined_centers)
            
            if refined_r > r:
                r = refined_r
                centers = refined_centers
        except Exception:
            pass
        
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
            
        # Polish with L-BFGS-B on soft-min
        if time.time() > deadline - 2.0:
            continue
            
        try:
            x0 = centers.flatten()
            res = minimize(
                _soft_min_obj,
                x0,
                args=(N, 100),
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * (2 * N),
                options={'maxiter': 50, 'ftol': 1e-12}
            )
            refined_centers = res.x.reshape(N, 2)
            refined_r = _compute_radius(refined_centers)
            
            if refined_r > r:
                r = refined_r
                centers = refined_centers
        except Exception:
            pass
        
        if r > best_r:
            best_r = r
            best_centers = centers.copy()

    # Final check
    if best_centers is None:
        best_centers = _init_grid(N)
        
    # Ensure validity
    eps = 1e-9
    best_centers = np.clip(best_centers, -0.5 + eps, 0.5 - eps)
    
    return best_centers

# EVOLVE-BLOCK-END
