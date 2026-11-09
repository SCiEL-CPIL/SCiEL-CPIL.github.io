"""Pack 300 equal circles in unit circle using gradient-based optimization with soft-min."""
import time
import numpy as np
from scipy.optimize import minimize, fmin_l_bfgs_b

N = 300

def _soft_min_obj(x, n, beta, m):
    """
    Objective: maximize min(pairwise_dist/2, boundary_dist)
    We minimize -min(...) using soft-min approximation.
    
    x: (2*n,) array of centers
    beta: inverse temperature for soft-min
    m: number of boundary constraints to include (for efficiency)
    """
    centers = x.reshape(-1, 2)
    
    # Pairwise distances
    # Use chunked computation for memory efficiency
    min_pair_dist = np.inf
    chunk_size = 50
    for i in range(0, n, chunk_size):
        i_end = min(i + chunk_size, n)
        diffs = centers[i:i_end, None, :] - centers[None, :, :]
        # Only compute upper triangle for this chunk
        for j in range(i_end - i):
            row_idx = i + j
            for k in range(row_idx + 1, n):
                d = np.sqrt(np.sum(diffs[j, k - i, :] ** 2))
                if d < min_pair_dist:
                    min_pair_dist = d
    
    # Actually, let's do this more efficiently with a different approach
    # Use a random subset of pairs for speed, but track the min
    # Better: use vectorized computation in chunks
    
    # Let me redo this properly
    min_pair_dist = np.inf
    for i in range(n):
        diffs = centers[i+1:] - centers[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        if len(dists) > 0:
            min_d = np.min(dists)
            if min_d < min_pair_dist:
                min_pair_dist = min_d
    
    # Boundary distances
    boundary_dists = 1.0 - np.sqrt(np.sum(centers ** 2, axis=1))
    min_boundary = np.min(boundary_dists)
    
    # Soft-min of {min_pair_dist/2, min_boundary}
    # Actually, the objective is min(min_pair_dist/2, min_boundary)
    # Let's use log-sum-exp for soft-min
    vals = np.array([min_pair_dist / 2.0, min_boundary])
    # soft-min = -log(sum(exp(-beta * vals))) / beta
    max_val = np.max(vals)
    log_sum = np.log(np.sum(np.exp(-beta * (vals - max_val))))
    soft_min = -np.log(np.sum(np.exp(-beta * vals))) / beta
    
    # We want to maximize this, so minimize negative
    return -soft_min


def _soft_min_obj_fast(x, n, beta):
    """Faster soft-min objective using vectorized operations."""
    centers = x.reshape(-1, 2)
    
    # Pairwise distances - compute all at once if memory allows
    # For n=300, 300*300*8 = 720KB, fine
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    boundary_dists = 1.0 - np.sqrt(np.sum(centers ** 2, axis=1))
    min_boundary = np.min(boundary_dists)
    
    # Soft-min of the two values
    vals = np.array([min_pair_dist / 2.0, min_boundary])
    max_val = np.max(vals)
    log_sum = np.log(np.sum(np.exp(-beta * (vals - max_val))))
    soft_min = max_val - np.log(np.sum(np.exp(-beta * (vals - max_val)))) / beta
    
    return -soft_min


def _init_hexagonal(n):
    """Initialize with hexagonal lattice."""
    # Estimate spacing
    area_per_circle = np.pi / n
    spacing = np.sqrt(area_per_circle)
    
    points = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        y = j * spacing * np.sqrt(3.0) / 2.0
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            if x * x + y * y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than n, randomly sample
    if len(points) > n:
        idx = np.random.choice(len(points), n, replace=False)
        points = points[idx]
    
    # If we have fewer, add random points
    while len(points) < n:
        r = np.sqrt(np.random.random()) * 0.9
        theta = np.random.random() * 2 * np.pi
        points = np.vstack([points, [[r * np.cos(theta), r * np.sin(theta)]]])
    
    return points[:n]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    # Initialize with hexagonal lattice
    centers = _init_hexagonal(N)
    
    best_score = -np.inf
    best_centers = centers.copy()
    
    def _score(c):
        """Compute the actual score."""
        c = c.reshape(-1, 2)
        # Check validity
        if not np.all(np.isfinite(c)):
            return -np.inf
        if np.any(np.sqrt(np.sum(c ** 2, axis=1)) >= 1.0):
            return -np.inf
        
        # Pairwise distances
        diff = c[:, None, :] - c[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Boundary
        boundary = 1.0 - np.sqrt(np.sum(c ** 2, axis=1))
        min_boundary = np.min(boundary)
        
        return min(min_pair, min_boundary)
    
    # Try multiple initializations and beta values
    betas = [50, 100, 200, 500]
    
    for init_idx in range(3):
        if time.time() > deadline:
            break
        
        if init_idx == 0:
            centers = _init_hexagonal(N)
        elif init_idx == 1:
            # Perturbed hexagonal
            centers = _init_hexagonal(N)
            centers += np.random.normal(0, 0.01, centers.shape)
            # Project back inside
            norms = np.sqrt(np.sum(centers ** 2, axis=1))
            mask = norms > 0.99
            centers[mask] *= 0.99 / norms[mask, None]
        else:
            # More random
            centers = np.random.normal(0, 0.3, (N, 2))
            norms = np.sqrt(np.sum(centers ** 2, axis=1))
            mask = norms > 0.95
            centers[mask] *= 0.95 / norms[mask, None]
        
        x0 = centers.flatten()
        
        for beta in betas:
            if time.time() > deadline:
                break
            
            try:
                def obj(x):
                    return _soft_min_obj_fast(x, N, beta)
                
                result = minimize(
                    obj,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                x_opt = result.x.reshape(-1, 2)
                
                # Project inside circle
                norms = np.sqrt(np.sum(x_opt ** 2, axis=1))
                mask = norms > 1.0
                if np.any(mask):
                    x_opt[mask] *= (1.0 - 1e-10) / norms[mask, None]
                
                score = _score(x_opt)
                
                if score > best_score:
                    best_score = score
                    best_centers = x_opt.copy()
                    
            except Exception:
                pass
    
    # Final polish: try to improve with higher beta
    if time.time() < deadline - 20:
        x0 = best_centers.flatten()
        for beta in [1000, 2000, 5000]:
            if time.time() > deadline:
                break
            
            try:
                def obj(x):
                    return _soft_min_obj_fast(x, N, beta)
                
                result = minimize(
                    obj,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
                )
                
                x_opt = result.x.reshape(-1, 2)
                
                # Project inside circle
                norms = np.sqrt(np.sum(x_opt ** 2, axis=1))
                mask = norms > 1.0
                if np.any(mask):
                    x_opt[mask] *= (1.0 - 1e-10) / norms[mask, None]
                
                score = _score(x_opt)
                
                if score > best_score:
                    best_score = score
                    best_centers = x_opt.copy()
                    x0 = x_opt.flatten()
                    
            except Exception:
                pass
    
    # Final check
    final_score = _score(best_centers)
    if final_score > best_score:
        best_score = final_score
    
    return best_centers
