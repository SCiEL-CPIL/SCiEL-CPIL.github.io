import numpy as np
import scipy.optimize as opt
import time

N = 300

def _hexagonal_lattice_init(n_points):
    """Generate a hexagonal lattice and select the best n_points."""
    # Estimate spacing
    # Area of unit circle is pi. Area per point is pi/n.
    # Hexagonal packing density is pi/sqrt(12).
    # spacing s such that s^2 * sqrt(3)/2 * n_points ~ pi
    # s ~ sqrt(2*pi / (n_points * sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0)))
    
    # Generate points in a large enough grid
    k = int(2.0 / s) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append((x, y))
    
    points = np.array(points)
    
    # If we have more than n_points, select the ones closest to center?
    # Or select the ones with the largest local clearance?
    # For now, closest to center is a safe bet for a valid start, 
    # but it might be suboptimal for packing (leaves boundary empty).
    # Actually, we want to fill the disk.
    # Let's select points such that they cover the disk.
    # A simple heuristic: sort by distance from center, but that fills center first.
    # Better: just take the first n_points from a spiral or random shuffle?
    # Let's shuffle and take n_points.
    if len(points) >= n_points:
        idx = np.random.choice(len(points), n_points, replace=False)
        points = points[idx]
    else:
        # If not enough, we need to add points or adjust spacing
        # This shouldn't happen with the estimate above, but just in case
        while len(points) < n_points:
            s *= 0.95
            points = []
            k = int(2.0 / s) + 2
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3.0) / 2.0
                    if x*x + y*y <= 1.0:
                        points.append((x, y))
            points = np.array(points)
            if len(points) >= n_points:
                idx = np.random.choice(len(points), n_points, replace=False)
                points = points[idx]
                break
    
    return points

def _compute_score(points):
    """Compute the packing score."""
    if len(points) != N:
        return 0.0
    # Check finite
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Boundary distance
    dists_center = np.hypot(points[:, 0], points[:, 1])
    min_boundary_dist = np.min(1.0 - dists_center)
    if min_boundary_dist < 0:
        return 0.0 # Invalid, points outside
    
    # Pairwise distances
    # Use scipy.spatial.distance.pdist for efficiency
    from scipy.spatial.distance import pdist
    pairwise_dists = pdist(points)
    min_pair_dist = np.min(pairwise_dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _objective(x, p=50, lambda_penalty=100.0):
    """
    Objective function to minimize.
    We maximize the power-mean of the distances.
    """
    points = x.reshape(-1, 2)
    
    # Penalty for being outside the circle
    dists_center = np.hypot(points[:, 0], points[:, 1])
    outside = np.maximum(0.0, dists_center - 1.0)
    penalty = lambda_penalty * np.sum(outside**2)
    
    # Boundary distances (clipped to 0 if outside, but penalty handles it)
    boundary_dists = 1.0 - dists_center
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    pairwise_dists = pdist(points)
    
    # Combine all distances
    # We want to maximize the minimum.
    # Power mean of order p approximates min as p -> inf.
    # We use distances squared to avoid sqrt? No, score is linear in r.
    # Let's use the actual distances.
    
    all_dists = np.concatenate([pairwise_dists, boundary_dists])
    
    # Clip small negative values due to numerical error if any
    all_dists = np.maximum(all_dists, 0.0)
    
    # Power mean
    # M_p = ( (1/M) sum d_k^p )^(1/p)
    # If any d_k is 0, M_p is 0.
    mean_val = np.mean(all_dists**p)
    power_mean = mean_val**(1.0/p)
    
    # We want to maximize power_mean, so minimize -power_mean
    return -power_mean + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # 1. Initialization
    points = _hexagonal_lattice_init(N)
    x0 = points.flatten()
    
    # Initial score
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # 2. Optimization
    # Use L-BFGS-B with bounds
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # We might run multiple iterations with increasing p
    p_values = [10, 20, 50, 100]
    
    current_x = x0.copy()
    
    for p in p_values:
        if time.time() > deadline:
            break
            
        try:
            res = opt.minimize(
                _objective, 
                current_x, 
                method='L-BFGS-B', 
                bounds=bounds, 
                args=(p,),
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Check time
            if time.time() > deadline:
                break
                
            new_points = res.x.reshape(-1, 2)
            new_score = _compute_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                current_x = res.x
        except Exception:
            pass

    # Final check and return
    return best_points

# Run construct to verify it works locally if needed
# if __name__ == "__main__":
#     pts = construct()
#     print(_compute_score(pts))
