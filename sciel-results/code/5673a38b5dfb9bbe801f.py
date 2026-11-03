import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_radius(centers):
    """Compute the maximum radius for a given set of centers."""
    # Check boundaries
    # Centers must be in [-0.5, 0.5]
    # Distance to walls: min(0.5 - x, x + 0.5) for each coordinate
    # Actually, distance to boundary for a point c is min(0.5 - c_k, c_k + 0.5) for k=0,1,2
    # The sphere must be inside, so radius <= min(0.5 - c_k, c_k + 0.5)
    
    # Compute min distance to walls
    # c is (N, 3)
    # dist_to_wall[i] = min(0.5 - c[i,0], c[i,0] + 0.5, 0.5 - c[i,1], c[i,1] + 0.5, 0.5 - c[i,2], c[i,2] + 0.5)
    # This can be vectorized.
    # dist_x = min(0.5 - c[:,0], c[:,0] + 0.5)
    # dist_y = min(0.5 - c[:,1], c[:,1] + 0.5)
    # dist_z = min(0.5 - c[:,2], c[:,2] + 0.5)
    # min_dist_wall = min(dist_x, dist_y, dist_z) -> min over all i and coords
    
    # Vectorized distance to walls:
    # For each point, distance to nearest wall is min(0.5 - c, c + 0.5)
    # Let's compute the min over all coordinates and points.
    # d_wall_i = min(0.5 - c[i,0], c[i,0]+0.5, 0.5 - c[i,1], c[i,1]+0.5, 0.5 - c[i,2], c[i,2]+0.5)
    
    # Efficient calculation:
    # d_x = np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5)
    # d_y = np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    # d_z = np.minimum(0.5 - centers[:, 2], centers[:, 2] + 0.5)
    # min_d_wall = np.min(np.minimum(np.minimum(d_x, d_y), d_z))
    
    d_x = np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5)
    d_y = np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    d_z = np.minimum(0.5 - centers[:, 2], centers[:, 2] + 0.5)
    min_d_wall = np.min(np.minimum(np.minimum(d_x, d_y), d_z))
    
    # Compute min distance between pairs
    # This is O(N^2) which is 100^2 = 10000, very fast.
    # We can use scipy.spatial.distance.pdist or just loop/vectorize.
    # Vectorized:
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dists = np.sqrt(np.sum(diff**2, axis=2))
    # np.fill_diagonal(dists, np.inf)
    # min_d_pair = np.min(dists)
    
    # To save memory and time, we can compute upper triangle only.
    # But for N=100, full matrix is fine.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_d_pair = np.min(dists)
    
    r = 0.5 * min(min_d_pair, min_d_wall)
    return r

def _repulsion_energy_and_grad(centers_flat):
    """
    Objective function: Minimize sum of inverse squared distances.
    This pushes spheres apart.
    """
    centers = centers_flat.reshape(N, 3)
    
    # Compute distances between all pairs
    # Use a loop for clarity and to avoid large temporary arrays if needed, 
    # but vectorization is better for speed.
    
    # Pairwise distances
    # To avoid O(N^2) memory if N was large, but N=100 is small.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    # Avoid division by zero
    np.fill_diagonal(dists_sq, np.inf)
    
    # Energy from pairs: sum 1/d^2
    # We only want to sum i<j, so divide by 2 if we sum all i!=j
    # E_pair = 0.5 * sum_{i!=j} 1/dists_sq[i,j]
    # Gradient wrt c_i: -2 * sum_{j!=i} (c_i - c_j) / dists_sq[i,j]^2
    
    # Let's compute the sum for each i
    # inv_dists_sq = 1 / dists_sq
    # inv_dists_sq[diag] = 0
    inv_dists_sq = np.zeros_like(dists_sq)
    mask = dists_sq > 1e-12
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    
    E_pair = 0.5 * np.sum(inv_dists_sq)
    
    # Gradient
    # grad_E_pair[i] = -2 * sum_{j!=i} (c_i - c_j) * inv_dists_sq[i,j]^2
    # Note: inv_dists_sq[i,j]^2 is 1/dists_sq[i,j]^2
    # Let W = inv_dists_sq^2
    # grad_i = -2 * sum_j (c_i - c_j) * W[i,j]
    
    W = inv_dists_sq ** 2
    
    # sum_j (c_i - c_j) * W[i,j]
    # = c_i * sum_j W[i,j] - sum_j c_j * W[i,j]
    
    sum_W = np.sum(W, axis=1) # Shape (N,)
    # sum_j c_j * W[i,j]
    # This is W @ centers
    sum_cW = W @ centers # Shape (N, 3)
    
    grad_pair = -2.0 * (centers * sum_W[:, np.newaxis] - sum_cW)
    
    # Energy from walls
    # E_wall = sum_i (1/d_xi^2 + 1/d_yi^2 + 1/d_zi^2)
    # d_xi = min(0.5 - x_i, x_i + 0.5)
    # If x_i < 0, d_xi = x_i + 0.5, grad = 1
    # If x_i > 0, d_xi = 0.5 - x_i, grad = -1
    # Actually, let's just compute distances and gradients numerically or analytically.
    # Analytical is better.
    
    # For each coordinate k and point i:
    # d_ik = min(0.5 - c_ik, c_ik + 0.5)
    # If c_ik <= 0: d_ik = c_ik + 0.5, dd/dc = 1
    # If c_ik > 0: d_ik = 0.5 - c_ik, dd/dc = -1
    
    # We can compute this vectorized.
    # c is (N, 3)
    # sign = np.where(centers <= 0, 1, -1)
    # d = np.minimum(0.5 - centers, centers + 0.5)
    # E_wall = sum 1/d^2
    # grad_wall[i,k] = -2/d^3 * sign[i,k]
    
    # Be careful if d is very small.
    c = centers
    sign = np.where(c <= 0, 1.0, -1.0)
    d = np.minimum(0.5 - c, c + 0.5)
    
    # Avoid division by zero
    d_safe = np.where(d > 1e-12, d, 1e-12)
    
    E_wall = np.sum(1.0 / d_safe**2)
    
    grad_wall = -2.0 * sign / d_safe**3
    
    E_total = E_pair + E_wall
    grad_total = (grad_pair + grad_wall).flatten()
    
    return E_total, grad_total

def _grid_init(jitter=0.0, seed=0):
    """Initialize with a grid, optionally with jitter."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    if jitter > 0:
        rng = np.random.default_rng(seed)
        centers += rng.uniform(-jitter, jitter, size=centers.shape)
        # Clamp to valid range? Optimization should handle it, but keep inside for safety
        # Actually, let's not clamp, let the optimizer push them back in.
        # But if they go outside, the wall energy becomes huge, which is good.
        
    return centers

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    best_centers = None
    best_radius = -1.0
    
    # Try a few different initializations
    # 1. Pure grid
    # 2. Jittered grids
    
    n_restarts = 20
    
    for i in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if i == 0:
            centers = _grid_init(jitter=0.0, seed=0)
        else:
            # Use different jitter amounts and seeds
            jitter = 0.05 * (1 + i/n_restarts)
            centers = _grid_init(jitter=jitter, seed=i)
            
        # Flatten for optimizer
        x0 = centers.flatten()
        
        # Run optimization
        try:
            res = minimize(_repulsion_energy_and_grad, x0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False})
            
            if res.success or res.status == 0:
                opt_centers = res.x.reshape(N, 3)
                # Ensure finite values
                if not np.all(np.isfinite(opt_centers)):
                    continue
                    
                r = _compute_radius(opt_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = opt_centers
                    
        except Exception as e:
            continue
            
    # If no valid result found (shouldn't happen), return grid
    if best_centers is None:
        best_centers = _grid_init(jitter=0.0, seed=0)
        
    return best_centers
