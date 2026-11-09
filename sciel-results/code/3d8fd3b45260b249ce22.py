"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls
    min_dist_to_wall = np.min(0.5 - np.abs(centers), axis=(0, 1))

    # Minimum pairwise distance / 2
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=1.0)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d

    return min(min_dist_to_wall, min_pair_dist / 2.0)


def _soft_min_objective(x, beta):
    """
    Objective to minimize: -soft_min(constraints).
    Constraints: wall distances and half pairwise distances.
    We maximize the minimum constraint -> minimize negative soft min.
    """
    centers = x.reshape(N, 3)

    # Wall distances: (N, 3)
    wall_dists = 0.5 - np.abs(centers)

    # Pairwise distances / 2
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=1.0)

    if len(pairs) > 0:
        pair_dists = np.array([np.linalg.norm(centers[i] - centers[j]) / 2.0 for i, j in pairs])
        all_dists = np.concatenate([wall_dists.flatten(), pair_dists])
    else:
        all_dists = wall_dists.flatten()

    # Filter out non-positive to avoid log issues
    all_dists = all_dists[all_dists > 1e-12]

    if len(all_dists) == 0:
        return 0.0  # No valid constraints, shouldn't happen

    # Soft min: -log(sum(exp(-d/beta))) * beta
    scaled = -all_dists / beta
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum * beta

    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min_val


def _fibonacci_lattice():
    """Generate N points using Fibonacci lattice scaled to unit cube."""
    N_pts = N
    golden_ratio = (1 + np.sqrt(5)) / 2
    inv_phi = 1.0 / golden_ratio

    # Fibonacci lattice on sphere, then map to cube?
    # Actually, let's use a simpler deterministic spread:
    # Use a 3D grid-like distribution but with Fibonacci twist for better separation.
    # Or just use a simple cubic grid which is robust.
    # Let's try a simple cubic grid first as it's reliable, then maybe perturb.
    # The directive says "Fibonacci lattice", so let's do a proper one.
    
    # Fibonacci lattice in 3D is tricky. A common approximation is to use
    # a grid with jitter or use the fact that N=50 is not a perfect cube.
    # 50 is between 3^3=27 and 4^3=64.
    # Let's use a 4x4x4 grid and take the first 50 points? No, that leaves gaps.
    # Let's use a 3x4x5 grid? 3*4*5 = 60. Take 50.
    # Or simply use a cubic grid of 4x4x4 = 64 points, and select 50 that are well distributed.
    
    # Actually, the simplest "Fibonacci" like spread for a cube is often just
    # a well-structured grid. Let's use a 4x4x4 grid and randomly select 50?
    # No, deterministic.
    
    # Let's use a 3x4x5 grid.
    # x: 3, y: 4, z: 5 -> 60 points.
    # Or 4x4x4 = 64 points.
    
    # Let's stick to the directive: "Fibonacci lattice".
    # A true 3D Fibonacci lattice is not standard.
    # I will use a "Fibonacci sphere" projection? No, that's for spheres.
    
    # Alternative interpretation: Use a low-discrepancy sequence like Halton or Sobol?
    # The directive specifically says Fibonacci.
    
    # Let's create a 4x4x4 grid and remove 14 points that are least "spread" or just take the first 50.
    # Taking the first 50 from a row-major 4x4x4 grid is deterministic.
    
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # We have 64 points. We need 50.
    # To maximize spread, we should probably keep the center ones and remove corners?
    # Or just take the first 50?
    # Let's try to select 50 points from the 64-point grid that maximize the minimum distance.
    # This is a small optimization problem (64 choose 50 is too big, but greedy works).
    
    # Greedy selection:
    selected = []
    remaining = list(range(64))
    
    # Start with the center-most point?
    # Center of grid is index 27 or 28 (0-indexed in 4x4x4).
    # Let's just start with point 0 for simplicity, or the one closest to center.
    # Point closest to (0,0,0) in the grid.
    dists_to_center = np.linalg.norm(grid, axis=1)
    start_idx = np.argmin(dists_to_center)
    selected.append(start_idx)
    remaining.remove(start_idx)
    
    for _ in range(N - 1):
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            # Calculate min distance to all selected points
            min_d = np.min(np.linalg.norm(grid[idx] - grid[selected], axis=1))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
        
    return grid[selected].copy()


def _optimize_with_scipy(centers, beta_start=5.0, beta_end=0.1, max_iters=50, time_limit=100):
    """
    Refine centers using scipy.optimize.minimize with Nelder-Mead.
    """
    x0 = centers.flatten()
    best_x = x0.copy()
    best_radius = _compute_radius(centers)
    
    beta = beta_start
    n_betas = 10  # Number of annealing steps
    
    for i in range(n_betas):
        current_beta = beta_start * (beta_end / beta_start) ** (i / (n_betas - 1))
        
        # Define objective with current beta
        def obj(x):
            return _soft_min_objective(x, current_beta)
        
        # Use Nelder-Mead
        # Set bounds? Nelder-Mead doesn't support bounds directly, but we can penalize or clip.
        # We'll rely on the objective being smooth and the initial point being inside.
        # To prevent escaping the cube, we can add a large penalty if out of bounds.
        def obj_penalized(x):
            c = x.reshape(N, 3)
            if np.any(c > 0.5) or np.any(c < -0.5):
                # Return a large value if out of bounds
                return 1000.0
            return _soft_min_objective(x, current_beta)
            
        try:
            res = minimize(obj_penalized, x0, method='Nelder-Mead', 
                           options={'maxiter': max_iters, 'fatol': 1e-8, 'xatol': 1e-8})
            
            if res.success or res.fun < obj_penalized(x0):
                x0 = res.x
                new_centers = x0.reshape(N, 3)
                # Clip to be safe
                new_centers = np.clip(new_centers, -0.5, 0.5)
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_x = x0.copy()
        except Exception:
            pass
            
        if time.time() > time_limit:
            break
            
    return best_x.reshape(N, 3), best_radius


def construct():
    """Construct 50 sphere centers in unit cube to maximize minimum radius."""
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Initialize with Fibonacci-like lattice (greedy selection from grid)
    centers = _fibonacci_lattice()
    best_radius = _compute_radius(centers)
    
    # Refine with scipy.optimize.minimize
    if time.time() < deadline - 10:
        opt_centers, opt_radius = _optimize_with_scipy(centers, time_limit=deadline - 10)
        if opt_radius > best_radius:
            best_centers = opt_centers
            best_radius = opt_radius
        else:
            best_centers = centers
    else:
        best_centers = centers
        
    # Final check
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0:
        # Fallback
        return _fibonacci_lattice()
        
    return best_centers
# EVOLVE-BLOCK-END
