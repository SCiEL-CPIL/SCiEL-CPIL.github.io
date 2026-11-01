import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# Pre-compute upper triangular indices (avoid recomputation in objective)
_IU = np.triu_indices(N, k=1)

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    r(p) approx -1/beta * ln(sum(exp(-beta * x_k)))
    where x_k are the constraints (distances).
    We want to maximize r, so we minimize -r, which is 1/beta * ln(sum(exp(-beta * x_k))).
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = (dists / 2.0)[_IU]
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_initial_grid(nx, ny, nz, noise_scale=0.005):
    """Build a grid initialization with small noise to break symmetry."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    rng = np.random.default_rng(42)
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_bcc_init(noise_scale=0.003):
    """
    Build a body-centered cubic (BCC) inspired initialization.
    BCC has higher packing density than simple cubic.
    We use a 5x5x4 simple grid plus a 4x4x3 interstitial grid (offset by half-spacing),
    then select the 100 best positions.
    """
    # Simple grid: 5x5x4 = 100
    margin = 0.02
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # BCC interstitial: offset by half the spacing in each dimension
    # Spacing in each dimension
    dx = (0.5 - margin - (-0.5 + margin)) / (nx - 1)
    dy = (0.5 - margin - (-0.5 + margin)) / (ny - 1)
    dz = (0.5 - margin - (-0.5 + margin)) / (nz - 1)
    
    # Interstitial positions: offset by dx/2, dy/2, dz/2 from grid points
    # For BCC, interstitials are at (i+0.5, j+0.5, k+0.5) in grid coords
    # But we need to be careful about boundaries
    xi = xs[:-1] + dx / 2.0
    yi = ys[:-1] + dy / 2.0
    zi = zs[:-1] + dz / 2.0
    
    Xi, Yi, Zi = np.meshgrid(xi, yi, zi, indexing='ij')
    interstitial_centers = np.column_stack((Xi.ravel(), Yi.ravel(), Zi.ravel()))
    
    # Combine and select 100 with best local spacing
    # For simplicity, we'll create a mixed initialization:
    # Take some from grid and some from interstitial
    # 50 from grid, 50 from interstitial
    n_grid = 50
    n_inter = 50
    
    rng = np.random.default_rng(123)
    # Shuffle and pick
    grid_indices = rng.permutation(len(grid_centers))[:n_grid]
    inter_indices = rng.permutation(len(interstitial_centers))[:n_inter]
    
    centers = np.vstack([
        grid_centers[grid_indices],
        interstitial_centers[inter_indices]
    ])
    
    # Add small noise
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _optimize_from_initial(centers, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=8):
    """
    Run L-BFGS-B optimization with increasing beta from initial centers.
    Returns (best_centers, best_radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Create beta schedule: geometric progression from beta_start to beta_end
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
            )
            
            # Check if we improved
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        if time.time() > deadline - 5.0:
            break
    
    return best_centers, best_radius

def _polish(centers, deadline):
    """
    Final high-precision polish using very high beta and tight tolerances.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Very high beta for near-exact min
    for beta in [10000.0, 20000.0, 50000.0]:
        if time.time() > deadline - 3.0:
            break
        
        x0 = best_centers.flatten()
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 80}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    # Use 130s deadline to leave safety margin (150s limit)
    deadline = start_time + 130.0
    
    best_centers = None
    best_radius = 0.0
    
    # Phase 1: Try BCC-inspired initialization (better packing density)
    if time.time() < deadline - 60.0:
        for noise_scale in [0.002, 0.005, 0.01]:
            if time.time() > deadline - 30.0:
                break
            centers = _build_bcc_init(noise_scale)
            r_init = _compute_radius(centers)
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(
                centers, deadline - 30.0,
                beta_start=50.0, beta_end=5000.0, n_beta_steps=6
            )
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
    
    # Phase 2: Try grid configurations
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
    ]
    
    noise_scales = [0.003, 0.008]
    
    for nx, ny, nz in grid_configs:
        if time.time() > deadline - 20.0:
            break
        
        for noise_scale in noise_scales:
            if time.time() > deadline - 10.0:
                break
            
            centers = _build_initial_grid(nx, ny, nz, noise_scale)
            r_init = _compute_radius(centers)
            
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(
                centers, deadline - 10.0,
                beta_start=50.0, beta_end=5000.0, n_beta_steps=5
            )
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
    
    # Phase 3: Final polish with high-precision settings
    if best_centers is not None and time.time() < deadline - 5.0:
        polished_centers, polished_radius = _polish(best_centers, deadline)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished_centers
    
    # Fallback
    if best_centers is None:
        best_centers = _build_initial_grid(5, 5, 4, 0.01)
    
    return best_centers

# EVOLVE-BLOCK-END
