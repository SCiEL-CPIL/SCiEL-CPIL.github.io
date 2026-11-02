import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)

    # Pairwise distances using cdist for efficiency
    dists = cdist(centers, centers, metric='euclidean')
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)

    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective_epigraph(x, beta, iu, lambda_val=1.0):
    """
    Objective function for L-BFGS-B using epigraph formulation.
    x = [centers (N*3), t (1)]
    Minimizes t + lambda * soft_min(dists - t)
    """
    centers = x[:N*3].reshape(-1, 3)
    t = x[N*3]
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # We want to maximize the minimum clearance.
    # Let d_i be clearances. We want max r s.t. r <= d_i.
    # Equivalent to minimizing max(0, r - min(d_i)).
    # Using soft-min: r - soft_min(d_i - r) approximates the penalty.
    # A common smooth approximation for max(0, x) is softplus.
    # Or we can just minimize t + lambda * soft_min(t - d_i)
    # If t > min(d_i), soft_min(t - d_i) is positive and penalizes.
    # If t < min(d_i), soft_min(t - d_i) is negative, reducing the objective,
    # which pushes t up towards min(d_i).
    
    # Calculate LSE: 1/beta * ln(sum(exp(beta * (t - d_i))))
    # This approximates max_i (t - d_i) = t - min_i d_i
    # So obj = t + lambda * (t - min(d_i)) approx = (1+lambda)t - lambda*min(d_i)
    # This pushes t up until it hits min(d_i).
    
    vals = t - all_vals
    max_val = np.max(vals)
    sum_exp = np.sum(np.exp(vals - max_val))
    soft_min_val = (max_val + np.log(sum_exp)) / beta
    
    return t + lambda_val * soft_min_val

def _build_fcc_init(seed=42):
    """
    Build a Face-Centered Cubic (FCC) initialization.
    """
    rng = np.random.default_rng(seed)
    points = []
    
    # FCC lattice points in a 4x4x4 supercell
    # We want 100 points.
    # 4x4x4 grid of unit cells gives 64 centers.
    # Let's use a denser grid or different scaling.
    
    # Try 5x5x4 grid points? 100 points.
    # Simple grid 5x5x4
    nx, ny, nz = 5, 5, 4
    margin = 0.05
    x_min, x_max = -0.5 + margin, 0.5 - margin
    dx = (x_max - x_min) / (nx - 1) if nx > 1 else 0
    dy = (x_max - x_min) / (ny - 1) if ny > 1 else 0
    dz = (x_max - x_min) / (nz - 1) if nz > 1 else 0
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = x_min + i * dx
                y = y_min + j * dy if False else x_min + j * dy # fix
                z = x_min + k * dz
                points.append([x, y, z])
                
    # Wait, 5*5*4 = 100. Perfect.
    # Add noise
    centers = np.array(points)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_bcc_init(seed=42):
    """
    Build a body-centered cubic (BCC) initialization.
    """
    rng = np.random.default_rng(seed)
    
    # BCC in a 4x4x4 block gives 4^3 * 2 = 128 points. We take 100.
    nx, ny, nz = 4, 4, 4
    points = []
    
    # Spacing to fit in [-0.5, 0.5]
    # Total width 1.0.
    # 4 cells wide. Cell width 0.25.
    # Centers at 0.125, 0.375...
    
    # Let's just use a simple grid with perturbation
    # Grid 4x5x5 = 100 points
    nx, ny, nz = 4, 5, 5
    x_min, x_max = -0.48, 0.48
    y_min, y_max = -0.48, 0.48
    z_min, z_max = -0.48, 0.48
    
    dx = (x_max - x_min) / (nx - 1) if nx > 1 else 0
    dy = (y_max - y_min) / (ny - 1) if ny > 1 else 0
    dz = (z_max - z_min) / (nz - 1) if nz > 1 else 0
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = x_min + i * dx
                y = y_min + j * dy
                z = z_min + k * dz
                points.append([x, y, z])
                
    centers = np.array(points[:N])
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(n_dims=3, seed=42):
    """
    Build a Halton sequence initialization.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, N + 1):
        point = [halton(i, base) for base in [2, 3, 5]]
        # Scale to [-0.48, 0.48]
        point = [p * 0.96 - 0.48 for p in point]
        points.append(point)
    
    centers = np.array(points)
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _force_directed_refinement(centers, deadline, iu, n_steps=500):
    """
    Refine centers using force-directed repulsion with vectorized operations.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_centers = centers.copy()
    current_radius = best_radius
    
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    
    step_size = 0.02
    cooling = 0.99
    
    for step in range(n_steps):
        if time.time() > deadline - 10.0:
            break
            
        # Compute pairwise differences and distances
        # diff[i, j, :] = centers[i] - centers[j]
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dists_mat = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal
        np.fill_diagonal(dists_mat, np.inf)
        
        # Target separation: slightly more than current best radius * 2
        # Use a soft target to allow some compression if needed
        target = 2.1 * best_radius
        
        # Force is proportional to (target - dist) / dist * direction
        # Only apply if dist < target
        mask = dists_mat < target
        
        # Safe distances to avoid div 0
        safe_dists = np.where(mask, dists_mat, 1.0)
        
        # Strength: (target - dist)
        strengths = np.where(mask, (target - dists_mat), 0.0)
        
        # Direction: diff / dist
        dir_mat = diff / safe_dists_mat[:, :, np.newaxis] if False else diff / safe_dists[:, :, np.newaxis]
        
        # Force vector
        forces = np.sum(strengths[:, :, np.newaxis] * dir_mat, axis=1)
        
        # Wall forces: Push away from walls if close
        wall_dist = 0.5 - np.abs(current_centers)
        wall_mask = wall_dist < best_radius
        # Force towards center: -centers
        wall_forces = np.where(wall_mask, -current_centers, 0.0)
        # Normalize wall forces?
        wall_norm = np.linalg.norm(wall_forces, axis=1, keepdims=True)
        wall_norm[wall_norm == 0] = 1.0
        wall_forces = wall_forces / wall_norm * np.where(wall_mask, 1.0, 0.0)
        
        forces += 0.5 * wall_forces
        
        # Update positions
        current_centers += step_size * forces
        current_centers = np.clip(current_centers, -0.499, 0.499)
        
        new_radius = _compute_radius(current_centers)
        
        if new_radius > current_radius:
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = current_centers.copy()
        else:
            # Revert if worse
            current_centers = best_centers.copy()
            
        # Cool down
        step_size *= cooling
        
    return best_centers, best_radius

def _optimize_sequence(centers, deadline, iu, beta_start=10.0, beta_end=50000.0, n_steps=10):
    """
    Run L-BFGS-B optimization with increasing beta.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    # Initial t estimate
    t_init = best_radius * 0.9
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
            
        x0 = np.concatenate([best_centers.flatten(), [t_init]])
        
        try:
            res = minimize(
                _soft_min_objective_epigraph,
                x0,
                args=(beta, iu, 1.0),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3) + [(0, 0.5)],
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            new_centers = res.x[:N*3].reshape(-1, 3)
            new_t = res.x[N*3]
            
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
                t_init = r * 0.95
        except Exception:
            pass
            
    return best_centers, best_radius

def _final_polish(centers, deadline, iu):
    """
    High-precision polish pass with very high beta.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    high_betas = [10000, 50000, 100000]
    t_init = best_radius * 0.99
    
    for beta in high_betas:
        if time.time() > deadline - 2.0:
            break
            
        x0 = np.concatenate([best_centers.flatten(), [t_init]])
        
        try:
            res = minimize(
                _soft_min_objective_epigraph,
                x0,
                args=(beta, iu, 5.0), # Higher lambda for sharper minimum
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3) + [(0, 0.5)],
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            
            new_centers = res.x[:N*3].reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
                t_init = r * 0.99
        except Exception:
            pass
            
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    # Phase 1: Generate diverse initializations
    inits = []
    
    # FCC/Grid Initialization
    grid_centers = _build_fcc_init(seed=42)
    inits.append(grid_centers)
    
    # BCC/Grid Initialization
    bcc_centers = _build_bcc_init(seed=42)
    inits.append(bcc_centers)
    
    # Halton Initialization
    halton_centers = _build_halton_init(seed=42)
    inits.append(halton_centers)
    
    # Phase 2: For each initialization, run optimization
    for i, init_centers in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        r_init = _compute_radius(init_centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init_centers.copy()
        
        # Optimization 1: Soft-Min L-BFGS
        opt_centers, opt_radius = _optimize_sequence(
            init_centers, 
            deadline, 
            iu,
            beta_start=5.0, 
            beta_end=10000.0, 
            n_steps=6
        )
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers.copy()
            
        # Optimization 2: Force-Directed Refinement
        if time.time() < deadline - 20.0:
            fd_centers, fd_radius = _force_directed_refinement(
                best_centers, 
                deadline, 
                iu,
                n_steps=300
            )
            if fd_radius > best_radius:
                best_radius = fd_radius
                best_centers = fd_centers.copy()
                
        # Optimization 3: Final Polish
        if time.time() < deadline - 5.0:
            polished_centers, polished_radius = _final_polish(best_centers, deadline, iu)
            if polished_radius > best_radius:
                best_radius = polished_radius
                best_centers = polished_centers.copy()

    if best_centers is None:
        best_centers = _build_fcc_init(seed=42)
        best_radius = _compute_radius(best_centers)
        
    return best_centers

# EVOLVE-BLOCK-END
