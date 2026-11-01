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

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq))

    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_grid(nx, ny, nz, noise_scale=0.005, seed=42):
    """Build a grid of points with noise."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers[:N]

def _local_refinement_nm(centers, max_iters=200):
    """
    Perform local refinement using Nelder-Mead on the raw radius (negative).
    """
    def neg_radius(c_flat):
        c = c_flat.reshape(-1, 3)
        return -_compute_radius(c)
    
    try:
        res = minimize(
            neg_radius,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        new_centers = res.x.reshape(-1, 3)
        new_centers = np.clip(new_centers, -0.5, 0.5)
        return new_centers
    except Exception:
        return centers

def _simulated_annealing(centers, deadline, initial_temp=0.01, cooling_rate=0.9995, min_temp=0.0001):
    """
    Simulated annealing to explore the configuration space.
    Uses Metropolis-Hastings moves.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    temp = initial_temp
    
    rng = np.random.default_rng(123)
    
    while time.time() < deadline and temp > min_temp:
        # Generate a random move: perturb one or a few spheres
        n_perturb = rng.integers(1, 4)
        idx = rng.choice(N, size=n_perturb, replace=False)
        
        new_centers = best_centers.copy()
        # Perturb positions
        step_size = temp * 0.5
        new_centers[idx] += rng.normal(0, step_size, (n_perturb, 3))
        
        # Keep within bounds
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        # Metropolis criterion
        delta = new_radius - current_radius
        if delta > 0:
            # Accept improvement
            best_centers = new_centers
            best_radius = new_radius
            current_radius = new_radius
        else:
            # Accept with probability exp(delta / temp)
            prob = np.exp(delta / temp)
            if rng.random() < prob:
                best_centers = new_centers
                current_radius = new_radius
        
        # Cool down
        temp *= cooling_rate
        
        # If we've been stuck for a while, maybe reheat slightly or break
        if current_radius < best_radius * 0.95 and temp < 0.005:
            break
            
    return best_centers, best_radius

def _penalty_objective(centers_flat, r, iu):
    """
    Penalty objective for maximizing radius r.
    Minimizes penalty = sum(max(0, 2r - d_ij)^2) + sum(max(0, r - dist_to_wall_i)^2)
    This directly optimizes the true bottleneck radius via a penalty method.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # Wall penalty: sum of (r - dist_to_wall)^2 for violated constraints
    dists_to_walls = 0.5 - np.abs(centers)
    wall_viol = np.maximum(0, r - dists_to_walls)
    wall_penalty = np.sum(wall_viol ** 2)
    
    # Pairwise penalty: sum of (2r - d_ij)^2 for violated constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    pair_viol = np.maximum(0, 2 * r - dists)
    pair_vals_unique = pair_viol[iu]
    pair_penalty = np.sum(pair_vals_unique ** 2)
    
    return wall_penalty + pair_penalty

def _penalty_optimize(centers, r_start, deadline, iu):
    """
    Iteratively optimize the penalty objective with increasing r.
    """
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    r = r_start
    
    # Increase r in small steps, optimizing penalty at each step
    r_step = 0.0005
    max_r = 0.12
    
    while time.time() < deadline - 5.0 and r < max_r:
        try:
            res = minimize(
                _penalty_objective,
                best_centers.flatten(),
                args=(r, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            # Check if the penalty is small enough to accept
            penalty = _penalty_objective(new_centers.flatten(), r, iu)
            if penalty < 1e-6:
                best_centers = new_centers
                actual_r = _compute_radius(best_centers)
                if actual_r > best_r:
                    best_r = actual_r
                r += r_step
            else:
                # If penalty is too large, try a smaller step
                r += r_step * 0.5
        except Exception:
            r += r_step * 0.5
    
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    
    # Initial guess: Grid
    best_centers = None
    best_radius = 0.0
    
    grids = [
        (5, 5, 4),
        (5, 4, 5),
        (4, 5, 5),
        (4, 4, 7),
        (7, 4, 4),
        (4, 7, 4),
    ]
    
    for nx, ny, nz in grids:
        if time.time() > deadline - 10.0:
            break
        centers = _build_grid(nx, ny, nz, noise_scale=0.005)
        
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    if best_centers is None:
        best_centers = _build_grid(5, 5, 4)
        best_radius = _compute_radius(best_centers)
        
    # Phase 1: Soft-min optimization with increasing beta
    beta = 10.0
    max_beta = 5000.0
    
    while time.time() < deadline - 30.0 and beta < max_beta:
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
            
        # Increase beta
        beta *= 1.3
        
    # Phase 2: Penalty-based optimization to directly target the radius
    if time.time() < deadline - 25.0:
        try:
            p_centers, p_radius = _penalty_optimize(
                best_centers,
                best_radius,
                deadline - 25.0,
                iu
            )
            if p_radius > best_radius:
                best_radius = p_radius
                best_centers = p_centers
        except Exception:
            pass
            
    # Phase 3: Simulated Annealing to escape local minima
    if time.time() < deadline - 15.0:
        try:
            sa_centers, sa_radius = _simulated_annealing(
                best_centers, 
                deadline - 15.0,
                initial_temp=0.005,
                cooling_rate=0.9998
            )
            if sa_radius > best_radius:
                best_radius = sa_radius
                best_centers = sa_centers
        except Exception:
            pass
            
    # Phase 4: Final local refinement with Nelder-Mead
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        try:
            refined = _local_refinement_nm(best_centers, max_iters=300)
            r_ref = _compute_radius(refined)
            if r_ref > best_radius:
                best_radius = r_ref
                best_centers = refined
        except Exception:
            pass
            
    # Phase 5: One more soft-min pass if time permits
    if time.time() < deadline - 2.0:
        try:
            x0 = best_centers.flatten()
            res = minimize(
                _soft_min_objective,
                x0,
                args=(10000.0, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
            
    return best_centers

# EVOLVE-BLOCK-END
