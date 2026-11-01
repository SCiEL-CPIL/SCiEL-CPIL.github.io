import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    Uses cKDTree for fast nearest-neighbor search.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)

    # Pairwise distances via KDTree (find 2nd nearest neighbor)
    try:
        tree = cKDTree(centers)
        # Query for 2 nearest neighbors (self + 1 other)
        dists, _ = tree.query(centers, k=2)
        min_pair_dist = np.min(dists[:, 1])  # exclude self (k=1 is nearest other)
    except Exception:
        # Fallback to brute force
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

def _build_bcc_init(n=100, noise_scale=0.005, seed=42):
    """
    Build a BCC (Body-Centered Cubic) inspired initialization.
    We create a 10x10x1 grid of BCC unit cells, giving 2*10*10*1 = 200 points,
    then randomly select 100.
    """
    nx, ny, nz = 10, 10, 1
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    all_centers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                all_centers.append([xs[i], ys[j], zs[k]])
                if i < nx-1 and j < ny-1 and k < nz-1:
                    all_centers.append([(xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2, (zs[k]+zs[k+1])/2])
                elif i < nx-1 and j < ny-1:
                    all_centers.append([(xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2, zs[k]])
                elif i < nx-1:
                    all_centers.append([(xs[i]+xs[i+1])/2, ys[j], zs[k]])
                elif j < ny-1:
                    all_centers.append([xs[i], (ys[j]+ys[j+1])/2, zs[k]])
    
    all_centers = np.array(all_centers)
    rng = np.random.default_rng(seed)
    
    if len(all_centers) >= n:
        idx = rng.choice(len(all_centers), n, replace=False)
        centers = all_centers[idx]
    else:
        centers = all_centers
        while len(centers) < n:
            centers = np.vstack([centers, centers[0]])
    
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(n=100, seed=42):
    """
    Build a Halton sequence based initialization for good space-filling properties.
    """
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 3))
    for i in range(1, n+1):
        centers[i-1, 0] = -0.5 + halton(i, 2)
        centers[i-1, 1] = -0.5 + halton(i, 3)
        centers[i-1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

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

def _simulated_annealing(centers, deadline, initial_temp=0.01, cooling_rate=0.9997, min_temp=0.0001):
    """
    Simulated annealing to explore the configuration space.
    Uses Metropolis-Hastings moves with adaptive step size.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    current_centers = centers.copy()
    temp = initial_temp
    
    rng = np.random.default_rng(123)
    
    # Track acceptance rate to adapt step size
    accept_count = 0
    total_count = 0
    step_scale = 1.0
    
    while time.time() < deadline and temp > min_temp:
        # Generate a random move: perturb one or a few spheres
        n_perturb = rng.integers(1, 4)
        idx = rng.choice(N, size=n_perturb, replace=False)
        
        new_centers = current_centers.copy()
        # Perturb positions with adaptive step size
        step_size = temp * 0.5 * step_scale
        new_centers[idx] += rng.normal(0, step_size, (n_perturb, 3))
        
        # Keep within bounds
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        total_count += 1
        
        # Metropolis criterion
        delta = new_radius - current_radius
        if delta > 0:
            # Accept improvement
            current_centers = new_centers
            current_radius = new_radius
            accept_count += 1
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        else:
            # Accept with probability exp(delta / temp)
            if delta < -1e-15:
                prob = np.exp(delta / temp)
                if rng.random() < prob:
                    current_centers = new_centers
                    current_radius = new_radius
                    accept_count += 1
        
        # Cool down
        temp *= cooling_rate
        
        # Adapt step size every 200 iterations
        if total_count % 200 == 0:
            accept_rate = accept_count / max(total_count, 1)
            # Target ~50% acceptance rate
            if accept_rate > 0.6:
                step_scale *= 1.2
            elif accept_rate < 0.3:
                step_scale *= 0.8
            step_scale = np.clip(step_scale, 0.1, 5.0)
            accept_count = 0
            total_count = 0
        
        # If we've been stuck for a while, break
        if current_radius < best_radius * 0.95 and temp < 0.003:
            break
            
    return best_centers, best_radius

def _penalty_objective(centers_flat, r, iu):
    """
    Penalty objective for maximizing radius r.
    """
    centers = centers_flat.reshape(-1, 3)
    
    dists_to_walls = 0.5 - np.abs(centers)
    wall_viol = np.maximum(0, r - dists_to_walls)
    wall_penalty = np.sum(wall_viol ** 2)
    
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
            
            penalty = _penalty_objective(new_centers.flatten(), r, iu)
            if penalty < 1e-6:
                best_centers = new_centers
                actual_r = _compute_radius(best_centers)
                if actual_r > best_r:
                    best_r = actual_r
                r += r_step
            else:
                r += r_step * 0.5
        except Exception:
            r += r_step * 0.5
    
    return best_centers, best_r

def _optimize_from_initial(centers, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=6):
    """
    Run L-BFGS-B optimization with increasing beta from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    iu = np.triu_indices(N, k=1)
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
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
        
        if time.time() > deadline - 5.0:
            break
    
    return best_centers, best_radius

def _polish(centers, deadline):
    """
    Final high-precision polish pass using tighter tolerances.
    Uses a more aggressive beta ramp and multiple polish stages.
    """
    iu = np.triu_indices(N, k=1)
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Use very high beta for final polish - more aggressive ramp
    betas = [5000.0, 10000.0, 20000.0, 50000.0, 100000.0]
    for beta in betas:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _soft_min_objective,
                best_centers.flatten(),
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    # Nelder-Mead final refinement with more iterations
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        try:
            refined = _local_refinement_nm(best_centers, max_iters=500)
            r_ref = _compute_radius(refined)
            if r_ref > best_radius:
                best_radius = r_ref
                best_centers = refined
        except Exception:
            pass
    
    # One more soft-min polish after NM
    remaining_time = deadline - time.time()
    if remaining_time > 3.0:
        try:
            res = minimize(
                _soft_min_objective,
                best_centers.flatten(),
                args=(200000.0, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
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
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    
    # Initial guess: Multiple diverse initializations
    best_centers = None
    best_radius = 0.0
    
    inits = []
    
    # Grid inits
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
        inits.append(centers)
        
    # BCC init
    inits.append(_build_bcc_init())
    
    # Halton init
    inits.append(_build_halton_init())
    
    # Run optimization from each initialization
    for centers in inits:
        if time.time() > deadline - 30.0:
            break
        
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
        
        # Run soft-min optimization from this initialization
        opt_centers, opt_radius = _optimize_from_initial(
            centers, 
            deadline - 30.0,
            beta_start=50.0,
            beta_end=5000.0,
            n_beta_steps=4
        )
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
    
    if best_centers is None:
        best_centers = _build_grid(5, 5, 4)
        best_radius = _compute_radius(best_centers)
        
    # Penalty-based optimization
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
            
    # Simulated Annealing - give it more time with improved adaptive step
    if time.time() < deadline - 20.0:
        try:
            sa_centers, sa_radius = _simulated_annealing(
                best_centers, 
                deadline - 20.0,
                initial_temp=0.008,
                cooling_rate=0.9997
            )
            if sa_radius > best_radius:
                best_radius = sa_radius
                best_centers = sa_centers
        except Exception:
            pass
            
    # Final polish - give it more time
    if time.time() < deadline - 1.0:
        try:
            polished_centers, polished_radius = _polish(best_centers, deadline - 1.0)
            if polished_radius > best_radius:
                best_radius = polished_radius
                best_centers = polished_centers
        except Exception:
            pass
            
    return best_centers

# EVOLVE-BLOCK-END
