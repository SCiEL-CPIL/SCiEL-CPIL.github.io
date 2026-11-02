import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

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
    dists = cdist(centers, centers, metric='euclidean')
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    # This approximates min(x)
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_grid_init(nx, ny, nz, noise_scale=0.0, seed=42):
    """Build a grid initialization."""
    if nx == 1: xs = np.array([0.0])
    else: xs = np.linspace(-0.5, 0.5, nx)
    
    if ny == 1: ys = np.array([0.0])
    else: ys = np.linspace(-0.5, 0.5, ny)
    
    if nz == 1: zs = np.array([0.0])
    else: zs = np.linspace(-0.5, 0.5, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if len(centers) != N:
        # Pad or truncate to get exactly N points
        if len(centers) < N:
            rng = np.random.default_rng(seed)
            extra = rng.uniform(-0.45, 0.45, (N - len(centers), 3))
            centers = np.vstack([centers, extra])
        else:
            centers = centers[:N]

    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        centers += rng.normal(0, noise_scale, centers.shape)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_bcc_init():
    """
    Build a BCC (body-centered cubic) lattice initialization.
    Using a 5x5x4 grid of points is a good approximation for N=100.
    """
    # Simple grid 5x5x4 = 100 points
    centers = _build_grid_init(5, 5, 4, noise_scale=0.01, seed=42)
    return centers

def _build_sobol_init(seed=42):
    """
    Generate N points using a Sobol sequence.
    """
    from scipy.stats import qmc
    sobol = qmc.Sobol(d=3, scramble=True, seed=seed)
    points = sobol.random(N)
    centers = points - 0.5
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(seed=42):
    """
    Generate N points using a Halton sequence.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    bases = [2, 3, 5]
    points = []
    for i in range(1, N + 1):
        pt = [halton(i, b) for b in bases]
        pt_mapped = [p - 0.5 for p in pt]
        points.append(pt_mapped)
        
    centers = np.array(points)
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _coordinate_descent(centers, max_iter=100, step_size=0.01):
    """
    Coordinate Descent polish.
    Iterates over each sphere and moves it in the direction that maximizes the minimum distance.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    
    for _ in range(max_iter):
        if _time_left() < 5.0:
            break
            
        improved = False
        for i in range(N):
            if _time_left() < 5.0:
                break
                
            c_i = centers[i]
            
            # Calculate min distance to others and walls for current position
            # This is expensive if done naively, but we only do it for one point at a time
            
            # Try moving in 6 directions ( +/- x, +/- y, +/- z ) and maybe diagonals?
            # Just +/- x, y, z is a good start for CD.
            best_pos = c_i
            best_r = _compute_radius(centers)
            
            for dim in range(3):
                for sign in [-1, 1]:
                    new_c = c_i.copy()
                    new_c[dim] += sign * step_size
                    # Clip to bounds
                    new_c = np.clip(new_c, -0.4999, 0.4999)
                    
                    # Check if this move improves the global radius
                    # We need to check the min distance involving i
                    # Dist to walls
                    wall_dists = 0.5 - np.abs(new_c)
                    min_wall = np.min(wall_dists)
                    
                    # Dist to other centers
                    if i > 0:
                        dists_prev = np.linalg.norm(centers[:i] - new_c, axis=1)
                    else:
                        dists_prev = np.array([])
                    if i < N - 1:
                        dists_next = np.linalg.norm(centers[i+1:] - new_c, axis=1)
                    else:
                        dists_next = np.array([])
                    
                    all_dists = np.concatenate([dists_prev, dists_next])
                    min_pair = np.min(all_dists) / 2.0 if len(all_dists) > 0 else np.inf
                    
                    new_r = min(min_wall, min_pair)
                    
                    if new_r > best_r + 1e-12:
                        best_r = new_r
                        best_pos = new_c
                        improved = True
            
            if improved:
                centers[i] = best_pos
        
        if not improved:
            # If no improvement, try smaller step? Or break?
            # For now, break if no improvement in a full pass
            break
            
    return centers

def _polish_high_beta(centers, deadline, iu):
    """
    Final high-precision polish using L-BFGS-B with very high beta.
    """
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    
    # Use a very high beta for near-exact min approximation
    beta = 2000.0
    
    x0 = best_centers.flatten()
    
    try:
        res = minimize(
            _soft_min_objective,
            x0,
            args=(beta, iu),
            method='L-BFGS-B',
            bounds=[(-0.4999, 0.4999)] * (N * 3),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 20}
        )
        
        new_centers = res.x.reshape(-1, 3)
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        r = _compute_radius(new_centers)
        if r > best_r:
            best_r = r
            best_centers = new_centers
            
    except Exception:
        pass
        
    return best_centers, best_r

def _optimize_from_initial(centers, deadline, iu, time_budget=20.0):
    """
    Run optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    
    # Phase 1: Soft-min optimization (Medium Beta)
    beta_start = 50.0
    beta_end = 500.0
    n_beta_steps = 3
    
    for step in range(n_beta_steps):
        if _time_left() < 5.0 or (time.time() - start_t) > time_budget - 5.0:
            break
            
        beta = beta_start * (beta_end / beta_start) ** (step / (n_beta_steps - 1))
        
        x0 = best_centers.flatten()
        
        try:
            max_iter = 100
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 15}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
                
        except Exception:
            pass
            
    # Phase 2: Coordinate Descent
    if _time_left() > 5.0 and (time.time() - start_t) < time_budget - 2.0:
        cd_centers = _coordinate_descent(best_centers, max_iter=20, step_size=0.005)
        cd_radius = _compute_radius(cd_centers)
        if cd_radius > best_radius:
            best_radius = cd_radius
            best_centers = cd_centers

    # Phase 3: High-Precision Polish
    if _time_left() > 3.0:
        polish_centers, polish_radius = _polish_high_beta(best_centers, deadline, iu)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
            
    return best_centers, best_radius

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    def try_init(centers, time_budget=15.0):
        nonlocal best_centers, best_radius
        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
        
        if _time_left() > time_budget + 2.0:
            opt_centers, opt_radius = _optimize_from_initial(centers, _DEADLINE, iu, time_budget)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
        else:
            # Fallback: Quick polish if little time left
            if _time_left() > 3.0:
                polish_centers, polish_radius = _polish_high_beta(centers, _DEADLINE, iu)
                if polish_radius > best_radius:
                    best_radius = polish_radius
                    best_centers = polish_centers

    # Initializations
    # 1. Grid (5x5x4) - Good aspect ratio
    if _time_left() > 30.0:
        centers = _build_grid_init(5, 5, 4, 0.01, seed=42)
        try_init(centers, time_budget=30.0)
        
    # 2. Sobol
    if _time_left() > 20.0:
        centers = _build_sobol_init()
        try_init(centers, time_budget=20.0)
        
    # 3. Halton
    if _time_left() > 15.0:
        centers = _build_halton_init()
        try_init(centers, time_budget=15.0)
        
    # 4. Random
    if _time_left() > 10.0:
        rng = np.random.default_rng(123)
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        try_init(centers, time_budget=10.0)
        
    # Final Global Polish
    if best_centers is not None and _time_left() > 5.0:
        # Run CD again on the best found
        cd_centers = _coordinate_descent(best_centers, max_iter=50, step_size=0.002)
        cd_radius = _compute_radius(cd_centers)
        if cd_radius > best_radius:
            best_radius = cd_radius
            best_centers = cd_centers
            
        # And finally, high-beta polish
        if _time_left() > 2.0:
            polish_centers, polish_radius = _polish_high_beta(best_centers, _DEADLINE, iu)
            if polish_radius > best_radius:
                best_radius = polish_radius
                best_centers = polish_centers

    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
