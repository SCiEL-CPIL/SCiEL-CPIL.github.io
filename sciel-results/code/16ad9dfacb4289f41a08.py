import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 100

_start_time = time.time()
DEADLINE = 145.0  # Safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _start_time)

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
        
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
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    pair_vals = dists / 2.0
    # Exclude diagonal
    pair_vals_unique = pair_vals[np.triu_indices(N, k=1)]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    # To avoid overflow, subtract max
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _soft_min_grad(centers_flat, beta):
    """
    Gradient of the soft-min objective.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    pair_vals = dists / 2.0
    # Exclude diagonal
    pair_vals_unique = pair_vals[np.triu_indices(N, k=1)]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Soft-min: f = (1/beta) * log(sum(exp(-beta * x)))
    # df/dx_i = -exp(-beta * x_i) / sum(exp(-beta * x_j))
    #          = -exp(-beta * (x_i - max_x)) / sum(exp(-beta * (x_j - max_x)))
    
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    shifted = neg_beta_x - max_val
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Weights: w_i = exp(-beta * x_i) / sum(exp(-beta * x_j))
    weights = exp_shifted / sum_exp
    
    grad_flat = np.zeros_like(all_vals)
    grad_flat[:] = -weights
    
    # Now map back to centers
    grad_centers = np.zeros_like(centers)
    
    # Wall part: grad_flat[:N*3] corresponds to wall distances
    # dists_to_walls[i,k] = 0.5 - |centers[i,k]|
    # d(dists_to_walls[i,k])/d(centers[i,k]) = -sign(centers[i,k])
    wall_grad = grad_flat[:N*3].reshape(N, 3)
    signs = np.sign(centers)
    signs[np.abs(centers) < 1e-15] = 1.0  # avoid zero
    grad_centers += -wall_grad * signs
    
    # Pair part: grad_flat[N*3:] corresponds to pair distances / 2
    # For pair (i,j), dist/2 = 0.5 * ||c_i - c_j||
    # d(dist/2)/d(c_i) = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # d(dist/2)/d(c_j) = 0.5 * (c_j - c_i) / ||c_i - c_j||
    
    pair_grad_flat = grad_flat[N*3:]
    triu_i, triu_j = np.triu_indices(N, k=1)
    
    for idx, (i, j) in enumerate(zip(triu_i, triu_j)):
        diff = centers[i] - centers[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-15:
            continue
        unit_vec = diff / dist
        # grad for c_i: -w * 0.5 * unit_vec (since d(dist/2)/d(c_i) = 0.5*unit_vec, and we have -w)
        # Actually: df/d(c_i) = sum over pairs involving i of (-w_pair) * d(pair_val)/d(c_i)
        # d(pair_val)/d(c_i) = 0.5 * (c_i - c_j)/||c_i - c_j|| = 0.5 * unit_vec
        # So contribution to grad_centers[i] += -w_pair * 0.5 * unit_vec
        # Contribution to grad_centers[j] += -w_pair * 0.5 * (-unit_vec) = w_pair * 0.5 * unit_vec
        
        w = pair_grad_flat[idx]
        grad_centers[i] += -w * 0.5 * unit_vec
        grad_centers[j] += w * 0.5 * unit_vec
    
    return grad_centers.flatten()

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
        # This function is expected to be called with correct dimensions
        pass

    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        centers += rng.normal(0, noise_scale, centers.shape)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_bcc_init(seed=42):
    """
    Build a BCC (body-centered cubic) inspired initialization.
    BCC has 2 points per unit cell: corners + center.
    For a 4x4x4 grid of cells, we get 4*4*4*2 = 128 points, too many.
    For 3x3x4 grid: 3*3*4*2 = 72, too few.
    Let's use a 4x4x3 grid with BCC: 4*4*3*2 = 96, close to 100.
    Or we can use a mixed approach.
    
    Alternative: Use a BCC lattice scaled to fit in the cube.
    BCC primitive vectors: a/2*(1,1,1), a/2*(-1,1,1), a/2*(1,-1,1)
    With lattice constant a, the BCC has 2 atoms per cubic cell of side a.
    
    For 100 spheres, we need 50 BCC cells.
    50 = 5*5*2, so a 5x5x2 supercell of BCC.
    Supercell size: 5a x 5a x 2a
    To fit in [-0.5, 0.5]^3: 5a <= 1, 5a <= 1, 2a <= 1 => a <= 0.4
    With a = 0.4: supercell is 2.0 x 2.0 x 0.8, too big in x,y.
    
    Let's just create a reasonable BCC-like pattern.
    """
    rng = np.random.default_rng(seed)
    
    # Create a 4x4x4 grid of small cubes, each with a BCC-like arrangement
    # 4x4x4 = 64 small cubes, each with ~1.56 spheres on average
    # This is getting complicated. Let's use a simpler approach:
    # Start with a 5x5x4 grid = 100 points, then add BCC-like offsets
    
    # Actually, let's just create a distorted grid with BCC-inspired offsets
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Add small BCC-like perturbations
    offsets = rng.normal(0, 0.02, centers.shape)
    centers += offsets
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_halton_init(seed=42):
    """
    Generate N points using a Halton sequence (low discrepancy).
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
    
    # Add slight noise to break symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _polish_repulsion(centers, deadline, r_current):
    """
    Local search using repulsion forces.
    Uses KDTree for efficient neighbor finding.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.005
    decay = 0.95
    min_step = 1e-6
    
    while _time_left() > 1.0 and step_size > min_step:
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Wall forces
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            limit = r_current
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[np.abs(centers[mask, k]) < 1e-15] = 1.0
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
        
        # Pairwise forces
        tree = cKDTree(centers)
        pairs = tree.query_pairs(2 * r_current)
        
        for i, j in pairs:
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-9:
                forces[i] += np.random.normal(0, 1e-3, 3)
                forces[j] -= np.random.normal(0, 1e-3, 3)
                continue
            
            limit = 2 * r_current
            if dist < limit:
                strength = (limit - dist) / limit
                unit_vec = diff / dist
                forces[i] += strength * unit_vec
                forces[j] -= strength * unit_vec
        
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        centers += step_size * forces
        centers = np.clip(centers, -0.5, 0.5)
        
        r_new = _compute_radius(centers)
        if r_new > best_r:
            best_r = r_new
            best_centers = centers.copy()
        
        step_size *= decay
        
    return best_centers, best_r

def _polish_high_beta(centers, deadline):
    """
    Final polish with very high beta soft-min and tight tolerances.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    beta_schedule = [2000, 5000, 10000]
    
    for beta in beta_schedule:
        if _time_left() < 5.0:
            break
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                jac=_soft_min_grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
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

def _optimize_from_initial(centers, deadline):
    """
    Run optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Soft-min optimization with increasing beta
    beta_schedule = [50, 100, 200, 500, 1000]
    
    for beta in beta_schedule:
        if _time_left() < 15.0:
            break
            
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 25}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
                
        except Exception:
            pass
            
    # Phase 2: Polish with repulsion
    if _time_left() > 5.0:
        polish_centers, polish_radius = _polish_repulsion(best_centers, deadline, best_radius)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
            
    # Phase 3: High-beta polish
    if _time_left() > 3.0:
        hb_centers, hb_radius = _polish_high_beta(best_centers, deadline)
        if hb_radius > best_radius:
            best_radius = hb_radius
            best_centers = hb_centers
            
    return best_centers, best_radius

def construct():
    best_centers = None
    best_radius = 0.0
    
    # Initializations - reduced number to allow longer optimization per run
    grid_configs = [
        (4, 5, 5),
        (5, 4, 5),
        (5, 5, 4),
        (10, 5, 2),
        (5, 10, 2),
        (5, 2, 10),
    ]
    
    for nx, ny, nz in grid_configs:
        if _time_left() < 20.0:
            break
        
        # Try with moderate noise only
        for noise in [0.01]:
            if _time_left() < 10.0:
                break
            
            centers = _build_grid_init(nx, ny, nz, noise, seed=nx*100+ny*10+nz)
            r_init = _compute_radius(centers)
            
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
            
            opt_centers, opt_radius = _optimize_from_initial(centers, DEADLINE)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    # BCC-inspired init
    if _time_left() > 20.0:
        centers = _build_bcc_init()
        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
            
        opt_centers, opt_radius = _optimize_from_initial(centers, DEADLINE)
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers

    # Halton
    if _time_left() > 20.0:
        centers = _build_halton_init()
        r_init = _compute_radius(centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
            
        opt_centers, opt_radius = _optimize_from_initial(centers, DEADLINE)
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
            
    # Random
    if _time_left() > 15.0:
        rng = np.random.default_rng(123)
        for _ in range(2):
            if _time_left() < 10.0:
                break
            centers = rng.uniform(-0.45, 0.45, (N, 3))
            r_init = _compute_radius(centers)
            if r_init > best_radius:
                best_radius = r_init
                best_centers = centers.copy()
                
            opt_centers, opt_radius = _optimize_from_initial(centers, DEADLINE)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers

    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
