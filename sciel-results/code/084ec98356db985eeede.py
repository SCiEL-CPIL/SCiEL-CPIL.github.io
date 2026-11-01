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
    BCC has 2 points per unit cell. For 100 points, use 50 cells.
    Arrange in a grid of cells within the cube.
    """
    # Use a 5x5x2 grid of BCC cells
    # Each cell has 2 points: corner and body center
    nx, ny, nz = 5, 5, 2
    n_cells = nx * ny * nz  # 50
    # Scale to fit in cube
    cell_size = 1.0 / max(nx, ny, nz)
    
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Corner point
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                points.append([cx, cy, cz])
                # Body center
                points.append([cx + 0.5/nx, cy + 0.5/ny, cz + 0.5/nz])
    
    centers = np.array(points[:N])
    
    # Add slight noise
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(seed=42):
    """
    Generate N points using a Halton sequence (low discrepancy).
    This provides a more uniform distribution than a grid.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    # We need 3 bases, e.g., 2, 3, 5
    bases = [2, 3, 5]
    points = []
    for i in range(1, N + 1):
        pt = [halton(i, b) for b in bases]
        # Map [0,1] to [-0.5, 0.5]
        pt_mapped = [p - 0.5 for p in pt]
        points.append(pt_mapped)
        
    centers = np.array(points)
    
    # Add slight noise to break symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_sobol_init(seed=42):
    """
    Generate N points using a Sobol sequence (low discrepancy).
    Sobol sequences have better equidistribution properties than Halton.
    """
    from scipy.stats import qmc
    
    sobol = qmc.Sobol(d=3, scramble=True, seed=seed)
    points = sobol.random(N)
    
    # Map [0,1] to [-0.5, 0.5]
    centers = points - 0.5
    
    # Add slight noise to break symmetry
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _polish_repulsion(centers, deadline, iu):
    """
    Local search using repulsion forces with KD-tree for fast neighbor queries.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    step_size = 0.002
    decay = 0.985
    min_step = 1e-7
    margin = 1.02
    
    no_improve_count = 0
    max_no_improve = 80
    
    while _time_left() > 1.0 and step_size > min_step and no_improve_count < max_no_improve:
        r = _compute_radius(centers)
        if r > best_r + 1e-12:
            best_r = r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Wall forces
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            limit = max(r_current * margin, 0.01)
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
                
        # Pairwise forces using KD-tree for efficiency
        tree = cKDTree(centers)
        limit = 2 * max(r_current * margin, 0.01)
        
        # Query neighbors within limit
        neighbors = tree.query_ball_point(centers, r=limit)
        
        for i in range(N):
            for j in neighbors[i]:
                if j <= i:
                    continue
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-9 and dist < limit:
                    strength = (limit - dist) / limit
                    forces[i] += strength * diff / dist
                    forces[j] -= strength * diff / dist
        
        total_forces = forces
        
        # Normalize forces to prevent large jumps
        norms = np.linalg.norm(total_forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        total_forces = total_forces / norms
        
        centers += step_size * total_forces
        centers = np.clip(centers, -0.4999, 0.4999)
        
        step_size *= decay
        
    return best_centers, best_r

def _optimize_from_initial(centers, deadline, iu, time_budget=20.0):
    """
    Run optimization from initial centers.
    time_budget: max seconds to spend on this initialization.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    remaining = time_budget
    
    # Phase 1: Soft-min optimization
    beta_start = 30.0
    beta_end = 1500.0
    n_beta_steps = 5
    
    if remaining > 8.0:
        for step in range(n_beta_steps):
            if _time_left() < 2.0 or (time.time() - start_t) > remaining - 2.0:
                break
                
            beta = beta_start * (beta_end / beta_start) ** (step / (n_beta_steps - 1))
            
            x0 = best_centers.flatten()
            
            try:
                # Reduce iterations as we run out of time
                max_iter = 100 if step < 2 else 50
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
    
    # Phase 2: Polish with repulsion
    if _time_left() > 3.0 and (time.time() - start_t) < remaining - 1.0:
        polish_deadline = min(deadline, start_t + remaining - 0.5)
        polish_centers, polish_radius = _polish_repulsion(best_centers, polish_deadline, iu)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
            
    return best_centers, best_radius

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0  # Safety margin: evaluator kills at 150s
    
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    def try_init(centers, time_budget=18.0):
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
            # Just do a quick polish if we have minimal time
            if _time_left() > 3.0:
                polish_centers, polish_radius = _polish_repulsion(centers, _DEADLINE, iu)
                if polish_radius > best_radius:
                    best_radius = polish_radius
                    best_centers = polish_centers
    
    # Initializations - prioritize diverse, good starting points
    # 1. BCC lattice (known good packing structure)
    if _time_left() > 30.0:
        centers = _build_bcc_init()
        try_init(centers, time_budget=30.0)
    
    # 2. Grids - key configurations
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
    ]
    
    for nx, ny, nz in grid_configs:
        if _time_left() < 28.0:
            break
        centers = _build_grid_init(nx, ny, nz, 0.0, seed=nx*100+ny*10+nz)
        try_init(centers, time_budget=25.0)
    
    # 3. Sobol sequence (better than Halton)
    if _time_left() > 25.0:
        centers = _build_sobol_init()
        try_init(centers, time_budget=22.0)
    
    # 4. Halton sequence
    if _time_left() > 20.0:
        centers = _build_halton_init()
        try_init(centers, time_budget=18.0)
    
    # 5. Random initialization
    if _time_left() > 15.0:
        rng = np.random.default_rng(123)
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        try_init(centers, time_budget=12.0)
    
    # Final polish with remaining time
    if best_centers is not None and _time_left() > 5.0:
        polish_centers, polish_radius = _polish_repulsion(best_centers, _DEADLINE, iu)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
    
    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
