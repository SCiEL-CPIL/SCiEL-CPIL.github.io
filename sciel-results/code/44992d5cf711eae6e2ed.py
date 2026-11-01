import numpy as np
import time
from scipy.optimize import minimize
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
    Uses cKDTree for efficiency.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances using cKDTree
    # We only need to check pairs that are potentially close.
    # If min radius is r, we care about pairs with dist < 2r.
    # For finding the *current* min radius, we can just find the global min pair dist.
    tree = cKDTree(centers)
    # Query pairs within a reasonable range. 
    # The max possible radius is 0.5. So max pair dist of interest is 1.0.
    # But usually, the min dist is small. 
    # Let's find the k-nearest neighbors to ensure we catch the closest one.
    # k=1 is enough to find the closest neighbor for each point, but we need min over all i<j.
    # query_pairs with r=0.5 is a safe bet for finding close pairs in a unit cube with 100 points.
    # If the packing is tight, min dist is ~0.2, so 2*r ~ 0.4.
    # If the packing is loose, min dist is larger, but we will still find the min if r is large enough.
    # To be safe, use r=0.6.
    pairs = tree.query_pairs(r=0.6)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        # Calculate distances for these pairs
        i_idx, j_idx = zip(*pairs)
        i_idx = np.array(i_idx)
        j_idx = np.array(j_idx)
        diffs = centers[i_idx] - centers[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        min_pair_dist = np.min(dists)
        
    return min(min_wall_dist, min_pair_dist / 2.0)

def _get_close_pairs(centers, r_search):
    """
    Return indices of pairs with distance < r_search.
    """
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=r_search)
    if not pairs:
        return np.array([]), np.array([]), np.array([])
    i_idx, j_idx = zip(*pairs)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    diffs = centers[i_idx] - centers[j_idx]
    dists = np.linalg.norm(diffs, axis=1)
    return i_idx, j_idx, dists

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    Only considers close pairs to save time.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    # Use a search radius that is generous enough to include the critical pairs.
    # If current r is small, we use a larger search radius to be safe.
    # Estimate current r roughly from wall dists or just use a fixed 0.5.
    # For speed, we can use a fixed search radius of 0.5.
    i_idx, j_idx, dists = _get_close_pairs(centers, 0.5)
    if len(i_idx) > 0:
        pair_vals = dists / 2.0
    else:
        pair_vals = np.array([0.5]) # Large value, won't affect min
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
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
    """
    nx, ny, nz = 5, 5, 2
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                points.append([cx, cy, cz])
                points.append([cx + 0.5/nx, cy + 0.5/ny, cz + 0.5/nz])
    
    centers = np.array(points[:N])
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
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
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _polish_repulsion(centers, deadline, iu):
    """
    Local search using repulsion forces with KD-tree acceleration.
    This is the primary optimizer. It directly maximizes the minimum distance.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    step_size = 0.005
    decay = 0.98
    min_step = 1e-8
    margin = 1.01
    
    no_improve_count = 0
    max_no_improve = 50
    
    while _time_left() > 0.5 and step_size > min_step and no_improve_count < max_no_improve:
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
        search_radius = 2 * max(r_current * margin, 0.01)
        
        neighbor_pairs = tree.query_pairs(r=search_radius)
        
        for i, j in neighbor_pairs:
            if i == j:
                continue
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9 and dist < search_radius:
                strength = (search_radius - dist) / search_radius
                unit_diff = diff / dist
                forces[i] += strength * unit_diff
                forces[j] -= strength * unit_diff
        
        # Normalize forces
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        centers += step_size * forces
        centers = np.clip(centers, -0.4999, 0.4999)
        
        step_size *= decay
        
    return best_centers, best_r

def _optimize_from_initial(centers, deadline, iu, time_budget=20.0):
    """
    Run optimization from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    start_t = time.time()
    remaining = time_budget
    
    # Phase 1: Soft-min optimization with adaptive beta
    beta_start = 50.0
    beta_end = 5000.0
    n_beta_steps = 4
    
    if remaining > 5.0:
        for step in range(n_beta_steps):
            if _time_left() < 2.0 or (time.time() - start_t) > remaining - 2.0:
                break
                
            beta = beta_start * (beta_end / beta_start) ** (step / (n_beta_steps - 1))
            
            x0 = best_centers.flatten()
            
            try:
                max_iter = 100 if step < 2 else 50
                res = minimize(
                    _soft_min_objective,
                    x0,
                    args=(beta, iu),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (N * 3),
                    options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 10}
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
    if _time_left() > 1.0 and (time.time() - start_t) < remaining:
        polish_deadline = min(deadline, start_t + remaining)
        polish_centers, polish_radius = _polish_repulsion(best_centers, polish_deadline, iu)
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
            if _time_left() > 3.0:
                polish_centers, polish_radius = _polish_repulsion(centers, _DEADLINE, iu)
                if polish_radius > best_radius:
                    best_radius = polish_radius
                    best_centers = polish_centers
    
    # Initializations
    if _time_left() > 20.0:
        centers = _build_bcc_init()
        try_init(centers, time_budget=20.0)
    
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
        (5, 5, 5), # 125 points, truncate to 100
    ]
    
    for nx, ny, nz in grid_configs:
        if _time_left() < 20.0:
            break
        for noise in [0.0, 0.02]:
            if _time_left() < 15.0:
                break
            centers = _build_grid_init(nx, ny, nz, noise, seed=nx*100+ny*10+nz)
            try_init(centers, time_budget=15.0)
    
    if _time_left() > 15.0:
        centers = _build_halton_init()
        try_init(centers, time_budget=15.0)
    
    rng = np.random.default_rng(123)
    for seed in [123, 456, 789, 1011]:
        if _time_left() < 15.0:
            break
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        try_init(centers, time_budget=12.0)
    
    if best_centers is not None and _time_left() > 5.0:
        polish_centers, polish_radius = _polish_repulsion(best_centers, _DEADLINE, iu)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers
    
    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
