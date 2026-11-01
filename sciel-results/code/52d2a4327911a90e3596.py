import numpy as np
import time
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
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _fast_compute_radius(centers, tree=None):
    """
    Fast radius computation using cKDTree for pairwise distances.
    """
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    if tree is None:
        tree = cKDTree(centers)
    
    # Query each point's nearest neighbor
    dists, idxs = tree.query(centers, k=2)
    min_pair_dist = np.min(dists[:, 1])
    
    return min(min_wall_dist, min_pair_dist / 2.0), tree

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

def _build_bcc_init():
    """
    Build a BCC (body-centered cubic) inspired initialization.
    """
    # Use a 5x5x4 grid with some jitter
    centers = _build_grid_init(5, 5, 4, noise_scale=0.02, seed=999)
    return centers

def _metropolis_anneal(centers, deadline):
    """
    Metropolis-Hastings annealing to maximize minimum clearance.
    Uses cKDTree for fast distance queries.
    """
    centers = centers.copy()
    current_r, tree = _fast_compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Initial temperature
    T = 0.01
    T_min = 1e-5
    cooling_rate = 0.9995
    
    rng = np.random.default_rng(12345)
    
    while time.time() < deadline - 2.0:
        if T < T_min:
            T = T_min
        
        # Pick a random sphere
        idx = rng.integers(N)
        
        # Random move with step size proportional to T
        step_size = T * 0.5
        delta = rng.normal(0, step_size, 3)
        
        new_center = centers[idx] + delta
        new_center = np.clip(new_center, -0.5, 0.5)
        
        old_center = centers[idx]
        centers[idx] = new_center
        
        # Rebuild tree and compute new radius
        tree = cKDTree(centers)
        new_r, tree = _fast_compute_radius(centers, tree)
        
        # Metropolis acceptance criterion
        delta_r = new_r - current_r
        if delta_r > 0 or rng.random() < np.exp(delta_r / T):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            centers[idx] = old_center
            # Rebuild tree for the restored state
            tree = cKDTree(centers)
            current_r, tree = _fast_compute_radius(centers, tree)
        
        T *= cooling_rate
    
    return best_centers, best_r

def _local_optimize(centers, deadline):
    """
    Structured local optimization: resolve closest-pair and wall constraints.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    while time.time() < deadline - 1.0:
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
            
        # Find limiting pair using cKDTree
        tree = cKDTree(centers)
        dists, idxs = tree.query(centers, k=2)
        min_pair_dist = np.min(dists[:, 1])
        min_idx = np.argmin(dists[:, 1])
        i = min_idx
        j = idxs[min_idx, 1]
        
        # Check wall constraint
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        wall_idx = np.argmin(wall_dists)
        i_wall = wall_idx // 3
        k_wall = wall_idx % 3
        
        if min_pair_dist / 2.0 < min_wall:
            # Pair is limiting: move i and j apart
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-9:
                dir_i = diff / norm
                dir_j = -diff / norm
            else:
                dir_i = np.random.normal(0, 1, 3)
                dir_i /= np.linalg.norm(dir_i)
                dir_j = -dir_i
                
            step = 0.005
            centers[i] += step * dir_i
            centers[j] += step * dir_j
        else:
            # Wall is limiting: move point away from wall
            dir = -np.sign(centers[i_wall, k_wall])
            if dir == 0:
                dir = 1.0
            step = 0.005
            centers[i_wall, k_wall] += step * dir
            
        centers = np.clip(centers, -0.5, 0.5)
        
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = _get_deadline()
    
    best_centers = None
    best_radius = 0.0
    
    # Initializations
    initializations = []
    
    # Grids
    grid_configs = [
        (4, 5, 5),
        (5, 4, 5),
        (5, 5, 4),
        (10, 5, 2),
        (5, 10, 2),
        (5, 2, 10),
    ]
    
    for nx, ny, nz in grid_configs:
        for noise in [0.0, 0.01, 0.02]:
            centers = _build_grid_init(nx, ny, nz, noise, seed=nx*100+ny*10+nz)
            initializations.append(centers)
            
    # Halton
    initializations.append(_build_halton_init())
    
    # BCC-inspired
    initializations.append(_build_bcc_init())
    
    # Random
    rng = np.random.default_rng(123)
    for _ in range(5):
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)
        
    # Optimize each initialization
    n_inits = len(initializations)
    time_per_init = (_get_deadline() - time.time() - 5.0) / n_inits
    
    for centers in initializations:
        if time.time() > deadline - 5.0:
            break
            
        # Phase 1: Local optimization (resolve constraints)
        local_deadline = min(time.time() + time_per_init * 0.4, deadline - 5.0)
        opt_centers, opt_radius = _local_optimize(centers, local_deadline)
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
            
        # Phase 2: Metropolis annealing (explore with acceptance)
        if time.time() < deadline - 5.0:
            anneal_deadline = min(time.time() + time_per_init * 0.6, deadline - 5.0)
            anneal_centers, anneal_radius = _metropolis_anneal(opt_centers, anneal_deadline)
            
            if anneal_radius > best_radius:
                best_radius = anneal_radius
                best_centers = anneal_centers

    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
