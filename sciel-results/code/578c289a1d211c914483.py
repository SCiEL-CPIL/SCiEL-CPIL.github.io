"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


def _score(pts):
    """Compute the packing score (minimum radius) for given centres."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds: centres must be in [-0.5, 0.5]^2
    if np.any(pts < -0.5 - 1e-15) or np.any(pts > 0.5 + 1e-15):
        return 0.0
    # Wall distances
    wall_dists = np.minimum(
        np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
        np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
    )
    wall_min = np.min(wall_dists)
    # Pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    pair_min = np.min(d)
    return min(wall_min, pair_min / 2.0)


def _hex_init(n):
    """Generate a hexagonal close-packed initial arrangement of n points."""
    pts = []
    row = 0
    while len(pts) < n:
        y = row * np.sqrt(3) / 2
        x_offset = (row % 2) * 0.5
        col = 0
        while len(pts) < n:
            x = col + x_offset
            pts.append([x, y])
            col += 1
        row += 1
    pts = np.array(pts[:n])
    
    # Center and scale to fit in [-0.5, 0.5]^2
    pts -= pts.mean(axis=0)
    max_extent = np.max(np.abs(pts))
    if max_extent > 0.5:
        pts *= (0.5 / max_extent)
    return pts


def _random_init(n, seed=42):
    """Generate random initial arrangement."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(-0.5, 0.5, size=(n, 2))
    return pts


def _local_optimize(pts, time_budget=100.0, seed=0):
    """Local optimization via random perturbation search."""
    rng = np.random.RandomState(seed)
    current_pts = pts.copy()
    current_score = _score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    start_time = time.time()
    deadline = start_time + time_budget
    temp = 0.01
    base_temp = 0.01
    
    while time.time() < deadline:
        # Generate perturbation
        delta = rng.normal(0, temp, size=current_pts.shape)
        new_pts = current_pts + delta
        
        # Clamp to bounds
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_score = _score(new_pts)
        
        if new_score >= current_score or rng.random() < np.exp((new_score - current_score) / max(temp, 1e-10)):
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        
        # Cool down slowly
        elapsed_frac = (time.time() - start_time) / time_budget
        temp = base_temp * (1 - 0.5 * elapsed_frac)
    
    return best_pts, best_score


def construct():
    """Construct 200 equal circles in unit square, maximizing common radius."""
    start_time = time.time()
    total_budget = 110.0  # Safety margin before 120s limit
    
    # Try multiple initializations
    inits = []
    
    # Hexagonal grid
    hex_pts = _hex_init(N)
    inits.append(('hex', hex_pts))
    
    # Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(('grid', grid[:N]))
    
    # Random
    rand_pts = _random_init(N, seed=123)
    inits.append(('random', rand_pts))
    
    # Allocate time to each initialization
    per_init_budget = (total_budget - 5.0) / len(inits)  # 5s overhead
    
    best_pts = None
    best_score = 0.0
    
    for idx, (name, init_pts) in enumerate(inits):
        remaining = total_budget - (time.time() - start_time)
        if remaining < 2.0:
            break
        
        alloc = min(per_init_budget, remaining - 1.0)
        if alloc < 1.0:
            break
        
        # Local optimize
        opt_pts, opt_score = _local_optimize(init_pts, time_budget=alloc, seed=idx + 1)
        
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()
    
    if best_pts is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]
    
    return best_pts
# EVOLVE-BLOCK-END
