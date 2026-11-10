import numpy as np
import time
import math
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START

def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, positive sides, non-overlapping. Vectorized."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside unit square
    if np.any(centers - sides[:, None] / 2.0 < -tol):
        return False
    if np.any(centers + sides[:, None] / 2.0 > 1.0 + tol):
        return False
    # Non-overlap
    dx = np.abs(centers[:, 0][:, None] - centers[:, 0][None, :])
    dy = np.abs(centers[:, 1][:, None] - centers[:, 1][None, :])
    t = (sides[:, None] + sides[None, :]) / 2.0 - tol
    iu = np.triu_indices(n, k=1)
    if np.any((dx[iu] < t[iu]) & (dy[iu] < t[iu])):
        return False
    return True

def _clamp_to_unit(centers, sides):
    """Clamp centers so squares stay inside [0,1]^2."""
    for i in range(len(sides)):
        half = sides[i] / 2.0
        centers[i, 0] = max(half, min(1.0 - half, centers[i, 0]))
        centers[i, 1] = max(half, min(1.0 - half, centers[i, 1]))
    return centers

def _max_side_at(cx, cy, centers, sides, i, tol=1e-7):
    """
    Compute the maximum side length for square i centered at (cx,cy)
    given the positions and sizes of all other squares.
    """
    max_s = min(2.0 * cx, 2.0 * (1.0 - cx), 2.0 * cy, 2.0 * (1.0 - cy))
    if max_s < 0:
        max_s = 0.0

    mask = np.arange(len(sides)) != i
    dx = np.abs(cx - centers[mask, 0])
    dy = np.abs(cy - centers[mask, 1])
    sj = sides[mask]
    bounds = np.maximum(2.0 * (dx + tol) - sj, 2.0 * (dy + tol) - sj)
    if len(bounds) > 0:
        max_s = min(max_s, float(np.min(bounds)))

    if max_s < 0:
        max_s = 0.0
    return max_s

def _grow_all_squares(centers, sides, tol=1e-7, max_rounds=200):
    """
    Iteratively grow all squares to their maximum given current positions.
    """
    n = len(sides)
    for _ in range(max_rounds):
        improved = False
        new_sides = sides.copy()
        for i in range(n):
            new_sides[i] = _max_side_at(centers[i, 0], centers[i, 1], centers, sides, i, tol)
        if np.any(new_sides > sides + 1e-10):
            sides = new_sides
            improved = True
        if not improved:
            break
    return sides

def _voronoi_side_length(cx, cy, centers, sides, i, tol=1e-7):
    """
    Compute the side length limited by Voronoi cell (distance to nearest neighbor/boundary).
    """
    # Distance to boundaries
    d_bound = min(cx, 1.0 - cx, cy, 1.0 - cy)
    s_bound = 2.0 * d_bound

    # Distance to other squares (Chebyshev-like for axis-aligned)
    mask = np.arange(len(sides)) != i
    if np.any(mask):
        dx = np.abs(cx - centers[mask, 0])
        dy = np.abs(cy - centers[mask, 1])
        sj = sides[mask]
        # The gap in x is dx - sj/2, in y is dy - sj/2
        # The square can extend until it touches the neighbor
        # s_i/2 + s_j/2 <= dx OR s_i/2 + s_j/2 <= dy
        # s_i <= 2*dx - s_j OR s_i <= 2*dy - s_j
        # The constraint is s_i <= min_j max(2*dx_j - s_j, 2*dy_j - s_j)
        bounds = np.maximum(2.0 * dx - sj, 2.0 * dy - sj)
        s_neighbor = float(np.min(bounds))
    else:
        s_neighbor = float('inf')

    return min(s_bound, s_neighbor)

def _voronoi_objective(x, tol=1e-7):
    """
    Objective for optimization: maximize sum of Voronoi-limited side lengths.
    x: [cx1, cy1, cx2, cy2, ..., cx10, cy10]
    Returns: -sum(s_i) where s_i is Voronoi-limited.
    """
    centers = x.reshape(10, 2)
    sides = np.zeros(10)
    for i in range(10):
        sides[i] = _voronoi_side_length(centers[i, 0], centers[i, 1], centers, sides, i, tol)
    return -np.sum(sides)

def _local_refine(centers, sides, tol=1e-7, n_iter=500, rng=None, deadline=None):
    """
    Local refinement: perturb positions, recompute Voronoi sides, grow.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = float(np.sum(best_s))
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb a random subset
        k = int(rng.integers(1, 4))
        idx = rng.choice(n, size=k, replace=False)
        trial_c = best_c.copy()
        for i in idx:
            step = rng.uniform(0.005, 0.05)
            trial_c[i, 0] += rng.uniform(-step, step)
            trial_c[i, 1] += rng.uniform(-step, step)
        
        trial_c = _clamp_to_unit(trial_c, best_s)
        
        # Compute Voronoi sides
        trial_s = np.zeros(10)
        for i in range(10):
            trial_s[i] = _voronoi_side_length(trial_c[i, 0], trial_c[i, 1], trial_c, trial_s, i, tol)
        
        # Grow
        trial_s = _grow_all_squares(trial_c, trial_s, tol=tol, max_rounds=100)
        
        if _is_valid(trial_c, trial_s, tol=tol):
            obj = float(np.sum(trial_s))
            if obj > best_obj + 1e-10:
                best_obj = obj
                best_c = trial_c.copy()
                best_s = trial_s.copy()
    
    return best_c, best_s

def _init_grid_2x5():
    """2x5 grid."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.full(10, 0.2, dtype=np.float64)
    idx = 0
    for row in range(2):
        for col in range(5):
            centers[idx, 0] = (col + 0.5) * 0.2
            centers[idx, 1] = (row + 0.5) * 0.5
            idx += 1
    return centers, sides

def _init_grid_5x2():
    """5x2 grid."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.full(10, 0.2, dtype=np.float64)
    idx = 0
    for row in range(5):
        for col in range(2):
            centers[idx, 0] = (col + 0.5) * 0.5
            centers[idx, 1] = (row + 0.5) * 0.2
            idx += 1
    return centers, sides

def _init_corner_cross():
    """4 corners + cross."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    s_large = 0.35
    centers[0] = [s_large/2, s_large/2]
    centers[1] = [1 - s_large/2, s_large/2]
    centers[2] = [s_large/2, 1 - s_large/2]
    centers[3] = [1 - s_large/2, 1 - s_large/2]
    sides[0:4] = s_large
    
    s_mid = 0.25
    centers[4] = [0.5, s_large + s_mid/2]
    centers[5] = [0.5, 1 - s_large - s_mid/2]
    centers[6] = [s_large + s_mid/2, 0.5]
    centers[7] = [1 - s_large - s_mid/2, 0.5]
    sides[4:8] = s_mid
    
    s_small = 0.20
    centers[8] = [0.5, 0.5]
    centers[9] = [0.5, 0.5 + s_small]
    sides[8:10] = s_small
    
    return centers, sides

def _init_random(rng):
    """Random initialization."""
    centers = rng.uniform(0.1, 0.9, size=(10, 2))
    sides = rng.uniform(0.1, 0.3, size=10)
    return centers, sides

def _differential_evolution_optimize(deadline):
    """
    Use differential evolution to optimize Voronoi-limited side lengths.
    """
    # Bounds for centers: [0, 1]
    bounds = [(0.0, 1.0)] * 20
    
    # Use a subset of evaluations to fit within time
    maxiter = 100
    popsize = 20
    
    def obj(x):
        if time.time() > deadline:
            return 1e10
        return _voronoi_objective(x)
    
    try:
        res = differential_evolution(
            obj,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
            seed=42,
            polish=False,
            init='sobol'
        )
        return res.x
    except Exception:
        return None

def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 non-overlapping axis-aligned squares in the unit square.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    inits = [
        _init_grid_2x5(),
        _init_grid_5x2(),
        _init_corner_cross(),
    ]
    
    rng_global = np.random.default_rng(12345)
    for _ in range(5):
        inits.append(_init_random(rng_global))
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    # Phase 1: Grow all initial layouts
    for centers, sides in inits:
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        sides = _grow_all_squares(centers, sides, tol=1e-7)
        
        if _is_valid(centers, sides, tol=1e-7):
            obj = float(np.sum(sides))
            if obj > best_obj:
                best_obj = obj
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # Phase 2: Differential evolution on Voronoi objective
    if time.time() < deadline - 120:
        x_opt = _differential_evolution_optimize(deadline - 120)
        if x_opt is not None:
            c_opt = x_opt.reshape(10, 2)
            c_opt = _clamp_to_unit(c_opt, np.zeros(10))
            s_opt = np.zeros(10)
            for i in range(10):
                s_opt[i] = _voronoi_side_length(c_opt[i, 0], c_opt[i, 1], c_opt, s_opt, i, tol=1e-7)
            s_opt = _grow_all_squares(c_opt, s_opt, tol=1e-7)
            
            if _is_valid(c_opt, s_opt, tol=1e-7):
                obj = float(np.sum(s_opt))
                if obj > best_obj:
                    best_obj = obj
                    best_centers = c_opt.copy()
                    best_sides = s_opt.copy()
    
    # Phase 3: Local refinement from best
    if best_centers is not None and time.time() < deadline - 30:
        rng = np.random.default_rng(777)
        opt_c, opt_s = _local_refine(best_centers, best_sides, tol=1e-7, n_iter=500, rng=rng, deadline=deadline - 30)
        if _is_valid(opt_c, opt_s, tol=1e-7):
            obj = float(np.sum(opt_s))
            if obj > best_obj:
                best_obj = obj
                best_centers = opt_c.copy()
                best_sides = opt_s.copy()
    
    # Phase 4: Final high-precision polish
    if best_centers is not None and time.time() < deadline - 10:
        polished_c = best_centers.copy()
        polished_s = best_sides.copy()
        polished_s = _grow_all_squares(polished_c, polished_s, tol=1e-9, max_rounds=500)
        if _is_valid(polished_c, polished_s, tol=1e-7):
            obj = float(np.sum(polished_s))
            if obj > best_obj:
                best_centers = polished_c.copy()
                best_sides = polished_s.copy()
    
    # Fallback
    if best_centers is None:
        centers, sides = _init_grid_2x5()
        centers = _clamp_to_unit(centers, sides)
        sides = _grow_all_squares(centers, sides, tol=1e-7)
        best_centers = centers
        best_sides = sides
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
