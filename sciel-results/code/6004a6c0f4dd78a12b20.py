"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

DEADLINE = 145.0  # safety margin before 150s kill


def _score(x):
    """Compute the radius for a flat array of 2N coords (x0,y0,x1,y1,...)."""
    pts = x.reshape(N, 2)
    # distance to walls
    wall = 0.5 - np.abs(pts)
    min_wall = np.min(wall)
    # pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(min_wall, min_pair)


def _neg_score(x):
    return -_score(x)


def _optimize_from(pts0, max_time=30.0):
    """Run L-BFGS-B from pts0, returning best pts found."""
    x0 = pts0.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    best_x = x0.copy()
    best_f = _neg_score(x0)
    
    t_start = time.time()
    try:
        res = minimize(_neg_score, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'maxfun': 50000,
                                'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    # Check deadline
    if time.time() - t_start > max_time:
        return best_x.reshape(N, 2), best_f
    
    return best_x.reshape(N, 2), best_f


def _hex_grid_init(n):
    """Generate a hexagonal packing initial configuration."""
    pts = []
    # Hexagonal lattice: rows offset
    # For n=200, we want roughly 14-15 rows
    row_count = int(np.ceil(n / 14))  # about 14-15 per row
    # Actually let's compute more carefully
    # Hex packing: each row has about sqrt(3)/2 spacing vertically
    # Let's just generate a hex grid and trim
    spacing = 1.0 / 15  # approximate
    y = -0.5
    while len(pts) < n and y < 0.5:
        x = -0.5
        row_offset = (len([p for p in pts]) % 2) * spacing * 0.5
        while x < 0.5 and len(pts) < n:
            px = x + row_offset
            if -0.5 <= px <= 0.5:
                pts.append([px, y])
            x += spacing
        y += spacing * np.sqrt(3) / 2
    pts = np.array(pts[:n])
    return pts


def _random_init(n, seed=None):
    """Random initial configuration inside the square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 2))


def _grid_init(n):
    """Square grid initial configuration."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def construct():
    t0 = time.time()
    
    # Generate multiple diverse initial configurations
    inits = []
    inits.append(_grid_init(N))
    inits.append(_hex_grid_init(N))
    for s in range(5):
        inits.append(_random_init(N, seed=42 + s))
    
    best_pts = inits[0].copy()
    best_r = _score(best_pts)
    
    # Optimize from each initial configuration
    for i, init_pts in enumerate(inits):
        if time.time() - t0 > DEADLINE - 5:
            break
        
        # Time budget per optimization
        remaining = DEADLINE - (time.time() - t0)
        max_time = max(5.0, remaining / (len(inits) - i))
        
        pts, f = _optimize_from(init_pts, max_time=min(max_time, 25.0))
        r = -f
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # If we have time, do a second round of refinement on the best
    if time.time() - t0 < DEADLINE - 10:
        # Perturb the best solution slightly and re-optimize
        rng = np.random.RandomState(123)
        for attempt in range(3):
            if time.time() - t0 > DEADLINE - 5:
                break
            perturbed = best_pts + rng.normal(0, 0.01, size=(N, 2))
            # Clamp to valid range
            perturbed = np.clip(perturbed, -0.49, 0.49)
            pts, f = _optimize_from(perturbed, max_time=15.0)
            r = -f
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    # Final check: ensure all points are strictly inside
    best_pts = np.clip(best_pts, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_pts
# EVOLVE-BLOCK-END
