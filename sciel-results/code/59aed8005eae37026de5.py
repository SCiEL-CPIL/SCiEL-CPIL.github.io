<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 13

# Pre-compute all triangle index combinations
def _precompute_triangles(n):
    tris = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                tris.append((i, j, k))
    return np.array(tris)

TRIANGLES = _precompute_triangles(N)

def _min_triangle_area(points):
    """Compute minimum triangle area for N points using precomputed indices."""
    # points: (N, 2)
    # For each triangle (i,j,k): area = 0.5 * |cross(pj-pi, pk-pi)|
    pi = points[TRIANGLES[:, 0]]
    pj = points[TRIANGLES[:, 1]]
    pk = points[TRIANGLES[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min_triangle_area / hull_area."""
    ma = _min_triangle_area(points)
    ha = _hull_area(points)
    if ha < 1e-10:
        return 0.0
    return ma / ha

def _objective(x):
    """Negative score for minimization."""
    points = x.reshape(N, 2)
    return -_score(points)

def _nelder_mead_optimize(x0, maxiter=2000, time_limit=None, start_time=None):
    """Run Nelder-Mead optimization with time guard."""
    if start_time is None:
        start_time = time.time()
    
    options = {
        'maxiter': maxiter,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True,
    }
    
    def callback(xk):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            raise TimeoutError()
    
    try:
        result = minimize(_objective, x0, method='Nelder-Mead', options=options, callback=callback)
        return result.x
    except TimeoutError:
        return x0
    except Exception:
        return x0

def _coordinate_descent_polish(points, n_iter=200, step_init=0.01, time_limit=None, start_time=None):
    """Polish with coordinate descent on each coordinate."""
    if start_time is None:
        start_time = time.time()
    
    x = points.flatten().copy()
    best_score = _score(x.reshape(N, 2))
    step = step_init
    
    for iteration in range(n_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        improved = False
        for i in range(2 * N):
            if time_limit is not None and (time.time() - start_time) > time_limit:
                break
            
            # Try increasing
            x_try = x.copy()
            x_try[i] += step
            s_try = _score(x_try.reshape(N, 2))
            if s_try > best_score:
                x = x_try
                best_score = s_try
                improved = True
            # Try decreasing
            x_try = x.copy()
            x_try[i] -= step
            s_try = _score(x_try.reshape(N, 2))
            if s_try > best_score:
                x = x_try
                best_score = s_try
                improved = True
        
        if not improved:
            step *= 0.7
            if step < 1e-10:
                break
    
    return x.reshape(N, 2)

def _make_initial_config(rng):
    """Generate an initial configuration."""
    # Use a structured pattern: place points on a slightly perturbed grid
    # 4x4 grid minus 3 points, or other patterns
    configs = []
    
    # Pattern 1: 4x4 grid minus 3 corners
    xs = [0, 1, 2, 3]
    ys = [0, 1, 2, 3]
    grid = [(x, y) for x in xs for y in ys]
    # Remove 3 points to get 13
    removed = rng.sample(range(16), 3)
    points = []
    for idx, (x, y) in enumerate(grid):
        if idx not in removed:
            points.append([x / 3.0 + rng.gauss(0, 0.05), y / 3.0 + rng.gauss(0, 0.05)])
    configs.append(np.array(points))
    
    # Pattern 2: Fibonacci-like distribution
    points = []
    for i in range(N):
        angle = 2 * np.pi * i / N + rng.gauss(0, 0.1)
        r = 0.5 + 0.4 * np.sqrt((i + 0.5) / N)
        points.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    configs.append(np.array(points))
    
    # Pattern 3: Random in unit square
    points = np.array([[rng.random(), rng.random()] for _ in range(N)])
    configs.append(points)
    
    # Pattern 4: Structured with some symmetry
    points = []
    # 4 points near corners
    for cx, cy in [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]:
        points.append([cx + rng.gauss(0, 0.03), cy + rng.gauss(0, 0.03)])
    # 4 points near edge midpoints
    for cx, cy in [(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]:
        points.append([cx + rng.gauss(0, 0.03), cy + rng.gauss(0, 0.03)])
    # 5 points in center region
    for i in range(5):
        angle = 2 * np.pi * i / 5
        r = 0.25
        points.append([0.5 + r * np.cos(angle) + rng.gauss(0, 0.02), 
                       0.5 + r * np.sin(angle) + rng.gauss(0, 0.02)])
    configs.append(np.array(points))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.RandomState(42)
    import random
    rng_py = random.Random(42)
    
    best_points = None
    best_score = -np.inf
    
    # Generate initial configurations
    initial_configs = _make_initial_config(rng_py)
    
    # Also add a pure random start
    initial_configs.append(np.array([[rng_py.random(), rng_py.random()] for _ in range(N)]))
    
    # Optimize each initial configuration
    for init_idx, x0 in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Normalize to unit square
        x0 = x0.flatten()
        
        # Run Nelder-Mead
        x_opt = _nelder_mead_optimize(x0, maxiter=500, time_limit=20.0, start_time=start_time)
        
        # Polish with coordinate descent
        points = x_opt.reshape(N, 2)
        points = _coordinate_descent_polish(points, n_iter=100, step_init=0.01, 
                                             time_limit=10.0, start_time=start_time)
        
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # If we have time, do another round with perturbed best
    if best_points is not None and time.time() < deadline - 30:
        for trial in range(5):
            if time.time() > deadline:
                break
            
            # Perturb the best solution
            x_pert = best_points.flatten() + rng.normal(0, 0.05, size=2 * N)
            
            x_opt = _nelder_mead_optimize(x_pert, maxiter=300, time_limit=15.0, start_time=start_time)
            points = x_opt.reshape(N, 2)
            points = _coordinate_descent_polish(points, n_iter=50, step_init=0.005,
                                                 time_limit=5.0, start_time=start_time)
            
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
    
    if best_points is None:
        # Fallback: random points
        best_points = np.array([[rng_py.random(), rng_py.random()] for _ in range(N)])
    
    return best_points
=======
"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13
DEADLINE_MARGIN = 5.0  # safety margin before 150s kill

# Pre-compute all triangle index combinations
def _precompute_triangles(n):
    tris = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                tris.append((i, j, k))
    return np.array(tris)

TRIANGLES = _precompute_triangles(N)
N_TRIS = len(TRIANGLES)

def _min_triangle_area(points):
    """Compute minimum triangle area for N points using precomputed indices."""
    pi = points[TRIANGLES[:, 0]]
    pj = points[TRIANGLES[:, 1]]
    pk = points[TRIANGLES[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min_triangle_area / hull_area."""
    ma = _min_triangle_area(points)
    ha = _hull_area(points)
    if ha < 1e-10:
        return 0.0
    return ma / ha

def _affine_transform(points, a, b, c, d, tx, ty):
    """Apply affine transform: [a b; c d] * [x;y] + [tx;ty]"""
    x = points[:, 0]
    y = points[:, 1]
    nx = a * x + b * y + tx
    ny = c * x + d * y + ty
    return np.column_stack([nx, ny])

def _optimize_affine(base_points):
    """Optimize affine transform of a fixed point pattern."""
    x0 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    
    def obj(params):
        a, b, c, d, tx, ty = params
        pts = _affine_transform(base_points, a, b, c, d, tx, ty)
        return -_score(pts)
    
    try:
        result = minimize(obj, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        a, b, c, d, tx, ty = result.x
        return _affine_transform(base_points, a, b, c, d, tx, ty)
    except Exception:
        return base_points

def _simulated_annealing(x0, start_time, deadline, rng, n_iter=5000, T_start=0.1, T_end=0.001):
    """Simulated Annealing with vectorized score evaluation."""
    x = x0.copy()
    best_x = x.copy()
    best_score = _score(x.reshape(N, 2))
    
    for i in range(n_iter):
        if time.time() > deadline - DEADLINE_MARGIN:
            break
        
        # Cooling schedule
        T = T_start * (T_end / T_start) ** (i / n_iter)
        
        # Generate perturbation
        scale = max(T, 0.001)
        x_try = x + rng.normal(0, scale, size=2 * N)
        
        s_try = _score(x_try.reshape(N, 2))
        s_cur = _score(x.reshape(N, 2))
        
        if s_try > s_cur:
            x = x_try
            if s_try > best_score:
                best_score = s_try
                best_x = x.copy()
        else:
            # Accept with probability exp((s_try - s_cur) / T)
            delta = s_try - s_cur
            if delta < 0 and T > 1e-15:
                if rng.random() < np.exp(delta / T):
                    x = x_try
    
    return best_x

def _coordinate_descent_polish(points, n_iter=100, step_init=0.01, start_time=None, deadline=None):
    """Polish with coordinate descent."""
    if start_time is None:
        start_time = time.time()
    
    x = points.flatten().copy()
    best_score = _score(x.reshape(N, 2))
    step = step_init
    
    for iteration in range(n_iter):
        if time.time() > deadline - DEADLINE_MARGIN:
            break
        
        improved = False
        for i in range(2 * N):
            if time.time() > deadline - DEADLINE_MARGIN:
                break
            
            x_try = x.copy()
            x_try[i] += step
            s_try = _score(x_try.reshape(N, 2))
            if s_try > best_score:
                x = x_try
                best_score = s_try
                improved = True
            
            x_try = x.copy()
            x_try[i] -= step
            s_try = _score(x_try.reshape(N, 2))
            if s_try > best_score:
                x = x_try
                best_score = s_try
                improved = True
        
        if not improved:
            step *= 0.7
            if step < 1e-10:
                break
    
    return x.reshape(N, 2)

def _make_grid_pattern(n_removed, rng):
    """Create a 4x4 grid pattern with n_removed points removed."""
    xs = np.linspace(0, 1, 4)
    ys = np.linspace(0, 1, 4)
    grid = np.array([[x, y] for y in ys for x in xs])
    idx = rng.choice(16, size=n_removed, replace=False)
    mask = np.ones(16, dtype=bool)
    mask[idx] = False
    return grid[mask]

def _make_circular_pattern(rng):
    """Points on a circle with some interior points."""
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    pts = np.column_stack([0.5 + 0.45 * np.cos(angles), 0.5 + 0.45 * np.sin(angles)])
    # Add 5 interior points
    inner_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + 0.3
    inner_pts = np.column_stack([0.5 + 0.2 * np.cos(inner_angles), 0.5 + 0.2 * np.sin(inner_angles)])
    return np.vstack([pts, inner_pts])

def _make_hex_pattern(rng):
    """Hexagonal-like pattern."""
    pts = []
    # 3 rings
    for r, n in [(0.0, 1), (0.3, 6), (0.55, 6)]:
        if r == 0:
            pts.append([0.5, 0.5])
        else:
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            for a in angles:
                pts.append([0.5 + r * np.cos(a), 0.5 + r * np.sin(a)])
    return np.array(pts[:13])

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = -np.inf
    
    def update_best(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Strategy 1: Grid patterns with affine optimization
    grid_patterns = []
    for n_rem in [3]:
        for trial in range(4):
            pat = _make_grid_pattern(n_rem, rng)
            grid_patterns.append(pat)
    
    # Strategy 2: Circular pattern
    grid_patterns.append(_make_circular_pattern(rng))
    
    # Strategy 3: Hex pattern
    grid_patterns.append(_make_hex_pattern(rng))
    
    # Optimize each pattern with affine transform
    for pat in grid_patterns:
        if time.time() > deadline - 20:
            break
        opt_pts = _optimize_affine(pat)
        update_best(opt_pts)
    
    # Strategy 4: Simulated Annealing from best
    if best_points is not None and time.time() < deadline - 30:
        sa_start = best_points.flatten().copy()
        sa_result = _simulated_annealing(sa_start, start_time, deadline, rng, n_iter=8000, T_start=0.15, T_end=0.002)
        update_best(sa_result)
    
    # Strategy 5: Coordinate descent polish
    if best_points is not None and time.time() < deadline - 10:
        polished = _coordinate_descent_polish(best_points, n_iter=80, step_init=0.01, 
                                               start_time=start_time, deadline=deadline)
        update_best(polished)
    
    # Strategy 6: More SA restarts if time permits
    if best_points is not None and time.time() < deadline - 15:
        for trial in range(3):
            if time.time() > deadline - DEADLINE_MARGIN:
                break
            perturbed = best_points.flatten() + rng.normal(0, 0.03, size=2 * N)
            sa_result = _simulated_annealing(perturbed, start_time, deadline, rng, n_iter=4000, T_start=0.08, T_end=0.001)
            update_best(sa_result)
    
    # Final polish
    if best_points is not None and time.time() < deadline - 3:
        polished = _coordinate_descent_polish(best_points, n_iter=50, step_init=0.005,
                                               start_time=start_time, deadline=deadline)
        update_best(polished)
    
    if best_points is None:
        best_points = np.random.RandomState(0).random((N, 2))
    
    return best_points
>>>>>>> REPLACE
<<<<<<< SEARCH
# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
