"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area.
Structural change: use log-sum-exp smoothing with adaptive beta, L-BFGS-B local
optimization, and a population-based search with diversity preservation."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 13
TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles. Returns array of shape (C(n,3),)."""
    n = len(pts)
    areas = []
    for i in range(n - 2):
        pi = pts[i]
        rest = pts[i + 1:]
        m = len(rest)
        v1 = rest[:, np.newaxis, :] - pi
        v2 = rest[np.newaxis, :, :] - pi
        cross = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
        idx = np.triu_indices(m, k=1)
        areas.append(np.abs(cross[idx]))
    areas = np.concatenate(areas)
    return 0.5 * areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        min_area = np.min(_all_triangle_areas(pts))
        return min_area / hull_area
    except Exception:
        return 0.0

def _smooth_min_score(pts, beta=100.0):
    """
    Smooth approximation of min(areas) using log-sum-exp.
    soft_min = -(1/beta) * log(sum(exp(-beta * a_i)))
    This is a smooth, differentiable surrogate for the minimum.
    """
    try:
        areas = _all_triangle_areas(pts)
        if len(areas) == 0:
            return 0.0
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        # Use log-sum-exp trick for numerical stability
        max_a = np.max(areas)
        shifted = areas - max_a
        log_sum_exp = np.log(np.sum(np.exp(-beta * shifted))) - beta * max_a
        soft_min = -log_sum_exp / beta
        return soft_min / hull_area
    except Exception:
        return 0.0

def _neg_smooth_score(x, beta=100.0):
    """Negative smooth score for minimization."""
    pts = x.reshape(N, 2)
    return -_smooth_min_score(pts, beta)

def _neg_score(x):
    """Negative exact score for minimization."""
    pts = x.reshape(N, 2)
    return -_score(pts)

def _lbfgs_polish(pts, time_budget=10.0, start_time=None, beta=100.0):
    """L-BFGS-B optimization on the smooth objective."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    
    x0 = pts.flatten().copy()
    
    try:
        result = minimize(
            _neg_smooth_score,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50
            }
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return pts, _score(pts)
        
        return result.x.reshape(N, 2), -result.fun
    except Exception:
        return pts, _score(pts)

def _nelder_mead_polish(pts, time_budget=10.0, start_time=None):
    """Nelder-Mead optimization on the exact objective."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    
    x0 = pts.flatten().copy()
    
    try:
        result = minimize(
            _neg_score,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 1500,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return pts, _score(pts)
        
        return result.x.reshape(N, 2), -result.fun
    except Exception:
        return pts, _score(pts)

def _halton_sequence(n, base):
    """Generate n Halton sequence values in [0,1)."""
    result = []
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        i_temp = i
        while i_temp > 0:
            f /= base
            x += f * (i_temp % base)
            i_temp //= base
        result.append(x)
    return np.array(result)

def _generate_initializations():
    """Generate multiple diverse initial configurations."""
    inits = []
    
    # 1. Regular 13-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts_circle)
    
    # 2. Halton sequence in unit square
    x = _halton_sequence(N, 2)
    y = _halton_sequence(N, 3)
    pts_halton = np.column_stack([x, y])
    inits.append(pts_halton)
    
    # 3. Halton on circle (angular Halton)
    angles_halton = 2 * np.pi * _halton_sequence(N, 2)
    radii = 0.5 + 0.5 * _halton_sequence(N, 3)
    pts_halton_circle = np.column_stack([radii * np.cos(angles_halton), radii * np.sin(angles_halton)])
    inits.append(pts_halton_circle)
    
    # 4. Grid-like: mix of boundary and interior
    angles_boundary = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    boundary_pts = np.column_stack([np.cos(angles_boundary), np.sin(angles_boundary)])
    interior_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    interior_radii = [0.3, 0.5, 0.3, 0.5]
    interior_pts = np.column_stack([
        [r * np.cos(a) for r, a in zip(interior_radii, interior_angles)],
        [r * np.sin(a) for r, a in zip(interior_radii, interior_angles)]
    ])
    pts_mixed = np.vstack([boundary_pts, interior_pts])
    inits.append(pts_mixed)
    
    # 5. Another mixed: 10 boundary, 3 interior
    angles_b = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    b_pts = np.column_stack([np.cos(angles_b), np.sin(angles_b)])
    i_angles = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 6
    i_radii = [0.4, 0.6, 0.4]
    i_pts = np.column_stack([
        [r * np.cos(a) for r, a in zip(i_radii, i_angles)],
        [r * np.sin(a) for r, a in zip(i_radii, i_angles)]
    ])
    pts_mixed2 = np.vstack([b_pts, i_pts])
    inits.append(pts_mixed2)
    
    # 6. Random with seed
    rng = np.random.RandomState(42)
    pts_rand = rng.random((N, 2))
    inits.append(pts_rand)
    
    # 7. Regular polygon with one center point and 12 on circle
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    circle_12 = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    center = np.array([[0.0, 0.0]])
    pts_center = np.vstack([circle_12, center])
    inits.append(pts_center)
    
    # 8. Jittered grid
    rng2 = np.random.RandomState(123)
    grid_size = 4
    grid_x = np.linspace(0.1, 0.9, grid_size)
    grid_y = np.linspace(0.1, 0.9, grid_size)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    # Take first 13 points with jitter
    jitter = rng2.normal(0, 0.05, (13, 2))
    pts_grid = grid_pts[:13] + jitter
    inits.append(pts_grid)
    
    # 9. Fibonacci spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    angles_fib = np.arange(N) * golden_angle
    radii_fib = np.sqrt(np.arange(N) / N) * 0.95
    pts_fib = np.column_stack([radii_fib * np.cos(angles_fib), radii_fib * np.sin(angles_fib)])
    inits.append(pts_fib)
    
    # 10. Two concentric rings
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_pts = np.column_stack([0.9 * np.cos(angles_outer), 0.9 * np.sin(angles_outer)])
    angles_inner = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 8
    inner_pts = np.column_stack([0.4 * np.cos(angles_inner), 0.4 * np.sin(angles_inner)])
    pts_rings = np.vstack([outer_pts, inner_pts])
    inits.append(pts_rings)
    
    return inits

def _differential_evolution_search(bounds, time_budget, start_time):
    """Run differential evolution to maximize smooth score."""
    deadline = start_time + time_budget
    
    def timeout_callback(xk, convergence):
        return time.time() > deadline - 5
    
    try:
        res = differential_evolution(
            _neg_smooth_score,
            bounds,
            args=(50.0,),
            x0=None,
            popsize=15,
            maxiter=50,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            callback=timeout_callback,
            init='latinhypercube'
        )
        if res.x is not None:
            return res.x
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    inits = _generate_initializations()
    
    best_pts = None
    best_score = -1
    
    # Evaluate all initializations
    for i, init in enumerate(inits):
        if time.time() > deadline - 60:
            break
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()
    
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_pts)
    
    # Phase 1: Differential evolution for global search
    remaining = deadline - 40 - time.time()
    if remaining > 30:
        bounds = [(-1.0, 1.0)] * (2 * N)
        de_result = _differential_evolution_search(bounds, remaining, start_time)
        if de_result is not None:
            de_pts = de_result.reshape(N, 2)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts
    
    # Phase 2: Polish each good initialization with L-BFGS-B on smooth objective
    for i, init in enumerate(inits):
        if time.time() > deadline - 25:
            break
        s_init = _score(init)
        if s_init < best_score * 0.7:
            continue
        
        remaining = deadline - 20 - time.time()
        if remaining < 8:
            break
        
        # Use L-BFGS-B for better local optimization on smooth objective
        pts_opt, score_opt = _lbfgs_polish(init, time_budget=min(remaining, 10.0), start_time=start_time, beta=100.0)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    # Phase 3: Nelder-Mead polish on the best result
    remaining = deadline - 10 - time.time()
    if remaining > 5:
        best_pts, best_score = _nelder_mead_polish(best_pts, time_budget=min(remaining, 8.0), start_time=start_time)
    
    # Phase 4: Final L-BFGS-B polish with higher beta
    remaining = deadline - 3 - time.time()
    if remaining > 3:
        best_pts, best_score = _lbfgs_polish(best_pts, time_budget=min(remaining, 4.0), start_time=start_time, beta=200.0)
    
    return best_pts

# EVOLVE-BLOCK-END
