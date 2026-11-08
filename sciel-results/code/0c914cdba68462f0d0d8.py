"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import sys

def _all_triangle_areas(points):
    """Compute all triangle areas for n points (n choose 3 values)."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas.append(area)
    return np.array(areas)

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    return np.min(_all_triangle_areas(points))

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _soft_score(points, temperature=0.01):
    """
    Smooth surrogate: log-sum-exp approximation of min triangle area / hull area.
    Uses -logsumexp(-areas/temp)/temp as a smooth approximation of min(areas).
    """
    areas = _all_triangle_areas(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    
    # Smooth min approximation: -logsumexp(-a/temp) * temp
    neg_areas = -areas / temperature
    max_neg = np.max(neg_areas)
    logsumexp = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    smooth_min = -logsumexp * temperature
    
    return smooth_min / hull_area

def _neg_soft_score(points, temperature=0.01):
    """Negative soft score for minimization."""
    return -_soft_score(points, temperature)

def _regular_polygon(n, radius=1.0):
    """Generate points on a regular polygon."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_rings(n, inner_frac=0.5):
    """Generate points on two concentric rings."""
    n_outer = n // 2
    n_inner = n - n_outer
    
    outer_angles = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    inner_angles = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + np.pi/n_inner
    
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_pts = inner_frac * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    return np.vstack([outer_pts, inner_pts])

def _hexagonal_packing(n):
    """Generate hexagonal packing arrangement."""
    points = []
    # Center
    points.append([0, 0])
    # First ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        points.append([np.cos(angle), np.sin(angle)])
    # Second ring (7 points to make 14 total)
    for i in range(7):
        angle = i * 2 * np.pi / 7 + np.pi/7
        points.append([2 * np.cos(angle), 2 * np.sin(angle)])
    return np.array(points)

def _optimize_powell(points, deadline, temperature=0.01):
    """
    Optimize using Powell's method on the smooth log-sum-exp surrogate,
    then polish with exact min.
    """
    # Flatten points for optimization
    x0 = points.flatten()
    
    # Track time
    start_time = time.time()
    timeout_flag = [False]
    
    def objective(x):
        if timeout_flag[0]:
            return 0.0
        if time.time() > deadline - 2:
            timeout_flag[0] = True
            return 0.0
        pts = x.reshape(-1, 2)
        return _neg_soft_score(pts, temperature)
    
    try:
        result = minimize(
            objective, x0, method='Powell',
            options={'maxiter': 200, 'maxfev': 5000}
        )
        if timeout_flag[0]:
            return points, _score(points)
        
        opt_points = result.x.reshape(-1, 2)
        opt_score = _score(opt_points)
        return opt_points, opt_score
    except Exception:
        return points, _score(points)

def _polish_exact(points, deadline, max_iters=2000):
    """
    Fine-tune using the exact min objective with small perturbations.
    """
    points = points.copy()
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    step_size = 0.01
    
    for it in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        # Try moving a random point
        idx = np.random.randint(len(points))
        delta = np.random.normal(0, step_size, 2)
        new_points = points.copy()
        new_points[idx] += delta
        
        new_score = _score(new_points)
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            if it % 200 == 0:
                step_size *= 0.95
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Generate diverse initial candidates
    candidates = []
    
    # Regular polygon
    candidates.append(_regular_polygon(N))
    
    # Two rings with different fractions
    for frac in [0.3, 0.5, 0.6, 0.7]:
        candidates.append(_two_rings(N, frac))
    
    # Hexagonal packing
    candidates.append(_hexagonal_packing(N))
    
    # Random initializations
    rng = np.random.RandomState(42)
    for seed in range(8):
        rng_seed = np.random.RandomState(seed * 137 + 7)
        candidates.append(rng_seed.rand(N, 2))
    
    # Evaluate initial candidates
    best_points = None
    best_score = -1
    
    for pts in candidates:
        if time.time() > deadline:
            break
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Use Powell on smooth surrogate from best and a few other candidates
    # This is the structural change: smooth log-sum-exp enables Powell to work
    for i, pts in enumerate(candidates):
        if time.time() > deadline - 15:
            break
        if np.allclose(pts, best_points, atol=1e-6):
            continue
        
        # Use decreasing temperature for annealing effect
        temp = 0.05 if i < 3 else 0.02
        opt_pts, opt_score = _optimize_powell(pts, deadline, temperature=temp)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
    
    # Also optimize the best initial with tighter temperature
    if time.time() < deadline - 15:
        opt_pts, opt_score = _optimize_powell(best_points, deadline, temperature=0.01)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
    
    # Final polish with exact objective
    if best_points is not None and time.time() < deadline - 5:
        best_points, best_score = _polish_exact(best_points, deadline, max_iters=3000)
    
    return best_points if best_points is not None else _regular_polygon(N)
# EVOLVE-BLOCK-END
