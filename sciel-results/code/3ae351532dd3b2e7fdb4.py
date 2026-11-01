import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START
def _triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(area)
    return np.array(areas)

def _score(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # in 2D, volume is area
        if hull_area <= 0:
            return 0.0
        areas = _triangle_areas(pts)
        min_area = np.min(areas)
        return min_area / hull_area
    except Exception:
        return 0.0

def _smooth_score(pts, beta=100.0):
    """Log-sum-exp smoothed version of min triangle area / hull area.
    Uses -logsumexp(-beta * areas) as a smooth approximation to min(areas)."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 0:
            return 0.0
        areas = _triangle_areas(pts)
        # Smooth min: -logsumexp(-beta * areas) / beta approximates min(areas)
        # For large beta, this approaches the true minimum
        shifted = -beta * areas
        max_shifted = np.max(shifted)
        logsumexp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        smooth_min = -logsumexp / beta
        return smooth_min / hull_area
    except Exception:
        return 0.0

def _get_points(x):
    """Reconstruct points from parameter vector. Fix p0=(0,0), p1=(1,0)."""
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    for i in range(N - 2):
        pts[i + 2, 0] = x[2*i]
        pts[i + 2, 1] = x[2*i + 1]
    return pts

def _negative_smooth_score(x, beta=100.0):
    """Negative smooth score for minimization."""
    pts = _get_points(x)
    return -_smooth_score(pts, beta)

def _negative_score(x):
    """Negative exact score for minimization."""
    pts = _get_points(x)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_x = None
    best_score = 0.0
    
    # Initial diverse starts
    rng = np.random.RandomState(42)
    
    # Regular polygon initialization
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    # Normalize: fix p0 at origin, p1 at (1,0)
    # Rotate so p1 is at (1,0)
    p1_idx = 1
    p0 = reg_pts[0]
    p1 = reg_pts[p1_idx]
    reg_pts = reg_pts - p0
    d = p1 - p0
    norm = np.linalg.norm(d)
    if norm > 1e-10:
        d = d / norm
        # Rotate
        cos_t, sin_t = d[0], d[1]
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        reg_pts = reg_pts @ R.T
    # Extract free parameters
    reg_x = reg_pts[2:].flatten()
    
    candidates = [reg_x]
    
    # Add random initializations
    for _ in range(5):
        x_rand = rng.uniform(-2, 2, size=2*(N-2))
        candidates.append(x_rand)
    
    # Add some structured initializations
    # Two clusters
    for _ in range(3):
        x_struct = np.zeros(2*(N-2))
        for i in range(N-2):
            if i % 2 == 0:
                x_struct[2*i] = rng.uniform(0.5, 1.5)
                x_struct[2*i+1] = rng.uniform(-0.5, 0.5)
            else:
                x_struct[2*i] = rng.uniform(-0.5, 0.5)
                x_struct[2*i+1] = rng.uniform(0.5, 1.5)
        candidates.append(x_struct)
    
    # Evaluate initial candidates
    for x_cand in candidates:
        pts = _get_points(x_cand)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_x = x_cand.copy()
    
    # Differential evolution with time budget
    bounds = [(-3, 3)] * (2*(N-2))
    
    # Use smoothed objective for DE
    pop_size = 30
    max_iter = 100
    
    de_result = None
    try:
        # Check time
        if time.time() - start_time > deadline - 30:
            de_result = None
        else:
            de_result = differential_evolution(
                _negative_smooth_score,
                bounds,
                args=(50.0,),  # beta for smoothing
                maxiter=max_iter,
                popsize=pop_size,
                tol=1e-6,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
    except Exception:
        de_result = None
    
    if de_result is not None and de_result.x is not None:
        x_de = de_result.x
        pts_de = _get_points(x_de)
        s_de = _score(pts_de)
        if s_de > best_score:
            best_score = s_de
            best_x = x_de.copy()
    
    # Local refinement with L-BFGS-B on exact objective
    if time.time() - start_time < deadline - 20:
        # Try refinement from best point
        try:
            res = minimize(
                _negative_score,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if res.x is not None:
                s_res = _score(_get_points(res.x))
                if s_res > best_score:
                    best_score = s_res
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Additional local refinements from diverse starting points
    if time.time() - start_time < deadline - 15:
        for x_cand in candidates[:5]:
            if time.time() - start_time > deadline - 10:
                break
            try:
                res = minimize(
                    _negative_score,
                    x_cand,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                if res.x is not None:
                    s_res = _score(_get_points(res.x))
                    if s_res > best_score:
                        best_score = s_res
                        best_x = res.x.copy()
            except Exception:
                pass
    
    # Final check
    if best_x is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    final_pts = _get_points(best_x)
    return final_pts
# EVOLVE-BLOCK-END
