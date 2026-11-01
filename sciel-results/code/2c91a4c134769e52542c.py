import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2):
    """
    Generate points on two concentric circles.
    n1, n2: number of points on outer and inner ring.
    r1, r2: radii.
    rot1, rot2: rotations.
    """
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    return np.vstack([pts1, pts2])

def _neg_score(params):
    """
    Negative score for two-ring configuration.
    params: [r_ratio, rot1, rot2]
    r1 is fixed to 1.0.
    """
    r_ratio = params[0]
    rot1 = params[1]
    rot2 = params[2]
    
    if r_ratio <= 0.0 or r_ratio >= 1.0:
        return 1e6 # Penalize invalid ratio (we want inner < outer typically, or just distinct)
        # Actually, let's allow any positive ratio. The hull area will handle scaling.
        # But if r_ratio is 0, points collapse.
    
    r1 = 1.0
    r2 = r_ratio * r1
    
    # We need to determine n1, n2 from the closure or pass them. 
    # Since this is a helper, let's assume it's used inside a closure that knows n1, n2.
    # To make it standalone for minimize, we should use a lambda or partial.
    
    # For now, this function is a placeholder for the logic inside construct.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None

    def _update_best(points):
        nonlocal best_score, best_points
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    def _optimize_coordinates(points, max_iter=200, tol=1e-8):
        """
        Local optimization of coordinates using Nelder-Mead on the exact negative score.
        """
        if time.time() > deadline - 5:
            return points
        
        def _neg_score_coords(x_flat):
            return -_score(x_flat.reshape(-1, 2))
        
        try:
            res = scipy.optimize.minimize(
                _neg_score_coords,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
            )
            return res.x.reshape(-1, 2)
        except Exception:
            return points

    # 1. Systematic Search over Two-Ring Configurations
    # Splits: (1,13) to (13,1). Symmetry: (n1, n2) ~ (n2, n1) if we swap radii/rotations appropriately?
    # Actually, (7,7) is symmetric. (6,8) and (8,6) are distinct in terms of which ring is outer?
    # Let's fix outer ring count n_out and inner n_in.
    # We assume outer radius r1=1, inner radius r2=r_ratio.
    
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4), (3, 11), (11, 3), (2, 12), (12, 2), (1, 13), (13, 1)]
    
    candidates = []
    
    for n_out, n_in in splits:
        if time.time() > deadline - 20:
            break
            
        def _ring_score(params, n_out_val=n_out, n_in_val=n_in):
            r_ratio = params[0]
            rot_out = params[1]
            rot_in = params[2]
            
            if r_ratio < 0.1 or r_ratio > 0.9:
                return 1e6
                
            r1 = 1.0
            r2 = r_ratio
            
            pts = _generate_two_ring_config(n_out_val, n_in_val, r1, r2, rot_out, rot_in)
            return -_score(pts)
        
        # Initial guesses
        # Rotations: usually 0 or pi/n is good.
        # Ratio: 0.5, 0.7, 0.8
        inits = [
            [0.5, 0.0, 0.0],
            [0.7, 0.0, np.pi/n_in],
            [0.8, 0.0, 0.0],
            [0.6, np.pi/n_out, np.pi/n_in],
            [0.4, 0.0, 0.0]
        ]
        
        for init in inits:
            if time.time() > deadline - 10:
                break
            try:
                res = scipy.optimize.minimize(
                    _ring_score,
                    init,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                r_ratio, rot_out, rot_in = res.x
                pts = _generate_two_ring_config(n_out, n_in, 1.0, r_ratio, rot_out, rot_in)
                sc = _score(pts)
                candidates.append((sc, pts.copy()))
            except Exception:
                pass

    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Refine Top Candidates with Coordinate Optimization
    # Take the top 5 distinct configurations (by score) and refine them.
    refined = set()
    for sc, pts in candidates:
        if time.time() > deadline - 15:
            break
        
        # Check if this point set is similar to one we already refined
        # Simple check: min distance to existing refined points
        is_dup = False
        for _, ref_pts in refined:
            if np.min(np.linalg.norm(pts - ref_pts, axis=1)) < 1e-3:
                is_dup = True
                break
        
        if is_dup:
            continue
            
        refined.add((sc, pts))
        
        # Refine
        refined_pts = _optimize_coordinates(pts, max_iter=100)
        _update_best(refined_pts)
        
        # Try a small jitter + refine to escape local minima
        if time.time() < deadline - 10:
            rng = np.random.RandomState(42)
            jittered = pts + rng.normal(0, 0.01, pts.shape)
            jittered_pts = _optimize_coordinates(jittered, max_iter=50)
            _update_best(jittered_pts)

    # 3. Fallback: Regular Polygon and Random Restarts if no good ring found
    if best_score < 0.01:
        reg = _regular_polygon(N, 1.0)
        _update_best(reg)
        ref_reg = _optimize_coordinates(reg, max_iter=100)
        _update_best(ref_reg)

    # 4. Final Global-ish Random Restarts with Light Optimization
    # If we have time, try a few random starts
    rng = np.random.RandomState(123)
    for i in range(5):
        if time.time() > deadline - 5:
            break
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        init = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        # Quick optimize
        opt_pts = _optimize_coordinates(init, max_iter=50)
        _update_best(opt_pts)

    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
