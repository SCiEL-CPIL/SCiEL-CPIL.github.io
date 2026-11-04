"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

def _min_triangle_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    i_idx, j_idx, k_idx = _get_indices(n)
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

_IDX_CACHE = {}
def _get_indices(n):
    if n not in _IDX_CACHE:
        i_idx, j_idx, k_idx = [], [], []
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    i_idx.append(i)
                    j_idx.append(j)
                    k_idx.append(k)
        _IDX_CACHE[n] = (np.array(i_idx), np.array(j_idx), np.array(k_idx))
    return _IDX_CACHE[n]

def _convex_hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _single_ring_from_angles(angles, radius=1.0):
    """Place N points on a circle of given radius with specified angles."""
    angles = np.asarray(angles, dtype=float)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _angular_min_triangle_area(angles, radius=1.0):
    """Compute min triangle area for points on a circle using angular differences."""
    n = len(angles)
    if n < 3:
        return 0.0
    # Sort angles for consistent ordering
    angles_sorted = np.sort(angles)
    # For points on a circle, triangle area = 0.5 * R^2 * |sin(a) + sin(b) + sin(c)|
    # where a, b, c are the three arc separations (summing to 2*pi)
    # Actually, area = 0.5 * R^2 * |sin(theta_i - theta_j) + sin(theta_j - theta_k) + sin(theta_k - theta_i)|
    # But simpler: for three points at angles a1 < a2 < a3 on unit circle,
    # area = 0.5 * R^2 * |sin(a2-a1) + sin(a3-a2) + sin(a1+2pi-a3)|
    # Let's just compute directly using the cross product formula for accuracy
    
    i_idx, j_idx, k_idx = _get_indices(n)
    pts = _single_ring_from_angles(angles, radius)
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _angular_score(angles, radius=1.0):
    """Score for points on a circle: min triangle area / convex hull area."""
    pts = _single_ring_from_angles(angles, radius)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    min_area = _angular_min_triangle_area(angles, radius)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_score = -np.inf
    best_pts = None
    
    # Strategy 1: Optimize angular positions on a single ring
    # The key insight: for points on a circle, the problem reduces to 1D angular packing
    
    def _angular_objective(x):
        angles = x  # Already angles in [0, 2*pi)
        return -_angular_score(angles)
    
    # Initial guess: evenly spaced angles
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Try multiple initializations for the angular optimization
    inits = [
        base_angles.copy(),
        base_angles + np.random.RandomState(1).normal(0, 0.1, N),
        base_angles + np.random.RandomState(2).normal(0, 0.2, N),
        np.sort(np.random.RandomState(3).uniform(0, 2 * np.pi, N)),
    ]
    
    for init_angles in inits:
        if time.time() > deadline:
            break
        
        # Evaluate initial
        current_score = _angular_score(init_angles)
        if current_score > best_score:
            best_score = current_score
            best_pts = _single_ring_from_angles(init_angles)
        
        # Nelder-Mead optimization on angles
        try:
            res = minimize(_angular_objective, init_angles, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if -res.fun > best_score:
                best_score = -res.fun
                best_pts = _single_ring_from_angles(res.x)
        except Exception:
            pass
        
        # Powell optimization
        if time.time() < deadline - 15.0:
            try:
                res_powell = minimize(_angular_objective, init_angles, method='Powell',
                                      options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-12})
                if -res_powell.fun > best_score:
                    best_score = -res_powell.fun
                    best_pts = _single_ring_from_angles(res_powell.x)
            except Exception:
                pass
    
    # Strategy 2: Two-ring optimization with optimized radial and angular parameters
    if time.time() < deadline - 20.0:
        def _two_ring_params_to_pts(params):
            r_inner = params[0]
            angle_offset = params[1]
            # 7 points inner, 7 points outer
            inner_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            outer_angles = inner_angles + angle_offset
            inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
            outer_pts = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
            return np.vstack([inner_pts, outer_pts])
        
        def _two_ring_objective(params):
            pts = _two_ring_params_to_pts(params)
            return -_score(pts)
        
        # Optimize r_inner and angle_offset
        init_params = np.array([0.5, np.pi / 7])
        try:
            res_2ring = minimize(_two_ring_objective, init_params, method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if -res_2ring.fun > best_score:
                best_score = -res_2ring.fun
                best_pts = _two_ring_params_to_pts(res_2ring.x)
        except Exception:
            pass
        
        # Also try a few different initial r_inner values
        for r_init in [0.3, 0.4, 0.6, 0.7]:
            if time.time() > deadline:
                break
            init_params = np.array([r_init, np.pi / 7])
            try:
                res_2ring = minimize(_two_ring_objective, init_params, method='Nelder-Mead',
                                     options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
                if -res_2ring.fun > best_score:
                    best_score = -res_2ring.fun
                    best_pts = _two_ring_params_to_pts(res_2ring.x)
            except Exception:
                pass
    
    # Strategy 3: Full 2D optimization from best single-ring solution
    if time.time() < deadline - 30.0 and best_pts is not None:
        def _full_2d_objective(x):
            return -_score(x.reshape(-1, 2))
        
        try:
            res_full = minimize(_full_2d_objective, best_pts.flatten(), method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if -res_full.fun > best_score:
                best_score = -res_full.fun
                best_pts = res_full.x.reshape(-1, 2)
        except Exception:
            pass
    
    # Final fallback
    if best_pts is None:
        best_pts = _single_ring_from_angles(base_angles)
    
    return best_pts
# EVOLVE-BLOCK-END
