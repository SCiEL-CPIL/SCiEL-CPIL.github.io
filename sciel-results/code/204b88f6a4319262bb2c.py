"""Rewritten program for Heilbronn triangles: 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
seed_val = 42

# EVOLVE-BLOCK-START

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices for vectorized area computation
# There are C(11, 3) = 165 triangles
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = _TRI_IDX.shape[0]

# Precompute for each point, which triangle indices involve it
# This allows incremental area updates
_POINT_TRI_MAP = {}
for i in range(N):
    _POINT_TRI_MAP[i] = np.where((_TRI_IDX == i).any(axis=1))[0]

def _all_areas(pts):
    """Fully vectorized computation of all triangle areas."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return areas

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    return float(np.min(_all_areas(pts)))

def _incremental_min_area(pts, moved_idx, old_areas):
    """
    Update areas when one point moves. Only recompute triangles involving moved_idx.
    Returns (new_min_area, updated_areas_array).
    """
    new_areas = old_areas.copy()
    tri_indices = _POINT_TRI_MAP[moved_idx]
    
    pi_idx = _TRI_IDX[tri_indices, 0]
    pj_idx = _TRI_IDX[tri_indices, 1]
    pk_idx = _TRI_IDX[tri_indices, 2]
    
    pi = pts[pi_idx]
    pj = pts[pj_idx]
    pk = pts[pk_idx]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    new_areas[tri_indices] = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    
    return float(np.min(new_areas)), new_areas

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle using barycentric coordinates."""
    x, y = float(p[0]), float(p[1])
    c = 2.0 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1.0 - b - c
    
    eps = 1e-10
    a = max(eps, min(1.0 - 2.0*eps, a))
    b = max(eps, min(1.0 - 2.0*eps, b))
    c = max(eps, min(1.0 - 2.0*eps, c))
    
    s = a + b + c
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _soft_min_area(pts, beta=100.0):
    """Soft minimum of all triangle areas (smooth approximation)."""
    areas = _all_areas(pts)
    # Use log-sum-exp trick for numerical stability
    min_a = np.min(areas)
    if min_a > 0:
        log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a)))) / beta
        return min_a + log_sum
    return min_a

def _soft_min_neg(pts, beta=100.0):
    """Negative soft minimum (for minimization)."""
    return -_soft_min_area(pts, beta)

def _generate_initial_config(seed):
    """Generate a random initial configuration inside the triangle."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        p = r1 * V1 + r2 * V2
        pts.append(p)
    return np.array(pts)

def _generate_symmetric_config():
    """Generate a symmetric initial configuration based on known good patterns."""
    # Place points at vertices and along edges/interior in a symmetric pattern
    # This gives a good starting point for optimization
    pts = []
    # Three vertices (slightly inside)
    eps = 0.01
    pts.append([eps, eps])
    pts.append([1 - eps, eps])
    pts.append([0.5, H - eps * 2])
    
    # Three midpoints of edges (slightly inside)
    pts.append([0.5, eps])
    pts.append([0.5 + 0.5*(1-eps), H*0.5*(1-eps)])
    pts.append([0.5*(1-eps), H*0.5*(1-eps)])
    
    # Three interior points forming a smaller triangle
    cx, cy = 0.5, H/3
    r = 0.15
    for k in range(3):
        angle = 2 * np.pi * k / 3
        pts.append([cx + r * np.cos(angle), cy + r * np.sin(angle)])
    
    return _clamp_all(np.array(pts))

def _simulated_annealing(start_points, time_limit, seed):
    """
    Perform Simulated Annealing with incremental area updates to maximize the minimum triangle area.
    """
    rng = np.random.default_rng(seed)
    
    points = np.array(start_points, dtype=float)
    points = _clamp_all(points)
    current_areas = _all_areas(points)
    current_score = float(np.min(current_areas))
    
    best_points = points.copy()
    best_score = current_score
    
    # SA Parameters
    T = 0.003
    T_min = 1e-12
    alpha = 0.9998
    
    step_size = 0.02
    min_step = 0.001
    
    start_time = time.time()
    steps = 0
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            # Restart with slightly perturbed best
            T = 0.002
            step_size = max(min_step, step_size * 0.8)
            # Small perturbation of best
            pert_idx = rng.integers(N)
            points[pert_idx] = _clamp_to_triangle(best_points[pert_idx] + rng.normal(0, 0.01, 2))
            current_areas = _all_areas(points)
            current_score = float(np.min(current_areas))
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        
        # Pick a random point to perturb
        idx = rng.integers(N)
        
        # Perturb with Gaussian noise
        noise = rng.normal(0, step_size, size=2)
        new_point = points[idx] + noise
        new_point = _clamp_to_triangle(new_point)
        
        old_point = points[idx].copy()
        points[idx] = new_point
        
        # Incremental area update - only recompute triangles involving idx
        new_score, new_areas = _incremental_min_area(points, idx, current_areas)
        
        delta = new_score - current_score
        if delta > 0 or (T > 0 and rng.random() < np.exp(delta / (T + 1e-15))):
            current_score = new_score
            current_areas = new_areas
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            points[idx] = old_point
            # Restore areas (they were modified in-place, so recompute for affected triangles)
            old_score, current_areas = _incremental_min_area(points, idx, new_areas)
            
        T *= alpha
        step_size *= 0.99997
        if step_size < min_step:
            step_size = min_step
        steps += 1

    return best_points, best_score

def _local_refine(pts, time_budget=10.0):
    """
    Local refinement using SLSQP with barycentric parameterization.
    Points are represented by barycentric coordinates (a,b,c) with a+b+c=1, a,b,c >= 0.
    We optimize over 2 coordinates per point (a, b), with c = 1 - a - b.
    """
    # Convert to barycentric coordinates
    bary = []
    for p in pts:
        x, y = float(p[0]), float(p[1])
        c = 2.0 * y / np.sqrt(3)
        b = x - 0.5 * c
        a = 1.0 - b - c
        # Clamp to valid range
        a = max(0.0, min(1.0, a))
        b = max(0.0, min(1.0, b))
        c = max(0.0, min(1.0, c))
        s = a + b + c
        if s < 1e-15:
            a, b, c = 1.0/3, 1.0/3, 1.0/3
        else:
            a /= s; b /= s; c /= s
        bary.append([a, b])
    bary = np.array(bary)
    
    def _bary_to_cartesian(flat_bary):
        """Convert flat barycentric array to Cartesian points."""
        pts = []
        for i in range(N):
            a = flat_bary[2*i]
            b = flat_bary[2*i+1]
            c = 1.0 - a - b
            # Clamp
            a = max(1e-12, a)
            b = max(1e-12, b)
            c = max(1e-12, c)
            s = a + b + c
            a /= s; b /= s; c /= s
            pts.append(a * V0 + b * V1 + c * V2)
        return np.array(pts)
    
    def _objective(flat_bary):
        pts = _bary_to_cartesian(flat_bary)
        return -_min_area(pts)
    
    # Use SLSQP
    try:
        res = minimize(
            _objective,
            bary.flatten(),
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
        if res.success or res.fun < _objective(bary.flatten()):
            res_pts = _bary_to_cartesian(res.x)
            res_pts = _clamp_all(res_pts)
            return res_pts, _min_area(res_pts)
    except Exception:
        pass
    
    # Fallback: Nelder-Mead on Cartesian
    try:
        res_nm = minimize(
            lambda flat: -_min_area(_clamp_all(flat.reshape(-1, 2))),
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        nm_pts = _clamp_all(res_nm.x.reshape(-1, 2))
        nm_score = _min_area(nm_pts)
        if nm_score > _min_area(pts):
            return nm_pts, nm_score
    except Exception:
        pass
    
    return pts, _min_area(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin: evaluator kills at 150s
    
    def time_left():
        return deadline - time.time()
        
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Global Search using Simulated Annealing from multiple diverse starts
    starts = [
        _generate_symmetric_config(),
        _generate_initial_config(42),
        _generate_initial_config(123),
        _generate_initial_config(789),
        _generate_initial_config(101),
        _generate_initial_config(555),
        _generate_initial_config(2024),
        _generate_initial_config(3141),
    ]
    
    for i, init_pts in enumerate(starts):
        if time_left() < 25.0:
            break
            
        # Allocate time based on position
        if i == 0:
            time_budget = 35.0
        elif i == 1:
            time_budget = 30.0
        elif i == 2:
            time_budget = 20.0
        else:
            time_budget = 12.0
            
        time_budget = min(time_budget, time_left() - 25.0)
        if time_budget < 3.0:
            break
            
        try:
            sa_pts, sa_score = _simulated_annealing(init_pts, time_budget, seed=seed_val + i * 1000)
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts
        except Exception:
            pass

    # If no points found, use symmetric fallback
    if best_pts is None:
        best_pts = _generate_symmetric_config()
        best_score = _min_area(best_pts)

    # Phase 2: Local Refinement
    if best_pts is not None and time_left() > 15.0:
        try:
            refined_pts, refined_score = _local_refine(best_pts, time_budget=min(20.0, time_left() - 8.0))
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts
        except Exception:
            pass

    # Phase 3: Final Polish with tight NM
    if best_pts is not None and time_left() > 8.0:
        try:
            res_final = minimize(
                lambda flat: -_min_area(_clamp_all(flat.reshape(-1, 2))),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            final_pts = _clamp_all(res_final.x.reshape(-1, 2))
            final_score = _min_area(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts
        except Exception:
            pass

    # Final verification: ensure all points are strictly inside
    final_pts = _clamp_all(best_pts)
    
    return final_pts

# EVOLVE-BLOCK-END
