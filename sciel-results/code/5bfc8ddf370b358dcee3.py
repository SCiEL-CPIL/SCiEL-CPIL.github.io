"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy:
  1. Centrally symmetric polygon parameterization: 14 points as 7 antipodal pairs
     defined by 7 independent edge vectors. This enforces convexity and central
     symmetry by construction, reducing search space to 14 dimensions.
  2. Ring-structured parameterization (2-ring 7+7) as an alternative representation.
  3. Fast vectorized triangle-area computation using pre-computed index combinations.
  4. Multi-phase optimization: Nelder-Mead on parameter spaces, then Cartesian NM/Powell/L-BFGS.
  5. Differential evolution for global exploration in Cartesian space.
  6. Time-guarded to finish well within 150s.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14

# Pre-compute all triangle index combinations
tri_indices = np.array(list(itertools.combinations(range(N), 3)))  # shape (364, 3)


def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    p = points
    pi = p[tri_indices[:, 0]]
    pj = p[tri_indices[:, 1]]
    pk = p[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _all_triangle_areas(points):
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _objective_hard(flat):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


def _objective_soft(flat, beta=80.0):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min


# ---------- Centrally symmetric polygon parameterization ----------
# 14 points as 7 antipodal pairs. Define 7 edge vectors e_0..e_6.
# The polygon vertices are built by cumulative sum of edges, with the
# second half being the negative of the first half (central symmetry).
# 
# For a centrally symmetric convex 14-gon, we can parameterize by 7 edge
# vectors. The vertices are:
#   v_0 = 0
#   v_i = e_0 + e_1 + ... + e_{i-1} for i=1..7
#   v_{7+i} = -v_{7-i} for i=0..6  (i.e., v_7 = -v_0 = 0? No...)
#
# Actually, for central symmetry with 14 vertices:
#   v_i and v_{i+7} are antipodal: v_{i+7} = -v_i
#   So we only need to define v_0..v_6, and v_7..v_13 = -v_6..-v_0
#
# The edges are: e_i = v_{i+1} - v_i for i=0..6, and e_7 = v_0 - v_6 (closing)
# But with central symmetry, the polygon has 7 "half-edges" on one side.
#
# Better: define 7 vectors a_0..a_6 that are the vertices of the "half-polygon".
# The full 14-gon vertices are: a_0, a_1, ..., a_6, -a_6, -a_5, ..., -a_0
# For convexity, the a_i must be in convex position with the origin inside.
#
# Even simpler: parameterize by 7 angles and 7 radii (radial parameterization
# of the half-polygon), ensuring the points are in order around the origin.

def _decode_central_symmetric(params):
    """
    params: [r_0, ..., r_6, theta_0, ..., theta_6]
    14 parameters: 7 radii and 7 angles for the first 7 vertices.
    The remaining 7 vertices are the antipodes.
    
    To ensure convexity and proper ordering, we enforce that the angles
    are monotonically increasing in [0, pi) and the radii are positive.
    """
    r = params[0:7]
    theta = params[7:14]
    
    # Ensure angles are in [0, pi) and sorted
    # We'll use a soft constraint: map raw params to valid range
    # theta_i should be in [0, pi) and increasing
    # Use cumulative approach: theta_0 in [0, pi/7), theta_i = theta_0 + sum of gaps
    
    # Actually, let's just use the raw angles and sort them, then assign
    # radii in the same order. This is simpler.
    
    # Clamp radii to positive
    r = np.maximum(r, 0.01)
    
    # Map angles to [0, pi)
    theta = np.mod(theta, np.pi)
    
    # Sort by angle to ensure proper ordering
    order = np.argsort(theta)
    theta_sorted = theta[order]
    r_sorted = r[order]
    
    # Build 7 vertices
    x1 = r_sorted * np.cos(theta_sorted)
    y1 = r_sorted * np.sin(theta_sorted)
    
    # Build 7 antipodal vertices
    x2 = -x1[::-1]
    y2 = -y1[::-1]
    
    # Full 14 points
    pts = np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])
    
    return pts


def _objective_central_symmetric(params):
    pts = _decode_central_symmetric(params)
    return _objective_hard(pts.flatten())


def _objective_central_symmetric_soft(params, beta=80.0):
    pts = _decode_central_symmetric(params)
    return _objective_soft(pts.flatten(), beta)


def _generate_central_symmetric_inits():
    """Generate diverse initial parameter sets for centrally symmetric polygon."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 14-gon (centrally symmetric)
    # Angles: 0, pi/7, 2pi/7, ..., 6pi/7
    # Radii: all 1.0
    r = np.ones(7)
    theta = np.linspace(0, np.pi, 7, endpoint=False)
    inits.append(np.concatenate([r, theta]).copy())
    
    # 2. Elliptical distortions
    for rx, ry in [(1.1, 0.9), (0.9, 1.1), (1.2, 0.85), (0.85, 1.2), (1.3, 0.77)]:
        r = np.ones(7)
        theta = np.linspace(0, np.pi, 7, endpoint=False)
        # Apply elliptical stretch by modifying radii
        r_adj = r * np.sqrt(rx**2 * np.cos(theta)**2 + ry**2 * np.sin(theta)**2)
        inits.append(np.concatenate([r_adj, theta]).copy())
    
    # 3. Varying radii
    for seed in range(15):
        rs = np.random.default_rng(seed * 13 + 7)
        r = rs.uniform(0.7, 1.3, 7)
        theta = np.linspace(0, np.pi, 7, endpoint=False) + rs.normal(0, 0.1, 7)
        inits.append(np.concatenate([r, theta]).copy())
    
    # 4. Larger perturbations
    for seed in range(10):
        rs = np.random.default_rng(seed * 23 + 11)
        r = rs.uniform(0.5, 1.5, 7)
        theta = rs.uniform(0, np.pi, 7)
        inits.append(np.concatenate([r, theta]).copy())
    
    # 5. Ring-like: alternate large/small radii
    for seed in range(5):
        rs = np.random.default_rng(seed * 31 + 17)
        r = np.ones(7)
        r[::2] = rs.uniform(0.8, 1.2)
        r[1::2] = rs.uniform(0.6, 1.0)
        theta = np.linspace(0, np.pi, 7, endpoint=False)
        inits.append(np.concatenate([r, theta]).copy())
    
    return inits


# ---------- Ring parameterization (2 rings of 7) ----------
def _decode_ring2(params):
    """Decode 18-dim ring params to (N,2) points."""
    r1 = params[0]
    r2 = params[1]
    phi1 = params[2]
    phi2 = params[3]
    d1 = params[4:11]
    d2 = params[11:18]
    
    base_angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    base_angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    
    angles1 = phi1 + base_angles1 + d1
    angles2 = phi2 + base_angles2 + d2
    
    x1 = r1 * np.cos(angles1)
    y1 = r1 * np.sin(angles1)
    x2 = r2 * np.cos(angles2)
    y2 = r2 * np.sin(angles2)
    
    return np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])


def _objective_ring2(params):
    pts = _decode_ring2(params)
    return _objective_hard(pts.flatten())


def _objective_ring2_soft(params, beta=80.0):
    pts = _decode_ring2(params)
    return _objective_soft(pts.flatten(), beta)


def _make_ring2_inits():
    """Generate diverse initial ring-2 parameter vectors."""
    inits = []
    rng = np.random.default_rng(42)
    
    for r1 in [0.3, 0.5, 0.6, 0.7, 0.8, 0.85]:
        for phi2 in [np.pi / 7, 0.0, np.pi / 14]:
            params = np.zeros(18)
            params[0] = r1
            params[1] = 1.0
            params[2] = 0.0
            params[3] = phi2
            inits.append(params)
    
    for js in [0.05, 0.1, 0.15]:
        params = np.zeros(18)
        params[0] = 0.6
        params[1] = 1.0
        params[3] = np.pi / 7
        params[4:18] = rng.normal(0, js, size=14)
        inits.append(params)
    
    for seed in range(10):
        rs = np.random.default_rng(seed * 13 + 7)
        params = np.zeros(18)
        params[0] = rs.uniform(0.3, 0.8)
        params[1] = rs.uniform(0.9, 1.3)
        params[2] = rs.uniform(-0.2, 0.2)
        params[3] = rs.uniform(0.0, 2 * np.pi)
        params[4:18] = rs.normal(0, 0.1, size=14)
        inits.append(params)
    
    return inits


# ---------- Optimizer helpers ----------

def _nm(flat0, deadline, obj, max_iter=2000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None


def _powell(flat0, deadline, obj, max_iter=1000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-11, 'ftol': 1e-13})
        return res
    except Exception:
        return None


def _lbfgs(flat0, deadline, obj, max_iter=3000):
    if time.time() > deadline - 5:
        return None
    try:
        res = minimize(obj, flat0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res
    except Exception:
        return None


def _normalize_unit_area(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points


def _make_cartesian_inits():
    """Generate diverse initial point configurations (Cartesian)."""
    inits = []
    rng = np.random.default_rng(42)
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    for r1 in [0.3, 0.5, 0.6, 0.7, 0.85]:
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        inits.append(pts)
    
    for r1, r2 in [(0.3, 0.6), (0.4, 0.75), (0.5, 0.85)]:
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        a3 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            r2 * np.column_stack([np.cos(a2), np.sin(a2)]),
            np.column_stack([np.cos(a3), np.sin(a3)])
        ])
        inits.append(pts)
    
    for seed in range(10):
        rs = np.random.default_rng(seed * 37 + 19)
        pts = rs.uniform(-1, 1, size=(N, 2))
        inits.append(pts)
    
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    for js in [0.05, 0.1, 0.2]:
        inits.append(base + rng.normal(0, js, size=(N, 2)))
    
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7), (0.8, 1.2)]:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        inits.append(np.column_stack([rx * np.cos(a), ry * np.sin(a)]))
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    best_score = 0.0
    best_pts = None
    best_obj = float('inf')
    
    def update_best(pts):
        nonlocal best_score, best_pts, best_obj
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_obj = _objective_hard(pts.flatten())
    
    # Phase 1: Centrally symmetric polygon optimization (14D parameter space)
    cs_inits = _generate_central_symmetric_inits()
    
    for i, p0 in enumerate(cs_inits):
        if time.time() > deadline - 60:
            break
        res = _nm(p0, deadline, _objective_central_symmetric, max_iter=1500)
        if res is not None and res.fun < best_obj:
            pts = _decode_central_symmetric(res.x)
            update_best(pts)
    
    # Phase 2: Ring-based optimization (18D parameter space)
    ring_inits = _make_ring2_inits()
    
    for i, p0 in enumerate(ring_inits):
        if time.time() > deadline - 50:
            break
        res = _nm(p0, deadline, _objective_ring2, max_iter=1200)
        if res is not None and res.fun < best_obj:
            pts = _decode_ring2(res.x)
            update_best(pts)
    
    # Phase 3: Cartesian local refinement from best found
    if best_pts is not None and time.time() < deadline - 35:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=2000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
    
    # Phase 4: L-BFGS-B with soft objective
    if best_pts is not None and time.time() < deadline - 25:
        res = _lbfgs(best_pts.flatten(), deadline, _objective_soft, max_iter=2000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
    
    # Phase 5: Differential Evolution (Global Search)
    if time.time() < deadline - 15:
        try:
            bounds = [(-1.5, 1.5)] * (N * 2)
            de_res = differential_evolution(
                _objective_hard, bounds,
                seed=42, maxiter=40, tol=1e-8,
                popsize=12, mutation=(0.4, 1.0), recombination=0.7,
                polish=False, update_interval=1,
                init='sobol'
            )
            if de_res.fun < best_obj:
                update_best(de_res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 6: Final Polish
    if best_pts is not None and time.time() < deadline - 5:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=3000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
        
        if time.time() < deadline - 2:
            res = _powell(best_pts.flatten(), deadline, _objective_hard, max_iter=1500)
            if res is not None and res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
    
    # Fallback to regular polygon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    best_pts = _normalize_unit_area(best_pts)
    
    return best_pts
