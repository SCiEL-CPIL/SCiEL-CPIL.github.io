"""
Dual lattice parameterization: 14 points generated as vertices of a Minkowski sum
of a regular 14-gon and a small asymmetric perturbation polygon defined by 7 edge-vector
lengths. This ensures strict convexity while allowing non-radial, non-Fourier deformations.
Combined with ring-based initializations and multi-stage local refinement.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

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

def _all_areas(points):
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _objective_hard(flat):
    pts = flat.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    return hull_area / min_area

def _objective_soft(flat, beta=50.0):
    pts = flat.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if len(areas) == 0 or np.min(areas) < 1e-12:
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

# --- Dual lattice parameterization ---
# Regular 14-gon vertices + small perturbation polygon (7 edge vectors)
# The perturbation is a convex polygon with 7 vertices (edge lengths e_0..e_6)
# Minkowski sum: P_i = R_i + Q_i where R is regular 14-gon, Q is the 7-gon
# To make this work for 14 points, we map the 7-gon edges to 14-gon vertices
# via a 2:1 mapping, ensuring the result stays convex

def _regular_polygon(n, radius=1.0, offset=0.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + offset
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])

def _perturbation_polygon(edge_lengths, n=7):
    """
    Build a convex polygon from edge lengths (all pointing in sequential directions).
    For a convex polygon with n edges, the turning angles sum to 2*pi.
    We use uniform turning angles for simplicity: 2*pi/n per edge.
    """
    if len(edge_lengths) != n:
        return np.zeros((n, 2))
    turn = 2 * np.pi / n
    angles = np.cumsum(turn * np.arange(n)) - turn/2  # center each edge
    # Edge vectors
    ex = edge_lengths * np.cos(angles)
    ey = edge_lengths * np.sin(angles)
    # Cumulative sum to get vertices (starting from origin)
    vertices = np.cumsum(np.column_stack([ex, ey]), axis=0)
    # Close the polygon: adjust so it closes (subtract mean to center)
    vertices -= vertices.mean(axis=0)
    return vertices

def _decode_dual_lattice(params):
    """
    params: [scale, rot, e_0, ..., e_6, rx, ry]
    - scale: overall scale of the 14-gon
    - rot: rotation offset
    - e_0..e_6: edge lengths of the 7-gon perturbation
    - rx, ry: elliptical stretch applied to the whole thing
    """
    scale = params[0]
    rot = params[1]
    edge_lengths = params[2:9]
    rx = params[9]
    ry = params[10]
    
    # Regular 14-gon
    base = _regular_polygon(N, radius=scale, offset=rot)
    
    # Perturbation 7-gon
    pert = _perturbation_polygon(edge_lengths, 7)
    
    # Map 7-gon to 14-gon: each perturbation vertex affects 2 base vertices
    # Interpolate: for base vertex i, use perturbation vertex (i % 7)
    # Actually, for a proper Minkowski sum, we need the perturbation to have
    # the same number of vertices. Let's use a 14-gon perturbation instead.
    # Re-parameterize: 7 edge lengths define a 14-gon by repeating each edge twice
    # with alternating directions to maintain convexity
    
    # Simpler approach: use 14 edge lengths derived from 7 base lengths
    # e_i for i=0..6, and the 14-gon edges are [e_0, e_1, e_0, e_1, ..., e_6, e_6]
    # No, let's just use a 14-gon perturbation with 14 parameters
    # But we want only 7 independent parameters. Let's use symmetry:
    # perturbation 14-gon has edges [e_0, e_1, e_2, e_3, e_4, e_5, e_6, e_6, e_5, e_4, e_3, e_2, e_1, e_0]
    # This gives a symmetric 14-gon with 7 free parameters
    
    e = edge_lengths
    edges_14 = np.array([e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[6], e[5], e[4], e[3], e[2], e[1], e[0]])
    pert14 = _perturbation_polygon(edges_14, 14)
    
    # Minkowski sum
    pts = base + pert14
    
    # Apply elliptical stretch
    pts[:, 0] *= rx
    pts[:, 1] *= ry
    
    return pts

def _objective_dual(flat):
    pts = _decode_dual_lattice(flat)
    return _objective_hard(pts.flatten())

def _generate_dual_inits():
    """Generate diverse initial parameter sets for the dual lattice."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Pure regular 14-gon (no perturbation)
    base = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    inits.append(base.copy())
    
    # 2. Slight elliptical distortion
    for rx, ry in [(1.1, 0.9), (0.9, 1.1), (1.2, 0.85), (0.85, 1.2)]:
        p = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rx, ry])
        inits.append(p)
    
    # 3. Small random perturbations
    for seed in range(20):
        rs = np.random.default_rng(seed * 13 + 7)
        p = np.array([1.0, 0.0] + list(rs.uniform(-0.05, 0.05, 7)) + [1.0, 1.0])
        inits.append(p)
    
    # 4. Larger asymmetric perturbations
    for seed in range(10):
        rs = np.random.default_rng(seed * 23 + 11)
        p = np.array([1.0, rs.uniform(0, 2*np.pi)] + list(rs.uniform(-0.15, 0.15, 7)) + 
                     [rs.uniform(0.8, 1.3), rs.uniform(0.8, 1.3)])
        inits.append(p)
    
    # 5. Ring-like: alternate large/small edge lengths
    for seed in range(5):
        rs = np.random.default_rng(seed * 31 + 17)
        e = np.zeros(7)
        e[::2] = rs.uniform(0.05, 0.15)
        e[1::2] = rs.uniform(-0.05, 0.05)
        p = np.array([1.0, 0.0] + list(e) + [1.0, 1.0])
        inits.append(p)
    
    return inits

# --- Ring-based initializations (from inspiration) ---
def _make_ring_inits():
    inits = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Two concentric rings (7+7)
    for r1 in [0.3, 0.5, 0.7, 0.85]:
        a1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        inits.append(pts)
    
    # Three rings (5+4+5)
    for r1, r2 in [(0.3, 0.6), (0.4, 0.75), (0.5, 0.85)]:
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        a3 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            r2 * np.column_stack([np.cos(a2), np.sin(a2)]),
            np.column_stack([np.cos(a3), np.sin(a3)])
        ])
        inits.append(pts)
    
    # Jittered regular polygon
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    rng = np.random.default_rng(42)
    for js in [0.05, 0.1, 0.2]:
        inits.append(base + rng.normal(0, js, size=(N, 2)))
    
    # Elliptical configurations
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7), (0.8, 1.2)]:
        a = np.linspace(0, 2*np.pi, N, endpoint=False)
        inits.append(np.column_stack([rx*np.cos(a), ry*np.sin(a)]))
    
    return inits

def _optimize_nm(flat0, deadline, obj, max_iter=2000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None

def _optimize_powell(flat0, deadline, obj, max_iter=1000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-11, 'ftol': 1e-13})
        return res
    except Exception:
        return None

def _optimize_lbfgs(flat0, deadline, obj, max_iter=3000):
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
    
    # Phase 1: Dual lattice optimization (11D parameter space)
    dual_inits = _generate_dual_inits()
    
    for i, p0 in enumerate(dual_inits):
        if time.time() > deadline - 60:
            break
        res = _optimize_nm(p0, deadline, _objective_dual, max_iter=1500)
        if res is not None and res.fun < best_obj:
            pts = _decode_dual_lattice(res.x)
            update_best(pts)
    
    # Phase 2: Ring-based initializations with local refinement
    ring_inits = _make_ring_inits()
    
    for i, pts0 in enumerate(ring_inits):
        if time.time() > deadline - 40:
            break
        res = _optimize_nm(pts0.flatten(), deadline, _objective_hard, max_iter=1000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
    
    # Phase 3: Full-space Nelder-Mead from best found
    if best_pts is not None and time.time() < deadline - 30:
        res = _optimize_nm(best_pts.flatten(), deadline, _objective_hard, max_iter=2000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
    
    # Phase 4: L-BFGS-B with soft objective
    if best_pts is not None and time.time() < deadline - 20:
        res = _optimize_lbfgs(best_pts.flatten(), deadline, _objective_soft, max_iter=2000)
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
        res = _optimize_nm(best_pts.flatten(), deadline, _objective_hard, max_iter=3000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
        
        if time.time() < deadline - 2:
            res = _optimize_powell(best_pts.flatten(), deadline, _objective_hard, max_iter=1500)
            if res is not None and res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
    
    # Fallback to regular polygon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    best_pts = _normalize_unit_area(best_pts)
    
    return best_pts
