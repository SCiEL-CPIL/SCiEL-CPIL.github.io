"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy (combined: B-spline boundary parameterization + centrally symmetric polygon + ring):
  1. Primary representation: closed cubic B-spline with 7 control points (14D).
     Points are sampled at equal arc-length intervals along the curve.
  2. Centrally symmetric polygon parameterization (7 antipodal pairs) as alternative.
  3. Ring-structured parameterization (2-ring 7+7) as another representation.
  4. Fast vectorized triangle-area computation using pre-computed index combinations.
  5. Multi-phase optimization: Nelder-Mead on parameter spaces, then Cartesian
     NM/Powell/L-BFGS-B with soft objective.
  6. Differential evolution for global exploration.
  7. Time-guarded to finish well within 150s.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
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


# ---------- B-spline convex boundary parameterization ----------
def _decode_bspline(ctrl_params):
    """
    Decode 14-dim control point parameters to 14 points on a B-spline curve.
    ctrl_params: shape (14,) = [x0,y0, x1,y1, ..., x6,y6]
    """
    ctrl = ctrl_params.reshape(7, 2)
    
    try:
        ctrl_closed = np.vstack([ctrl, ctrl[0]])
        t = np.arange(8)
        cs_x = CubicSpline(t, ctrl_closed[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, ctrl_closed[:, 1], bc_type='periodic')
        
        t_fine = np.linspace(0, 7, 2000)
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)
        
        dx = np.diff(x_fine)
        dy = np.diff(y_fine)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        total_s = s[-1]
        
        if total_s < 1e-10:
            raise ValueError("degenerate curve")
        
        s_targets = np.linspace(0, total_s, N, endpoint=False)
        x_pts = np.interp(s_targets, s, x_fine)
        y_pts = np.interp(s_targets, s, y_fine)
        
        pts = np.column_stack([x_pts, y_pts])
        pts -= pts.mean(axis=0)
        return pts
        
    except Exception:
        ctrl_centered = ctrl - ctrl.mean(axis=0)
        pts = np.vstack([ctrl_centered, -ctrl_centered[::-1]])
        return pts


def _objective_bspline(ctrl_params):
    pts = _decode_bspline(ctrl_params)
    return _objective_hard(pts.flatten())


def _objective_bspline_soft(ctrl_params, beta=80.0):
    pts = _decode_bspline(ctrl_params)
    return _objective_soft(pts.flatten(), beta)


def _generate_bspline_inits():
    """Generate diverse initial control point configurations for B-spline."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular heptagon control points
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(ctrl.flatten().copy())
    
    # 2. Elliptical control points
    for rx, ry in [(1.2, 0.8), (0.8, 1.2), (1.3, 0.7), (0.7, 1.3), (1.1, 0.9)]:
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ctrl = np.column_stack([rx * np.cos(angles), ry * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
    
    # 3. Perturbed regular heptagon
    for js in [0.05, 0.1, 0.15, 0.2]:
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
        ctrl += rng.normal(0, js, size=(7, 2))
        inits.append(ctrl.flatten().copy())
    
    # 4. Random convex-ish polygons
    for seed in range(15):
        rs = np.random.default_rng(seed * 13 + 7)
        angles = np.sort(rs.uniform(0, 2 * np.pi, 7))
        radii = rs.uniform(0.6, 1.4, 7)
        ctrl = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
    
    # 5. Star-like control points
    for seed in range(5):
        rs = np.random.default_rng(seed * 23 + 11)
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        radii = np.ones(7)
        radii[::2] = rs.uniform(0.8, 1.2)
        radii[1::2] = rs.uniform(0.5, 0.8)
        ctrl = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
    
    # 6. Larger perturbations
    for seed in range(10):
        rs = np.random.default_rng(seed * 31 + 17)
        angles = np.sort(rs.uniform(0, 2 * np.pi, 7))
        radii = rs.uniform(0.4, 1.6, 7)
        ctrl = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
    
    return inits


# ---------- Centrally symmetric polygon parameterization ----------
def _decode_central_symmetric(params):
    """
    params: [r_0, ..., r_6, theta_0, ..., theta_6]
    14 parameters: 7 radii and 7 angles for the first 7 vertices.
    """
    r = params[0:7]
    theta = params[7:14]
    
    r = np.maximum(r, 0.01)
    theta = np.mod(theta, np.pi)
    
    order = np.argsort(theta)
    theta_sorted = theta[order]
    r_sorted = r[order]
    
    x1 = r_sorted * np.cos(theta_sorted)
    y1 = r_sorted * np.sin(theta_sorted)
    
    x2 = -x1[::-1]
    y2 = -y1[::-1]
    
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
    r = np.ones(7)
    theta = np.linspace(0, np.pi, 7, endpoint=False)
    inits.append(np.concatenate([r, theta]).copy())
    
    # 2. Elliptical distortions
    for rx, ry in [(1.1, 0.9), (0.9, 1.1), (1.2, 0.85), (0.85, 1.2), (1.3, 0.77)]:
        r = np.ones(7)
        theta = np.linspace(0, np.pi, 7, endpoint=False)
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


def _coordinate_descent(pts, deadline, max_rounds=30):
    """Cyclic coordinate descent on the hard objective."""
    pts = pts.copy()
    n = len(pts)
    step0 = 0.05
    for rnd in range(max_rounds):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for coord in range(2):
                orig = pts[i, coord]
                step = step0
                for _ in range(25):
                    pts[i, coord] = orig + step
                    s_plus = _score(pts)
                    pts[i, coord] = orig - step
                    s_minus = _score(pts)
                    pts[i, coord] = orig
                    if s_plus > s_minus:
                        pts[i, coord] = orig + step
                    elif s_minus > s_plus:
                        pts[i, coord] = orig - step
                    else:
                        break
                    improved = True
                    step *= 0.5
                    if step < 1e-8:
                        break
        if not improved:
            break
    return pts


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
    
    # Phase 1: B-spline optimization (14D parameter space)
    bspline_inits = _generate_bspline_inits()
    
    for i, p0 in enumerate(bspline_inits):
        if time.time() > deadline - 55:
            break
        res = _nm(p0, deadline, _objective_bspline, max_iter=1500)
        if res is not None and res.fun < best_obj:
            pts = _decode_bspline(res.x)
            update_best(pts)
    
    # Phase 2: Centrally symmetric polygon optimization (14D parameter space)
    cs_inits = _generate_central_symmetric_inits()
    
    for i, p0 in enumerate(cs_inits):
        if time.time() > deadline - 45:
            break
        res = _nm(p0, deadline, _objective_central_symmetric, max_iter=1500)
        if res is not None and res.fun < best_obj:
            pts = _decode_central_symmetric(res.x)
            update_best(pts)
    
    # Phase 3: Ring-based optimization (18D parameter space)
    ring_inits = _make_ring2_inits()
    
    for i, p0 in enumerate(ring_inits):
        if time.time() > deadline - 35:
            break
        res = _nm(p0, deadline, _objective_ring2, max_iter=1200)
        if res is not None and res.fun < best_obj:
            pts = _decode_ring2(res.x)
            update_best(pts)
    
    # Phase 4: Cartesian local refinement from best found
    if best_pts is not None and time.time() < deadline - 30:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=2000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
    
    # Phase 5: L-BFGS-B with soft objective
    if best_pts is not None and time.time() < deadline - 25:
        res = _lbfgs(best_pts.flatten(), deadline, _objective_soft, max_iter=2000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
    
    # Phase 6: Differential Evolution (Global Search)
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
    
    # Phase 7: Coordinate descent refinement
    if best_pts is not None and time.time() < deadline - 10:
        cd_pts = _coordinate_descent(best_pts, deadline, max_rounds=20)
        update_best(cd_pts)
    
    # Phase 8: Final Polish
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
