"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Structural approach:
  1. Closed cubic B-spline with 7 control points (14D) defines a smooth convex
     boundary. The 14 points are sampled at equal arc-length intervals.
  2. Soft minimum triangle area objective (log-sum-exp) for smooth optimization.
  3. Multi-phase optimization: differential evolution for global search, then
     Nelder-Mead / Powell refinement on B-spline control points, then full 2D
     Cartesian refinement.
  4. Vectorized triangle area computation using pre-computed index triples.
  5. Time-guarded to finish well within 150s.
"""
import time
import numpy as np
import itertools
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline

N = 14

# Pre-compute all triangle index combinations (364 triples)
tri_indices = np.array(list(itertools.combinations(range(N), 3)))


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
    """Soft objective: hull_area / soft_min_triangle_area."""
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
    Decode 14-dim control point parameters to 14 points on a closed cubic B-spline.
    ctrl_params: shape (14,) = [x0,y0, ..., x6,y6]
    Points are sampled at equal arc-length intervals along the curve.
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
    """Generate diverse initial control point configurations."""
    inits = []
    rng = np.random.default_rng(42)
    
    # Regular heptagon
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(ctrl.flatten().copy())
    
    # Elliptical
    for rx, ry in [(1.2, 0.8), (0.8, 1.2), (1.3, 0.7), (0.7, 1.3), (1.1, 0.9)]:
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ctrl = np.column_stack([rx * np.cos(angles), ry * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
    
    # Perturbed regular heptagon
    for js in [0.05, 0.1, 0.15, 0.2]:
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
        ctrl += rng.normal(0, js, size=(7, 2))
        inits.append(ctrl.flatten().copy())
    
    # Random convex-ish polygons
    for seed in range(15):
        rs = np.random.default_rng(seed * 13 + 7)
        angles = np.sort(rs.uniform(0, 2 * np.pi, 7))
        radii = rs.uniform(0.6, 1.4, 7)
        ctrl = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
    
    # Star-like
    for seed in range(5):
        rs = np.random.default_rng(seed * 23 + 11)
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        radii = np.ones(7)
        radii[::2] = rs.uniform(0.8, 1.2)
        radii[1::2] = rs.uniform(0.5, 0.8)
        ctrl = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
    
    # Larger perturbations
    for seed in range(10):
        rs = np.random.default_rng(seed * 31 + 17)
        angles = np.sort(rs.uniform(0, 2 * np.pi, 7))
        radii = rs.uniform(0.4, 1.6, 7)
        ctrl = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
    
    return inits


def _normalize_unit_area(points):
    """Normalize points to unit convex hull area."""
    try:
        hull = ConvexHull(points)
        area = hull.volume
        if area > 1e-10:
            scale = np.sqrt(1.0 / area)
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
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    best_score_val = 0.0
    best_pts = None
    
    def _consider(pts):
        nonlocal best_score_val, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score_val:
            best_score_val = s
            best_pts = pts.copy()
    
    # ---- Phase 1: Differential evolution on B-spline control points ----
    if time.time() < deadline - 80:
        try:
            bounds = [(-3, 3)] * 14
            de_res = differential_evolution(
                _objective_bspline_soft, bounds,
                seed=42, maxiter=30, popsize=15,
                tol=1e-6, mutation=(0.3, 1.0), recombination=0.9,
                polish=True, init='sobol',
                updating='deferred'
            )
            pts_de = _decode_bspline(de_res.x)
            _consider(pts_de)
        except Exception:
            pass
    
    # ---- Phase 2: Multi-start Nelder-Mead on B-spline control points ----
    inits = _generate_bspline_inits()
    nm_budget = max(0, deadline - time.time() - 30)
    per_init_time = nm_budget / max(len(inits), 1)
    
    for idx, init in enumerate(inits):
        if time.time() > deadline - 25:
            break
        try:
            res_nm = minimize(
                _objective_bspline_soft, init,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            pts_nm = _decode_bspline(res_nm.x)
            _consider(pts_nm)
        except Exception:
            pass
    
    # ---- Phase 3: Powell refinement on B-spline from best ----
    if best_pts is not None and time.time() < deadline - 15:
        # Invert: find best control params by trying to match best_pts
        # Simpler: just do Powell on the best B-spline init we had
        try:
            # Use the regular heptagon as a safe start for Powell
            angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            ctrl_start = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
            res_pw = minimize(
                _objective_bspline, ctrl_start,
                method='Powell',
                options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            pts_pw = _decode_bspline(res_pw.x)
            _consider(pts_pw)
        except Exception:
            pass
    
    # ---- Phase 4: Full 2D Cartesian refinement ----
    if best_pts is not None and time.time() < deadline - 10:
        pts_flat = best_pts.flatten().copy()
        
        def _obj_2d(flat):
            if time.time() > deadline - 3:
                return 1e10
            pts = flat.reshape(N, 2)
            if np.any(~np.isfinite(pts)):
                return 1e10
            s = _score(pts)
            if s <= 0:
                return 1e10
            return -s
        
        # Nelder-Mead on full 2D
        if time.time() < deadline - 5:
            try:
                res_2d = minimize(
                    _obj_2d, pts_flat,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                pts_2d = res_2d.x.reshape(N, 2)
                _consider(pts_2d)
            except Exception:
                pass
        
        # Powell on full 2D
        if time.time() < deadline - 3:
            try:
                res_pw2d = minimize(
                    _obj_2d, res_2d.x,
                    method='Powell',
                    options={'maxiter': 300, 'ftol': 1e-12, 'xtol': 1e-12}
                )
                pts_pw2d = res_pw2d.x.reshape(N, 2)
                _consider(pts_pw2d)
            except Exception:
                pass
    
    # ---- Phase 5: Coordinate descent polish ----
    if best_pts is not None and time.time() < deadline - 2:
        pts_cd = best_pts.copy()
        for _ in range(8):
            if time.time() > deadline - 1.5:
                break
            improved = False
            for i in range(N):
                for dim in range(2):
                    def obj_1d(val, pts_tmp=pts_cd, ii=i, dd=dim):
                        pts_t = pts_tmp.copy()
                        pts_t[ii, dd] = val
                        s = _score(pts_t)
                        return -s if s > 0 else 1e10
                    
                    try:
                        res_cd = minimize(obj_1d, pts_cd[i, dim], method='Brent',
                                          bounds=(-5.0, 5.0), options={'xtol': 1e-12})
                        if res_cd.fun < obj_1d(pts_cd[i, dim]):
                            pts_cd[i, dim] = res_cd.x
                            improved = True
                    except Exception:
                        pass
            
            _consider(pts_cd)
            if not improved:
                break
    
    # Normalize to unit area
    if best_pts is not None:
        best_pts = _normalize_unit_area(best_pts)
    
    # Fallback: regular 14-gon
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_pts = _normalize_unit_area(best_pts)
    
    return best_pts
