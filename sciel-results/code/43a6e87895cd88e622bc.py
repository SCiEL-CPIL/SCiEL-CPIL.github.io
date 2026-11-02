"""Heilbronn convex regions n=14: multi-start optimization of min-triangle/hull-area.

Key insight: for points on a circle, the minimum-area triangle is always among
three consecutive points (sorted by angle).  The area of triangle (i,i+1,i+2)
with arc gaps d1,d2,d3 (d1+d2+d3=2*pi) is 0.5*sin(d1)*sin(d2)*sin(d3).
This O(n) evaluation enables aggressive multi-start local optimization.

The regular 14-gon equalizes all consecutive gaps at 2*pi/14, but non-regular
configurations may trade hull area (convex hull of non-regular polygon) against
minimum triangle area to achieve a better ratio.

This program combines the multi-start strategy from the current program with
the angle-space optimization and clean reconstruction from the inspiration.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14


def _min_triangle_area_vec(points):
    """Exact minimum triangle area over all C(n,3) triples."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    m = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = 0.5 * abs((x[j] - x[i]) * (y[k] - y[i]) -
                              (y[j] - y[i]) * (x[k] - x[i]))
                if a < m:
                    m = a
    return m


def _hull_area_fast(points):
    try:
        return ConvexHull(points).volume
    except Exception:
        return 0.0


def _score(points):
    """min triangle area / convex hull area."""
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2) or np.any(~np.isfinite(points)):
        return 0.0
    ha = _hull_area_fast(points)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area_vec(points)
    if ma <= 0.0:
        return 0.0
    return ma / ha


def _regular_polygon(k, r=1.0, phase=0.0):
    a = phase + 2 * np.pi * np.arange(k) / k
    return r * np.column_stack([np.cos(a), np.sin(a)])


def _pts_from_angles(theta, r=1.0):
    """Points on unit circle from sorted angles."""
    th = np.sort(theta)
    return r * np.column_stack([np.cos(th), np.sin(th)])


def _fast_ratio(points):
    """Fast ratio: min consecutive-triple area / hull area.
    Valid when points are in convex position (sorted on hull boundary)."""
    n = len(points)
    if n < 3:
        return 0.0
    # Sort by angle around centroid
    cx, cy = points.mean(axis=0)
    ang = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    order = np.argsort(ang)
    p = points[order]
    # Check convexity: all cross products same sign
    cross = []
    for i in range(n):
        dx1 = p[(i+1) % n, 0] - p[i, 0]
        dy1 = p[(i+1) % n, 1] - p[i, 1]
        dx2 = p[(i+2) % n, 0] - p[(i+1) % n, 0]
        dy2 = p[(i+2) % n, 1] - p[(i+1) % n, 1]
        cross.append(dx1 * dy2 - dy1 * dx2)
    cross = np.array(cross)
    if np.any(cross <= 1e-14):
        # Not strictly convex; fall back to exact
        ma = _min_triangle_area_vec(points)
        ha = _hull_area_fast(points)
        if ha < 1e-12:
            return 0.0
        return ma / ha
    # All consecutive triples
    min_area = float('inf')
    for i in range(n):
        i1, i2, i3 = i, (i+1) % n, (i+2) % n
        a = 0.5 * abs((p[i1,0]-p[i2,0])*(p[i3,1]-p[i1,1]) -
                      (p[i1,1]-p[i2,1])*(p[i3,0]-p[i1,0]))
        if a < min_area:
            min_area = a
    ha = _hull_area_fast(points)
    if ha < 1e-12:
        return 0.0
    return min_area / ha


def _objective_from_flat(flat):
    """Objective for minimization: minimize -ratio (maximize ratio)."""
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    ratio = _fast_ratio(pts)
    if ratio < 1e-12:
        return 1e10
    return -ratio


def _min_tris_angles(theta):
    """Compute triangle areas for all consecutive triples from angles.
    Returns array of areas and the ratio min/max."""
    n = len(theta)
    th = np.sort(theta)
    # For each triple (i, i+1, i+2), area = 0.5*sin(d1)*sin(d2)*sin(d1+d2)
    # where d1 = th[i+1]-th[i], d2 = th[i+2]-th[i+1]
    # Actually, for points on unit circle, area of triangle with angles a,b,c
    # is 0.5*|sin(b-a)*sin(c-b)*sin(c-a)|... let me use the cross product formula
    areas = np.zeros(n)
    for i in range(n):
        a1 = th[i]
        a2 = th[(i+1) % n]
        a3 = th[(i+2) % n]
        # Handle wrap-around
        if i + 2 >= n:
            a2 = th[(i+1) % n] + 2*np.pi * ((i+1) >= n)
            a3 = th[(i+2) % n] + 2*np.pi * ((i+2) >= n)
        # Area = 0.5 * |sin(a2-a1) * sin(a3-a2) * sin(a3-a1)|... 
        # Actually simpler: use the formula for triangle on unit circle
        # Area = 0.5 * |sin(a2-a1) + sin(a3-a2) + sin(a1-a3)|
        # Let me just compute directly
        p1 = np.array([np.cos(a1), np.sin(a1)])
        p2 = np.array([np.cos(a2), np.sin(a2)])
        p3 = np.array([np.cos(a3), np.sin(a3)])
        areas[i] = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
    t = areas
    if t.size == 0:
        return t, 0.0
    t2min = np.min(t)
    t2max = np.max(t)
    if t2max > 1e-10:
        ratio = t2min / t2max
    else:
        ratio = 0.0
    return t, ratio


def _angle_objective(theta_flat, beta=2.0):
    """Objective in angle space: minimize -(min_area / max_area)^beta
    This encourages equalizing consecutive triple areas."""
    th = np.asarray(theta_flat, dtype=float)
    if th.shape != (N,) or np.any(~np.isfinite(th)):
        return 1e10
    t, ratio = _min_tris_angles(th)
    if t.size == 0 or np.min(t) < 1e-8:
        return 1e10
    # We want to maximize the ratio, so minimize -ratio^beta
    return -(ratio ** beta)


def _nm_optimize(x0, deadline, maxiter=2000, obj=None):
    """Nelder-Mead with time guard."""
    if time.time() > deadline:
        return None
    if obj is None:
        obj = _objective_from_flat
    try:
        result = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-10,
                                   'fatol': 1e-12, 'adaptive': True,
                                   'maxfev': maxiter * N})
        if time.time() > deadline:
            return None
        return result
    except Exception:
        return None


def _powell_optimize(x0, deadline, maxiter=300, obj=None):
    """Powell with time guard."""
    if time.time() > deadline:
        return None
    if obj is None:
        obj = _objective_from_flat
    try:
        result = minimize(obj, x0, method='Powell',
                          options={'maxiter': maxiter, 'xtol': 1e-12,
                                   'ftol': 1e-14})
        if time.time() > deadline:
            return None
        return result
    except Exception:
        return None


def _consider(pts, best):
    """Update best if pts gives a better score."""
    if pts is None or pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
        return
    s = _score(pts)
    if s > best['score']:
        best['score'] = s
        best['pts'] = pts.copy()


def _normalize_unit_area(points):
    """Scale so convex hull area = 1."""
    try:
        ha = ConvexHull(points).volume
        if ha > 1e-10:
            return points * np.sqrt(1.0 / ha)
    except Exception:
        pass
    xs, ys = points[:, 0], points[:, 1]
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        return points / np.sqrt(w * h)
    return points


def _generate_starts(deadline):
    """Generate diverse starting configurations."""
    starts = []
    t0 = time.time()
    
    # 1. Regular 14-gon
    starts.append(_regular_polygon(N).flatten())
    
    # 2. Regular 14-gon with various phases
    for phase in [np.pi/14, np.pi/7, 3*np.pi/14]:
        if time.time() > deadline - 30:
            break
        starts.append(_regular_polygon(N, phase=phase).flatten())
    
    # 3. Perturbed regular polygons (random small perturbations)
    rng = np.random.default_rng(42)
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        base = _regular_polygon(N)
        # Perturb angles
        dtheta = rng.normal(0, 0.1, N)
        angles = np.arctan2(base[:, 1], base[:, 0]) + dtheta
        angles = np.sort(angles)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        starts.append(pts.flatten())
    
    # 4. Points on a square boundary (evenly spaced)
    if time.time() < deadline - 30:
        sq = []
        side_len = N // 4
        remaining = N - 4 * side_len
        # 4 corners + distribute along edges
        for i in range(N):
            frac = i / N
            if frac < 0.25:
                t = frac / 0.25
                sq.append([t, 1.0])
            elif frac < 0.5:
                t = (frac - 0.25) / 0.25
                sq.append([1.0, 1.0 - t])
            elif frac < 0.75:
                t = (frac - 0.5) / 0.25
                sq.append([1.0 - t, 0.0])
            else:
                t = (frac - 0.75) / 0.25
                sq.append([0.0, t])
        sq = np.array(sq)
        sq = (sq - 0.5)  # center
        starts.append(sq.flatten())
    
    # 5. Random points in unit disk (then project to convex hull positions)
    for trial in range(10):
        if time.time() > deadline - 30:
            break
        rng = np.random.default_rng(100 + trial)
        # Generate random angles, sort, use as circle points
        angles = np.sort(rng.uniform(0, 2*np.pi, N))
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        starts.append(pts.flatten())
    
    # 6. Elliptical configurations
    for a_ratio in [0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5]:
        if time.time() > deadline - 30:
            break
        base = _regular_polygon(N)
        base[:, 0] *= a_ratio
        starts.append(base.flatten())
    
    return starts


def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin before 150s kill
    best = {'score': 0.0, 'pts': None}
    
    # Evaluate all starts
    starts = _generate_starts(deadline)
    for s in starts:
        if time.time() > deadline - 20:
            break
        pts = s.reshape(N, 2)
        _consider(pts, best)
    
    # If we have a decent best, try to polish it
    if best['score'] > 1e-6:
        x0 = best['pts'].flatten()
        
        # Nelder-Mead polish in position space
        if time.time() < deadline - 15:
            res = _nm_optimize(x0.copy(), deadline - 10, maxiter=1500)
            if res is not None and np.isfinite(res.fun):
                pts = res.x.reshape(N, 2)
                _consider(pts, best)
        
        # Powell polish in position space
        if time.time() < deadline - 10:
            x0 = best['pts'].flatten()
            res = _powell_optimize(x0.copy(), deadline - 5, maxiter=200)
            if res is not None and np.isfinite(res.fun):
                pts = res.x.reshape(N, 2)
                _consider(pts, best)
    
    # Try angle-space optimization to equalize consecutive triple areas
    if best['pts'] is not None and time.time() < deadline - 25:
        # Extract angles from best configuration
        pts = best['pts']
        cx, cy = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        angles = np.sort(angles)
        
        # Optimize in angle space
        if time.time() < deadline - 20:
            res = _nm_optimize(angles.copy(), deadline - 15, maxiter=1000,
                              obj=_angle_objective)
            if res is not None and np.isfinite(res.fun):
                pts_angle = _pts_from_angles(res.x)
                _consider(pts_angle, best)
        
        # Powell in angle space
        if time.time() < deadline - 15:
            res = _powell_optimize(angles.copy(), deadline - 10, maxiter=300,
                                  obj=_angle_objective)
            if res is not None and np.isfinite(res.fun):
                pts_angle = _pts_from_angles(res.x)
                _consider(pts_angle, best)
    
    # Try a few more diverse local optimizations from perturbed best
    if best['pts'] is not None and time.time() < deadline - 15:
        rng = np.random.default_rng(200)
        base_pts = best['pts']
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            # Small random perturbation
            pert = base_pts + rng.normal(0, 0.05, (N, 2))
            _consider(pert, best)
            
            if time.time() < deadline - 10:
                res = _nm_optimize(pert.flatten(), deadline - 8, maxiter=800)
                if res is not None and np.isfinite(res.fun):
                    pts = res.x.reshape(N, 2)
                    _consider(pts, best)
    
    # Fallback
    if best['pts'] is None or best['score'] < 1e-6:
        best['pts'] = _regular_polygon(N)
        best['score'] = _score(best['pts'])
    
    return _normalize_unit_area(best['pts'])
