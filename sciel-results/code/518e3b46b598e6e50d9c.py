import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)

def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _soft_min_logsumexp(areas, beta):
    """
    Log-sum-exp based soft-min: -1/beta * log(sum(exp(-beta * a_i)))
    Approximates min(a_i) as beta -> inf, but is smooth and numerically stable.
    """
    beta = max(beta, 1e-6)
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    if max_val > 500:  # numerical safety
        max_val = 500
    sum_exp = np.sum(np.exp(neg_beta_areas - max_val))
    if sum_exp < 1e-300:
        return 0.0
    soft_min = -(max_val + np.log(sum_exp)) / beta
    return max(soft_min, 1e-15)

def _smooth_objective_cartesian(x_flat, beta=100.0):
    """
    Smooth surrogate using log-sum-exp soft-min in Cartesian coords.
    """
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    # Fix rotation: point 0 on positive x-axis
    p0 = points[0]
    norm_p0 = np.linalg.norm(p0)
    if norm_p0 > 1e-9:
        theta = np.arctan2(p0[1], p0[0])
        c, s = np.cos(-theta), np.sin(-theta)
        points = points @ np.array([[c, -s], [s, c]]).T
    else:
        p1 = points[1]
        norm_p1 = np.linalg.norm(p1)
        if norm_p1 > 1e-9:
            theta = np.arctan2(p1[1], p1[0])
            c, s = np.cos(-theta), np.sin(-theta)
            points = points @ np.array([[c, -s], [s, c]]).T
        else:
            return 1e10

    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
        
    areas = _triangle_areas_vectorized(points)
    sm = _soft_min_logsumexp(areas, beta)
    score = sm / hull_area
    if score <= 0:
        return 1e10
    return -score

def _objective_hard(x_flat):
    """Hard objective: negative of min triangle area / hull area."""
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    p0 = points[0]
    norm_p0 = np.linalg.norm(p0)
    if norm_p0 > 1e-9:
        theta = np.arctan2(p0[1], p0[0])
        c, s = np.cos(-theta), np.sin(-theta)
        points = points @ np.array([[c, -s], [s, c]]).T
    else:
        p1 = points[1]
        norm_p1 = np.linalg.norm(p1)
        if norm_p1 > 1e-9:
            theta = np.arctan2(p1[1], p1[0])
            c, s = np.cos(-theta), np.sin(-theta)
            points = points @ np.array([[c, -s], [s, c]]).T
        else:
            return 1e10

    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _coordinate_descent_polish(points, max_iter=200, tol=1e-10, time_budget=15.0):
    """
    Coordinate descent: move each point slightly in 8 directions to improve score.
    """
    start_t = time.time()
    pts = points.copy()
    current_score = _score(pts)
    if current_score <= 0:
        return pts
    
    step_sizes = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    directions = np.array([[1,0],[-1,0],[0,1],[0,-1],
                           [0.707,0.707],[-0.707,0.707],
                           [0.707,-0.707],[-0.707,-0.707]])
    
    improved = True
    while improved and (time.time() - start_t) < time_budget:
        improved = False
        for idx in range(N):
            if (time.time() - start_t) > time_budget:
                break
            for ss in step_sizes:
                if (time.time() - start_t) > time_budget:
                    break
                for d in directions:
                    new_pts = pts.copy()
                    new_pts[idx] = pts[idx] + ss * d
                    new_score = _score(new_pts)
                    if new_score > current_score + tol:
                        pts = new_pts
                        current_score = new_score
                        improved = True
                        break  # try next step size with this direction
                if improved:
                    break
            if improved:
                break
    return pts

def _make_ring_config(n_outer, n_inner, r_outer, r_inner, angle_offset=0.0):
    """Create a ring configuration: n_outer points on outer ring, n_inner on inner."""
    total = n_outer + n_inner
    pts = np.zeros((total, 2))
    if n_outer > 0:
        a_out = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + angle_offset
        pts[:n_outer, 0] = r_outer * np.cos(a_out)
        pts[:n_outer, 1] = r_outer * np.sin(a_out)
    if n_inner > 0:
        a_in = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + angle_offset + np.pi/n_inner
        pts[n_outer:, 0] = r_inner * np.cos(a_in)
        pts[n_outer:, 1] = r_inner * np.sin(a_in)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin (150s limit)
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two rings: 7 outer, 6 inner
    inits.append(_make_ring_config(7, 6, 1.0, 0.55))
    
    # 3. Two rings: 7 outer, 6 inner (different radii)
    inits.append(_make_ring_config(7, 6, 1.0, 0.45))
    
    # 4. Three rings: 5-4-4
    r1, r2, r3 = 1.0, 0.65, 0.25
    a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    a2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    a3 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    pts_5_4_4 = np.vstack([
        np.column_stack([r1*np.cos(a1), r1*np.sin(a1)]),
        np.column_stack([r2*np.cos(a2), r2*np.sin(a2)]),
        np.column_stack([r3*np.cos(a3), r3*np.sin(a3)])
    ])
    inits.append(pts_5_4_4)
    
    # 5. Three rings: 6-4-3
    r1, r2, r3 = 1.0, 0.6, 0.2
    a1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    a2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    a3 = np.linspace(0, 2*np.pi, 3, endpoint=False)
    pts_6_4_3 = np.vstack([
        np.column_stack([r1*np.cos(a1), r1*np.sin(a1)]),
        np.column_stack([r2*np.cos(a2), r2*np.sin(a2)]),
        np.column_stack([r3*np.cos(a3), r3*np.sin(a3)])
    ])
    inits.append(pts_6_4_3)
    
    # 6. Two rings: 8 outer, 5 inner
    inits.append(_make_ring_config(8, 5, 1.0, 0.5))
    
    # 7. Two rings: 6 outer, 7 inner
    inits.append(_make_ring_config(6, 7, 1.0, 0.55))
    
    # 8-12. Random starts
    rng = np.random.default_rng(42)
    for seed in range(5):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts)
    
    # Optimize each initial guess
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        if remaining < 8.0:
            break
        
        x0 = init_pts.flatten()
        
        # Phase 1: Smooth optimization with log-sum-exp soft-min
        try:
            # Adaptive beta: start moderate, increase for sharper min
            beta = 50.0
            res_smooth = minimize(
                lambda x: _smooth_objective_cartesian(x, beta=beta),
                x0,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            opt_pts = res_smooth.x.reshape(N, 2)
            _update_best(opt_pts)
        except Exception:
            pass
        
        # Phase 2: Hard objective with Nelder-Mead
        if time.time() < deadline - 5.0:
            try:
                start_x = res_smooth.x if 'res_smooth' in dir() else x0
                res_nm = minimize(
                    _objective_hard,
                    start_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                opt_pts_nm = res_nm.x.reshape(N, 2)
                _update_best(opt_pts_nm)
            except Exception:
                pass
        
        # Phase 3: Coordinate descent polish
        if best_points is not None and time.time() < deadline - 3.0:
            try:
                cd_pts = _coordinate_descent_polish(
                    best_points, max_iter=100, tol=1e-10, 
                    time_budget=min(5.0, max(1.0, deadline - time.time() - 2.0))
                )
                _update_best(cd_pts)
            except Exception:
                pass

    # Final polish on the best candidate
    if best_points is not None and time.time() < deadline - 2.0:
        try:
            res_final = minimize(
                _objective_hard,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
            )
            opt_final = res_final.x.reshape(N, 2)
            _update_best(opt_final)
        except Exception:
            pass
        
        # Final coordinate descent
        if time.time() < deadline - 0.5:
            try:
                cd_final = _coordinate_descent_polish(
                    best_points, max_iter=50, tol=1e-11,
                    time_budget=max(0.5, deadline - time.time() - 0.5)
                )
                _update_best(cd_final)
            except Exception:
                pass

    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
