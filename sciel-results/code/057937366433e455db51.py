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
        # Ensure at least 3 non-collinear points
        if len(points) < 3:
            return 0.0
        # Check for degeneracy
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

def _objective(x_flat):
    """
    Objective function for optimization.
    1. Reshape to N x 2
    2. Center (remove translation)
    3. Rotate (remove rotation)
    4. Compute score and return negative
    """
    points = x_flat.reshape(N, 2)
    
    # Center
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Check if points are all the same (degenerate)
    if np.allclose(points, 0):
        return 1e10
    
    # Rotate such that point 0 is on the positive x-axis
    # This removes rotational symmetry
    p0 = points[0]
    norm_p0 = np.linalg.norm(p0)
    if norm_p0 > 1e-9:
        # Angle of p0
        theta = np.arctan2(p0[1], p0[0])
        # Rotation matrix to rotate by -theta
        c = np.cos(-theta)
        s = np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        points = points @ R.T
    else:
        # If p0 is at center, use point 1 to fix rotation
        p1 = points[1]
        norm_p1 = np.linalg.norm(p1)
        if norm_p1 > 1e-9:
            theta = np.arctan2(p1[1], p1[0])
            c = np.cos(-theta)
            s = np.sin(-theta)
            R = np.array([[c, -s], [s, c]])
            points = points @ R.T
        else:
            # Degenerate case, points are extremely clustered
            return 1e10

    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _smooth_objective(x_flat, p=20):
    """
    Smooth surrogate objective using a power mean (soft-min) of triangle areas.
    Minimizing the negative of the soft-min of areas divided by hull area.
    """
    points = x_flat.reshape(N, 2)
    
    # Center
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Check if points are all the same (degenerate)
    if np.allclose(points, 0):
        return 1e10
    
    # Rotate such that point 0 is on the positive x-axis
    p0 = points[0]
    norm_p0 = np.linalg.norm(p0)
    if norm_p0 > 1e-9:
        theta = np.arctan2(p0[1], p0[0])
        c = np.cos(-theta)
        s = np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        points = points @ R.T
    else:
        p1 = points[1]
        norm_p1 = np.linalg.norm(p1)
        if norm_p1 > 1e-9:
            theta = np.arctan2(p1[1], p1[0])
            c = np.cos(-theta)
            s = np.sin(-theta)
            R = np.array([[c, -s], [s, c]])
            points = points @ R.T
        else:
            return 1e10

    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
        
    areas = _triangle_areas_vectorized(points)
    
    # Soft min using power mean: (1/n * sum(a_i^{-p}))^{-1/p}
    # This approximates min(a_i) as p -> infinity
    # Using negative power to get min behavior
    # a_i > 0
    inv_areas = 1.0 / (areas + 1e-12)
    soft_min = (np.mean(inv_areas ** p)) ** (-1.0 / p)
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Initial guesses
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(reg_pts.flatten())
    
    # 2. Two rings: 7 outer, 6 inner
    r_out, r_in = 1.0, 0.5
    ang_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ang_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6 # offset
    pts_7_6 = np.vstack([
        np.column_stack([r_out * np.cos(ang_out), r_out * np.sin(ang_out)]),
        np.column_stack([r_in * np.cos(ang_in), r_in * np.sin(ang_in)])
    ])
    inits.append(pts_7_6.flatten())
    
    # 3. Three rings: 5-4-4
    r1, r2, r3 = 1.0, 0.6, 0.2
    a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    pts_5_4_4 = np.vstack([
        np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
    ])
    inits.append(pts_5_4_4.flatten())
    
    # 4. Random starts
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Random points in unit disk
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
        
    # Optimization Loop
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # Phase 1: Smooth optimization to find a good basin
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=20), 
                x0, 
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            opt_pts = res.x.reshape(N, 2)
            _update_best(opt_pts)
        except Exception:
            pass
            
        # Phase 2: Hard objective polish with Nelder-Mead
        if best_score > 0 and time.time() < deadline - 5:
            try:
                # Use the result from smooth phase as start for hard phase
                # Or restart from best known if smooth failed
                start_x = res.x if 'res' in locals() and res.success else x0
                res_nm = minimize(
                    _objective,
                    start_x,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                opt_pts_nm = res_nm.x.reshape(N, 2)
                _update_best(opt_pts_nm)
            except Exception:
                pass

    # Final high-precision polish on the best candidate found
    if best_points is not None and time.time() < deadline - 2:
        try:
            # Flatten and center/rotate best points to a consistent state for optimization
            # Actually, _objective handles centering/rotation internally, so we can just pass best_points.flatten()
            res_final = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            opt_pts_final = res_final.x.reshape(N, 2)
            _update_best(opt_pts_final)
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
