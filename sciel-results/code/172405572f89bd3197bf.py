import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize, differential_evolution

N = 14

# Precompute triple indices
TRIPLE_INDICES = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            TRIPLE_INDICES.append((i, j, k))
TRIPLE_INDICES = np.array(TRIPLE_INDICES)

def _score_general(points):
    """
    Calculate score for general N points.
    points: (N, 2) array
    """
    if points.shape != (N, 2):
        return 0.0
    
    min_area = float('inf')
    pts = points
    for i, j, k in TRIPLE_INDICES:
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        area = 0.5 * abs(val)
        if area < min_area:
            min_area = area
            
    if min_area == 0.0:
        return 0.0
        
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-9:
        return 0.0
        
    return min_area / hull_area

def _get_points_2p(r, theta):
    """Two concentric heptagons."""
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    x_out = np.cos(angles_outer)
    y_out = np.sin(angles_outer)
    
    angles_inner = angles_outer + theta
    x_in = r * np.cos(angles_inner)
    y_in = r * np.sin(angles_inner)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    return points

def _score_2p(params):
    r, theta = params
    if r <= 0 or r >= 1:
        return 0.0
    points = _get_points_2p(r, theta)
    hull_area = 0.5 * 7 * np.sin(2*np.pi/7)
    
    min_area = float('inf')
    pts = points
    for i, j, k in TRIPLE_INDICES:
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        area = 0.5 * abs(val)
        if area < min_area:
            min_area = area
            
    return min_area / hull_area

def _get_points_paired(radii):
    """
    7 pairs of points at heptagon angles.
    radii: array of shape (14,) - 7 outer + 7 inner radial offsets
    Points 0-6: outer ring at angles 2*pi*k/7, radii radii[0:7]
    Points 7-13: inner ring at angles 2*pi*k/7, radii radii[7:14]
    """
    angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    
    # Outer ring
    x_out = radii[0:7] * np.cos(angles)
    y_out = radii[0:7] * np.sin(angles)
    
    # Inner ring
    x_in = radii[7:14] * np.cos(angles)
    y_in = radii[7:14] * np.sin(angles)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    return points

def _score_paired(params):
    """Score for paired heptagon representation."""
    radii = np.abs(params)
    if np.any(radii <= 0.01) or np.any(radii > 2.0):
        return 0.0
    
    points = _get_points_paired(radii)
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-9:
        return 0.0
        
    min_area = float('inf')
    pts = points
    for i, j, k in TRIPLE_INDICES:
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        area = 0.5 * abs(val)
        if area < min_area:
            min_area = area
            
    return min_area / hull_area

def _coordinate_polish(points, max_iter=500, tol=1e-8):
    """
    Coordinate-level Nelder-Mead polish on all 28 coordinates.
    Returns improved points and score.
    """
    x0 = points.flatten()
    
    def neg_score(x):
        pts = x.reshape(N, 2)
        return -_score_general(pts)
    
    try:
        res = minimize(
            neg_score,
            x0,
            method='Nelder-Mead',
            options={'xatol': tol, 'fatol': tol, 'maxiter': max_iter}
        )
        score = -res.fun
        return res.x.reshape(N, 2), score
    except Exception:
        return points, _score_general(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_points = None
    
    # 1. Optimize 2-ring model (keep as baseline)
    bounds_2p = [(0.1, 0.9), (0, 2*np.pi/7)]
    try:
        res_de = differential_evolution(
            lambda p: -_score_2p(p),
            bounds_2p,
            seed=42,
            maxiter=60,
            popsize=12,
            tol=1e-5
        )
        if res_de.success:
            p2 = res_de.x
            res_nm = minimize(
                lambda p: -_score_2p(p),
                p2,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 150}
            )
            score_2p = -res_nm.fun
            points_2p = _get_points_2p(res_nm.x[0], res_nm.x[1])
            if score_2p > best_score:
                best_score = score_2p
                best_points = points_2p
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_points is not None:
            return best_points
        return np.random.rand(N, 2)
    
    # 2. Optimize paired heptagon (14 radial parameters)
    bounds_paired = [(0.1, 1.5)] * 14
    try:
        res_de_p = differential_evolution(
            lambda p: -_score_paired(p),
            bounds_paired,
            seed=789,
            maxiter=80,
            popsize=15,
            tol=1e-5
        )
        if res_de_p.success:
            p_p = res_de_p.x
            points_p = _get_points_paired(np.abs(p_p))
            score_p = _score_general(points_p)
            if score_p > best_score:
                best_score = score_p
                best_points = points_p
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_points is not None:
            return best_points
        return np.random.rand(N, 2)
    
    # 3. Coordinate-level polish with random restarts to escape local optima
    if best_points is not None:
        # First, polish the current best
        polished, polished_score = _coordinate_polish(best_points, max_iter=400, tol=1e-8)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
    
    if time.time() > deadline:
        if best_points is not None:
            return best_points
        return np.random.rand(N, 2)
    
    # 4. Random perturbations from best solution to explore nearby landscape
    rng = np.random.RandomState(12345)
    num_perturbations = 5
    
    for _ in range(num_perturbations):
        if time.time() > deadline - 30:
            break
        
        # Create perturbation
        noise_scale = 0.1
        perturbed = best_points + rng.randn(N, 2) * noise_scale
        
        # Polish the perturbed solution
        polished_p, polished_p_score = _coordinate_polish(perturbed, max_iter=300, tol=1e-8)
        
        if polished_p_score > best_score:
            best_score = polished_p_score
            best_points = polished_p
    
    if best_points is None:
        return np.random.rand(N, 2)
        
    return best_points
