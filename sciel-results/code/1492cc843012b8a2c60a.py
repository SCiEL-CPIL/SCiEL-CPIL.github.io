import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
        
    # Compute all triangle areas
    n = len(points)
    min_area = float('inf')
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dx2 * dy1)
                if area < min_area:
                    min_area = area
                    if min_area < 1e-12:
                        return 0.0
                        
    return min_area / hull_area

def _cyclic_polygon_points(angles):
    """
    Generate 14 points as a cyclic polygon (on unit circle).
    angles: array of 14 positive values summing to 2*pi, representing arc lengths.
    Returns points on the unit circle.
    """
    # Normalize angles to sum to 2*pi
    total = np.sum(angles)
    if total < 1e-12:
        angles = np.ones(N) * 2 * np.pi / N
    else:
        angles = angles * (2 * np.pi / total)
    
    # Cumulative angles give vertex positions on circle
    theta = np.cumsum(angles)
    theta[0] = 0.0  # Start at angle 0
    
    # Actually, we want vertices at cumulative positions
    # Vertex i is at angle sum(angles[0:i])
    vertex_angles = np.zeros(N)
    for i in range(1, N):
        vertex_angles[i] = np.sum(angles[:i])
    
    points = np.column_stack((np.cos(vertex_angles), np.sin(vertex_angles)))
    return points

def _neg_cyclic_score(angles):
    """
    Objective for cyclic polygon optimization.
    angles: 14 positive values (will be normalized to sum to 2*pi).
    """
    if np.any(angles < 0):
        return 1e9
    points = _cyclic_polygon_points(angles)
    score = compute_score(points)
    return -score

def _two_ring_points(r, theta):
    """
    Generate 14 points: 7 on outer ring (radius 1.0), 7 on inner ring (radius r).
    Inner ring is rotated by theta.
    """
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = angles_outer + theta
    
    p_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    p_inner = np.column_stack((r * np.cos(angles_inner), r * np.sin(angles_inner)))
    
    points = np.vstack((p_outer, p_inner))
    return points

def _neg_ring_score(params):
    """
    Objective function for the 2-ring parameterization.
    params: [r, theta]
    """
    r, theta = params
    if r < 0.0 or r > 1.0:
        return 1e9
    points = _two_ring_points(r, theta)
    score = compute_score(points)
    return -score

def _three_ring_points(params):
    """
    Generate 14 points using 3 rings: 7 outer, 4 middle, 3 inner.
    params: [r_mid, r_inner, theta_mid, theta_inner]
    """
    r_mid, r_inner, theta_mid, theta_inner = params
    
    # Outer ring: 7 points at radius 1.0
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    
    # Middle ring: 4 points at radius r_mid
    angles_mid = np.linspace(0, 2 * np.pi, 4, endpoint=False) + theta_mid
    p_mid = np.column_stack((r_mid * np.cos(angles_mid), r_mid * np.sin(angles_mid)))
    
    # Inner ring: 3 points at radius r_inner
    angles_inner = np.linspace(0, 2 * np.pi, 3, endpoint=False) + theta_inner
    p_inner = np.column_stack((r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)))
    
    points = np.vstack((p_outer, p_mid, p_inner))
    return points

def _neg_three_ring_score(params):
    """
    Objective function for the 3-ring parameterization.
    params: [r_mid, r_inner, theta_mid, theta_inner]
    """
    r_mid, r_inner, theta_mid, theta_inner = params
    
    if r_mid < 0.0 or r_mid > 1.0 or r_inner < 0.0 or r_inner > 1.0:
        return 1e9
    
    points = _three_ring_points(params)
    score = compute_score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = 140.0

    best_points = None
    best_score = -float('inf')

    # 1. NEW: Cyclic polygon optimization (structural change)
    # Optimize 14 arc lengths (positive) that sum to 2*pi
    # Use differential evolution in the 14-dim space of positive arc lengths
    try:
        # Initial guess: equal spacing
        equal_arcs = np.ones(N) * 2 * np.pi / N
        
        bounds_cyclic = [(0.01, 2 * np.pi - 0.01)] * N
        
        res_cyclic = differential_evolution(
            _neg_cyclic_score,
            bounds_cyclic,
            seed=42,
            maxiter=200,
            tol=1e-8,
            polish=True,
            init='sobol',
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.9
        )
        
        if res_cyclic.fun < 0:
            points_cyclic = _cyclic_polygon_points(res_cyclic.x)
            score_cyclic = compute_score(points_cyclic)
            
            if score_cyclic > best_score:
                best_score = score_cyclic
                best_points = points_cyclic.copy()
    except Exception:
        pass

    # 2. Local refinement of cyclic solution
    if best_points is not None and time.time() - start_time < deadline - 30:
        x0 = best_points.flatten()
        
        try:
            res_nm = minimize(_neg_cyclic_score, res_cyclic.x, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res_nm.fun < res_cyclic.fun:
                points_refined = _cyclic_polygon_points(res_nm.x)
                score_refined = compute_score(points_refined)
                if score_refined > best_score:
                    best_score = score_refined
                    best_points = points_refined.copy()
        except Exception:
            pass

    # 3. Also search 3-ring family as backup
    if time.time() - start_time < deadline - 30:
        bounds_3ring = [(0.0, 1.0), (0.0, 1.0), (0.0, np.pi/2), (0.0, 2*np.pi/3)]
        
        try:
            res_de_3 = differential_evolution(
                _neg_three_ring_score, 
                bounds_3ring, 
                seed=42, 
                maxiter=100, 
                tol=1e-8,
                polish=True,
                init='sobol',
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.9
            )
            
            if res_de_3.success or res_de_3.fun < 0:
                points_3 = _three_ring_points(res_de_3.x)
                score_3 = compute_score(points_3)
                
                if score_3 > best_score:
                    best_score = score_3
                    best_points = points_3.copy()
        except Exception:
            pass

    # 4. 2-ring family
    if time.time() - start_time < deadline - 20:
        bounds_2ring = [(0.0, 1.0), (0.0, 2 * np.pi / 7)]
        try:
            res_de_2 = differential_evolution(
                _neg_ring_score, 
                bounds_2ring, 
                seed=42, 
                maxiter=80, 
                tol=1e-8,
                polish=True,
                init='sobol',
                popsize=15
            )
            
            if res_de_2.success or res_de_2.fun < 0:
                points_2 = _two_ring_points(res_de_2.x[0], res_de_2.x[1])
                score_2 = compute_score(points_2)
                
                if score_2 > best_score:
                    best_score = score_2
                    best_points = points_2.copy()
        except Exception:
            pass

    # Fallback: coarse grid
    if best_points is None:
        r_vals = np.linspace(0.2, 0.9, 6)
        th_vals = np.linspace(0, 2*np.pi/7, 6)
        for r in r_vals:
            for th in th_vals:
                points = _two_ring_points(r, th)
                s = compute_score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()

    # 5. Final local refinement in full 28D space
    if best_points is not None and time.time() - start_time < deadline - 10:
        x0 = best_points.flatten()
        
        def _neg_score_full(x):
            pts = x.reshape((N, 2))
            return -compute_score(pts)
        
        try:
            res_final = minimize(_neg_score_full, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            new_points = res_final.x.reshape((N, 2))
            new_score = compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        except Exception:
            pass

    return best_points
