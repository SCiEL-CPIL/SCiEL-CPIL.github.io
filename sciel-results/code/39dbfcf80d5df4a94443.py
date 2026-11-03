"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                  (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def negative_min_area_flat(x):
    points = x.reshape(-1, 2)
    penalty = 0.0
    for p in points:
        if not point_in_triangle(p):
            penalty += 1.0
    return -min_triangle_area(points) + penalty

def project_to_triangle(p):
    """Project point p onto the triangle (0,0),(1,0),(0.5,sqrt(3)/2) if outside."""
    if point_in_triangle(p):
        return p
    
    px, py = p
    c = 2 * py / np.sqrt(3)
    b = px - 0.5 * c
    a = 1 - b - c
    
    a = max(0, a)
    b = max(0, b)
    c = max(0, c)
    
    s = a + b + c
    if s < 1e-15:
        return V0.copy()
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def compute_grad(points):
    """Compute gradient of the minimum triangle area w.r.t. point coordinates."""
    n = len(points)
    min_area = float('inf')
    min_triple = None
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                  (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
                    min_triple = (i, j, k)
    
    grad = np.zeros((n, 2))
    if min_triple is None:
        return grad
    
    i, j, k = min_triple
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    # Cross product: (pj - pi) x (pk - pi)
    # Area = 0.5 * |cross|
    # d(area)/d(pi) = -0.5 * (pk - pj) rotated 90 deg
    # d(area)/d(pj) = 0.5 * (pi - pk) rotated 90 deg
    # d(area)/d(pk) = 0.5 * (pj - pi) rotated 90 deg
    
    # For cross product (a x b) where a = pj - pi, b = pk - pi:
    # d/dpi (a x b) = -(pk - pj) rotated 90 = (pk[1]-pj[1], -(pk[0]-pj[0]))
    # d/dpj (a x b) = (pi - pk) rotated 90 = (pk[1]-pi[1], -(pk[0]-pi[0]))... wait
    # Let's be more careful.
    # cross = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pk[0]-pi[0])*(pj[1]-pi[1])
    # d(cross)/d(pi[0]) = -(pk[1]-pi[1]) - (pk[0]-pi[0])*0 = -(pk[1]-pi[1]) + ... 
    # Actually: d/dpi[0]: -(pk[1]-pi[1]) - 0 = pi[1] - pk[1]
    # d/dpi[1]: -(pj[0]-pi[0]) - 0 = pi[0] - pj[0]
    # d/dpj[0]: 0 - (pk[0]-pi[0]) = pi[0] - pk[0]
    # d/dpj[1]: (pk[1]-pi[1]) - 0 = pk[1] - pi[1]
    # d/dpk[0]: 0 - (pj[1]-pi[1]) = pi[1] - pj[1]
    # d/dpk[1]: (pj[0]-pi[0]) - 0 = pj[0] - pi[0]
    
    # So gradient of cross product w.r.t. each point:
    # grad_pi = (pi[1] - pk[1], pi[0] - pj[0])
    # grad_pj = (pi[0] - pk[0], pk[1] - pi[1])
    # grad_pk = (pi[1] - pj[1], pj[0] - pi[0])
    
    # Since area = 0.5 * |cross|, and assuming cross > 0 (we can flip signs if needed)
    # The sign of the cross product determines the direction.
    cross_val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1])
    sign_cross = np.sign(cross_val)
    
    # Gradient of area = 0.5 * sign(cross) * grad(cross)
    # But we want to maximize min_area, so we want to move in direction of increasing area for the min triangle.
    # The gradient of the min area function is the gradient of the active constraint.
    
    g_pi = 0.5 * sign_cross * np.array([pi[1] - pk[1], pi[0] - pj[0]])
    g_pj = 0.5 * sign_cross * np.array([pi[0] - pk[0], pk[1] - pi[1]])
    g_pk = 0.5 * sign_cross * np.array([pi[1] - pj[1], pj[0] - pi[0]])
    
    grad[i] += g_pi
    grad[j] += g_pj
    grad[k] += g_pk
    
    return grad

def negative_min_area_flat_grad(x):
    points = x.reshape(-1, 2)
    penalty = 0.0
    for p in points:
        if not point_in_triangle(p):
            penalty += 1.0
    val = -min_triangle_area(points) + penalty
    grad = -compute_grad(points)
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    def initial_points(seed_offset=0):
        """Generate initial points using a structured pattern."""
        rng_local = np.random.RandomState(42 + seed_offset)
        pts = []
        
        # Place points near vertices
        pts.append(V0 + 0.05 * np.array([0.5, np.sqrt(3)/6]))
        pts.append(V1 + 0.05 * np.array([-0.5, np.sqrt(3)/6]))
        pts.append(V2 + 0.05 * np.array([0, -np.sqrt(3)/3]))
        
        # Add more points in a pattern
        for i in range(8):
            u = rng_local.random() * 0.85 + 0.05
            v = rng_local.random() * (1 - u) * 0.85 + 0.05
            p = u * V1 + v * V2
            pts.append(p)
        return np.array(pts)
    
    best_points = None
    best_score = -float('inf')
    
    # Try multiple initial configurations
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
        
        points = initial_points(restart)
        x0 = points.flatten()
        
        # First try L-BFGS-B with gradient information
        try:
            bounds = [(0, 1), (0, np.sqrt(3)/2)] * 11
            result = minimize(
                negative_min_area_flat_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if time.time() > deadline - 15.0:
                break
            
            optimized_points = result.x.reshape(-1, 2)
            
            # Project any points outside the triangle back inside
            for i in range(len(optimized_points)):
                if not point_in_triangle(optimized_points[i]):
                    optimized_points[i] = project_to_triangle(optimized_points[i])
            
            valid = all(point_in_triangle(p) for p in optimized_points)
            
            if valid:
                score = min_triangle_area(optimized_points)
                if score > best_score:
                    best_score = score
                    best_points = optimized_points
                    
        except Exception:
            pass
        
        # If we have time, also try Nelder-Mead from this starting point
        if time.time() > deadline - 10.0:
            break
        
        try:
            result_nm = minimize(
                negative_min_area_flat,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            optimized_points_nm = result_nm.x.reshape(-1, 2)
            
            for i in range(len(optimized_points_nm)):
                if not point_in_triangle(optimized_points_nm[i]):
                    optimized_points_nm[i] = project_to_triangle(optimized_points_nm[i])
            
            valid_nm = all(point_in_triangle(p) for p in optimized_points_nm)
            
            if valid_nm:
                score_nm = min_triangle_area(optimized_points_nm)
                if score_nm > best_score:
                    best_score = score_nm
                    best_points = optimized_points_nm
                    
        except Exception:
            pass
    
    if best_points is None:
        best_points = initial_points(0)
    
    return best_points
# EVOLVE-BLOCK-END
