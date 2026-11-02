import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 13

def _min_triangle_area(points):
    # Vectorized calculation of min triangle area
    n = len(points)
    min_area = float('inf')
    # 13*12*11/6 = 286 triangles. Loop is fast enough and memory safe.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _all_triangle_areas(points):
    """Compute all triangle areas as a vector."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(cross)
                areas.append(area)
    return np.array(areas)

def _soft_min_score(points, beta):
    """
    Soft-min log-sum-exp surrogate for smooth optimization.
    Returns a smooth approximation of min(triangle_areas) / hull_area.
    Higher beta -> sharper approximation of true min.
    """
    areas = _all_triangle_areas(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    
    # Log-sum-exp trick for numerical stability
    min_val = np.min(areas)
    if min_val < 1e-15:
        return 0.0
    
    # Soft min: -(1/beta) * log(sum(exp(-beta * areas)))
    # This approximates min(areas) as beta -> inf
    scaled = -beta * (areas - min_val)  # shift for stability
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = min_val - (1.0/beta) * log_sum
    
    return soft_min / hull_area

def _soft_min_score_grad(points, beta):
    """
    Gradient of soft-min score with respect to point coordinates.
    Uses automatic differentiation via finite differences for simplicity,
    but we'll use analytic gradient for speed.
    """
    n = len(points)
    areas = _all_triangle_areas(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return np.zeros(2*n)
    
    min_val = np.min(areas)
    if min_val < 1e-15:
        return np.zeros(2*n)
    
    # Soft min approximation
    scaled = -beta * (areas - min_val)
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = min_val - (1.0/beta) * log_sum
    
    # Gradient of soft_min w.r.t. each area
    # d(soft_min)/d(area_i) = -exp(-beta*area_i) / sum(exp(-beta*area_j))
    # = -exp(scaled_i - max_scaled) / sum(exp(scaled_j - max_scaled))
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    grad_areas = -exp_scaled / sum_exp
    
    # Now compute gradient w.r.t. point coordinates
    # Each area = 0.5 * |cross(v1, v2)| where v1 = P[j]-P[i], v2 = P[k]-P[i]
    # For positive cross: area = 0.5 * (v1_x * v2_y - v1_y * v2_x)
    # d(area)/d(P[i]) = -0.5 * (v2_y, -v2_x) if cross > 0, else +0.5 * (v2_y, -v2_x)
    # d(area)/d(P[j]) = 0.5 * (v2_y, -v2_x) if cross > 0, else -0.5 * (v2_y, -v2_x)  
    # d(area)/d(P[k]) = -0.5 * (v1_y, v1_x) if cross > 0, else +0.5 * (v1_y, v1_x)
    
    grad = np.zeros(2*n)
    
    # Rebuild triangle indices
    tri_idx = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tri_idx.append((i, j, k))
    
    for t, (i, j, k) in enumerate(tri_idx):
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        sign = 1.0 if cross >= 0 else -1.0
        
        # Gradient of 0.5*sign*cross w.r.t. coordinates
        # d(cross)/d(P[i]_x) = -v2_y, d(cross)/d(P[i]_y) = v2_x
        # d(cross)/d(P[j]_x) = v2_y, d(cross)/d(P[j]_y) = -v2_x
        # d(cross)/d(P[k]_x) = -v1_y, d(cross)/d(P[k]_y) = v1_x
        
        g = 0.5 * sign * grad_areas[t]
        
        grad[2*i] += g * (-v2[1])
        grad[2*i+1] += g * (v2[0])
        grad[2*j] += g * (v2[1])
        grad[2*j+1] += g * (-v2[0])
        grad[2*k] += g * (-v1[1])
        grad[2*k+1] += g * (v1[0])
    
    # Normalize by hull_area (approximate - treat hull_area as constant for gradient)
    return grad / hull_area

def _regular_polygon(n, center=(0.0, 0.0), radius=1.0, angle_offset=0.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + angle_offset
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def _construct_symmetric(r_outer, r_inner, n_outer, n_inner, angle_offset_outer=0.0, angle_offset_inner=0.0):
    """
    Construct points using two concentric regular polygons.
    n_outer + n_inner must equal N.
    """
    pts_outer = _regular_polygon(n_outer, radius=r_outer, angle_offset=angle_offset_outer)
    pts_inner = _regular_polygon(n_inner, radius=r_inner, angle_offset=angle_offset_inner)
    return np.vstack([pts_outer, pts_inner])

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin (limit is 150s)
    
    # 1. Regular Polygon Baseline (all 13 on circle)
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # 2. Soft-min L-BFGS-B optimization with beta annealing
    # Start with lower beta (smoother) and increase for precision
    
    def _soft_min_neg(x, beta):
        if time.time() - start_time > deadline:
            return 0.0
        points = x.reshape(-1, 2)
        return -_soft_min_score(points, beta)
    
    def _soft_min_neg_grad(x, beta):
        if time.time() - start_time > deadline:
            return np.zeros(2*N)
        points = x.reshape(-1, 2)
        grad = _soft_min_score_grad(points, beta)
        return -grad
    
    # Beta annealing: start smooth, get sharper
    betas = [20, 50, 100]
    
    for beta in betas:
        if time.time() - start_time > deadline - 20:
            break
        
        x0 = best_points.flatten()
        try:
            res = minimize(
                _soft_min_neg, 
                x0, 
                method='L-BFGS-B',
                jac=lambda x: _soft_min_neg_grad(x, beta),
                bounds=[(-2.0, 2.0)] * (2*N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            candidate = res.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate
        except Exception:
            pass
    
    # 3. Symmetric Parameter Search for additional exploration
    def _symmetric_objective(params):
        if time.time() - start_time > deadline:
            return 0.0
        r_outer, r_inner, a_outer, a_inner = params
        r_outer = abs(r_outer) + 0.1
        r_inner = abs(r_inner) + 0.01
        
        n_outer, n_inner = 7, 6
        pts = _construct_symmetric(r_outer, r_inner, n_outer, n_inner, a_outer, a_inner)
        return -_score(pts)
    
    # Try split 7-6
    if time.time() - start_time < deadline - 15:
        try:
            x0 = [1.0, 0.5, 0.0, 0.0]
            res = minimize(_symmetric_objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < -best_score:
                r_o, r_i, a_o, a_i = res.x
                r_o = abs(r_o) + 0.1
                r_i = abs(r_i) + 0.01
                candidate = _construct_symmetric(r_o, r_i, 7, 6, a_o, a_i)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
        except Exception:
            pass

    # 4. Final high-beta polish
    if time.time() - start_time < deadline - 5:
        x0 = best_points.flatten()
        try:
            res = minimize(
                _soft_min_neg, 
                x0, 
                method='L-BFGS-B',
                jac=lambda x: _soft_min_neg_grad(x, 200),
                bounds=[(-2.0, 2.0)] * (2*N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            candidate = res.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate
        except Exception:
            pass

    # 5. Local Nelder-Mead refinement from best
    if time.time() - start_time < deadline - 3:
        x0 = best_points.flatten()
        def objective(x):
            if time.time() - start_time > deadline:
                return 0.0
            points = x.reshape(-1, 2)
            return -_score(points)
        try:
            res_nm = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            candidate = res_nm.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate
        except Exception:
            pass

    return best_points
