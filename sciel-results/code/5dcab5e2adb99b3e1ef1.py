import numpy as np
import scipy.optimize
import time
from itertools import combinations

N = 14

def _get_points_from_gaps(gaps):
    """
    Convert angle gaps to points on the unit circle.
    gaps: array of shape (N-1,) representing the angle between consecutive points.
    The last gap is implicitly 2*pi - sum(gaps).
    We ensure all gaps are positive by using exp or similar, or by bounding.
    For optimization, it's better to work with unconstrained variables.
    Let gaps = exp(raw_gaps). Then normalize so sum = 2*pi.
    """
    raw_gaps = np.abs(gaps) + 1e-6 # Ensure positive
    total = np.sum(raw_gaps)
    if total == 0:
        return None
    normalized_gaps = raw_gaps / total * 2 * np.pi
    
    angles = np.zeros(N)
    angles[1:] = np.cumsum(normalized_gaps)
    # angles[0] is 0.
    # The last point is at angles[N-1]. The gap from last to first (wrapping around) is 2*pi - angles[N-1].
    # This is automatically handled by the geometry on the circle.
    
    points = np.vstack([np.cos(angles), np.sin(angles)]).T
    return points

def _calculate_score(pts):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    if pts is None or len(pts) < 3:
        return 0.0
    
    # Calculate all triangle areas
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                areas.append(0.5 * abs(cross))
    
    if not areas:
        return 0.0
        
    min_area = min(areas)
    
    # Calculate convex hull area
    # Since points are on the unit circle and ordered by angle, the convex hull is the polygon itself.
    # Area can be calculated as sum of cross products of consecutive vertices / 2.
    hull_area = 0.0
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i+1) % n]
        hull_area += p1[0]*p2[1] - p1[1]*p2[0]
    hull_area = 0.5 * abs(hull_area)
    
    if hull_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def _smoothed_negative_score(gaps, epsilon=10.0):
    """
    Smoothed negative score for optimization.
    gaps: raw parameters.
    """
    pts = _get_points_from_gaps(gaps)
    if pts is None:
        return 0.0
        
    n = len(pts)
    # Calculate areas vectorized for speed in optimization loop
    # We need all triples
    # For N=14, 364 triples.
    # Vectorized calculation:
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    pi = pts[i]
    pj = pts[j]
    pk = pts[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Hull area
    # Points are on unit circle, ordered.
    # Area = 0.5 * sum(sin(theta_{i+1} - theta_i))
    # Or just use the polygon formula on pts.
    hull_area = 0.0
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i+1) % n]
        hull_area += p1[0]*p2[1] - p1[1]*p2[0]
    hull_area = 0.5 * abs(hull_area)
    
    if hull_area < 1e-10:
        return 0.0
        
    normalized_areas = areas / hull_area
    
    # Soft min approximation: -1/epsilon * log(mean(exp(-epsilon * x)))
    # We want to maximize min(x), so we minimize -min(x).
    # Soft min of x is approx -1/epsilon * log(mean(exp(-epsilon*x)))
    # So objective is -1/epsilon * log(mean(exp(-epsilon * normalized_areas)))
    # Wait, if we minimize this, we are maximizing the soft-min.
    
    # To avoid underflow/overflow, use log-sum-exp trick
    # log(sum(exp(-epsilon * a))) = -epsilon * min(a) + log(sum(exp(-epsilon * (a - min(a)))))
    min_a = np.min(normalized_areas)
    exp_sum = np.sum(np.exp(-epsilon * (normalized_areas - min_a)))
    soft_min = min_a - (1/epsilon) * np.log(exp_sum / len(normalized_areas))
    
    return -soft_min

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave margin
    
    best_score = 0.0
    best_gaps = np.ones(N-1) # Regular polygon gaps
    
    # Initial guess: Regular polygon
    # Gaps are all 2*pi/N
    initial_gaps = np.ones(N-1)
    
    # Multi-start
    n_restarts = 10
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if r == 0:
            current_gaps = initial_gaps.copy()
        else:
            # Perturb the regular polygon
            # Random gaps
            current_gaps = np.random.exponential(1.0, size=N-1)
            # Or mix of regular and random
            if r % 2 == 0:
                current_gaps = initial_gaps + np.random.normal(0, 0.1, size=N-1)
                current_gaps = np.abs(current_gaps) + 0.1
            else:
                # Two-ring like structure? 
                # Just random for now, the optimizer will fix it.
                current_gaps = np.random.uniform(0.5, 1.5, size=N-1)
        
        # Optimization Phase 1: Smoothed objective with L-BFGS-B
        # We optimize the raw gaps.
        # The function _smoothed_negative_score handles the normalization.
        
        try:
            res1 = scipy.optimize.minimize(
                _smoothed_negative_score, 
                current_gaps, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            best_gaps_candidate = res1.x
        except Exception:
            best_gaps_candidate = current_gaps
            
        # Optimization Phase 2: Exact objective with Nelder-Mead
        # Refine the result from Phase 1
        def _exact_negative_score(gaps):
            pts = _get_points_from_gaps(gaps)
            if pts is None:
                return 0.0
            score = _calculate_score(pts)
            return -score
        
        try:
            res2 = scipy.optimize.minimize(
                _exact_negative_score, 
                best_gaps_candidate, 
                method='Nelder-Mead', 
                options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 1000, 'maxfev': 1000}
            )
            best_gaps_candidate = res2.x
        except Exception:
            pass
            
        # Evaluate final score
        final_pts = _get_points_from_gaps(best_gaps_candidate)
        final_score = _calculate_score(final_pts)
        
        if final_score > best_score:
            best_score = final_score
            best_gaps = best_gaps_candidate
            
        # Check time
        if time.time() - start_time > time_limit:
            break
            
    # Return the best points found
    best_points = _get_points_from_gaps(best_gaps)
    
    # If something went wrong, fallback to regular polygon
    if best_points is None or len(best_points) != N:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.vstack([np.cos(angles), np.sin(angles)]).T
        
    return best_points
