import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized implementation for speed.
    """
    min_area = float('inf')
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Loop over i and j, vectorize over k
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            # k ranges from j+1 to n-1
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _score_sym(params):
    """
    Objective function for centrally symmetric configuration.
    params: array of shape (14,).
    First 7 are radii r_0..r_6, next 7 are angles theta_0..theta_6.
    """
    r = params[:7]
    theta = params[7:]
    
    # Ensure angles are sorted and within (0, pi)
    # The optimizer might violate bounds, so we clip or penalize
    if np.any(theta <= 0) or np.any(theta >= np.pi):
        return 1e10
    if not np.all(np.diff(theta) > 0):
        return 1e10
    if np.any(r <= 0.01):
        return 1e10
        
    # Construct 7 points in the upper half plane (conceptually)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    P_half = np.column_stack((x, y))
    
    # Construct full 14 points: P_0..P_6 and -P_6..-P_0 (reversed for convex order?)
    # If we go counter-clockwise around the origin:
    # Angles are theta_0 < ... < theta_6 < pi.
    # The symmetric points have angles theta_i + pi.
    # So the order is:
    # 0: theta_0
    # 1: theta_1
    # ...
    # 6: theta_6
    # 7: theta_0 + pi
    # ...
    # 13: theta_6 + pi
    
    # Wait, if we just take -P_half, the angles are theta_i + pi.
    # The sequence of angles is theta_0, ..., theta_6, theta_0+pi, ..., theta_6+pi.
    # This is a valid angular ordering.
    
    P_full = np.vstack((P_half, -P_half))
    
    # Check if P_full forms a convex polygon
    # A polygon is convex if all cross products of consecutive edges have the same sign
    # and non-zero.
    n = 14
    is_convex = True
    for i in range(n):
        p1 = P_full[i]
        p2 = P_full[(i + 1) % n]
        p3 = P_full[(i + 2) % n]
        
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        if cross < 1e-9: # Allow small tolerance, but strictly positive for CCW
            is_convex = False
            break
            
    if not is_convex:
        return 1e10
        
    try:
        hull = ConvexHull(P_full)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area < 1e-6:
        return 1e10
        
    min_area = _min_triangle_area(P_full)
    
    if min_area < 1e-9:
        return 1e10
        
    score = min_area / hull_area
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Bounds for Differential Evolution
    # Radii: [0.1, 5.0]
    # Angles: [0.01, pi - 0.01] but must be sorted.
    # DE doesn't handle sorted constraints well directly.
    # We can define bounds for each angle such that theta_i is in [i*pi/7, (i+1)*pi/7] roughly?
    # Or just use [0, pi] and penalize in the objective.
    
    # Better: Define bounds for theta_i as [i * (pi/7), (i+1) * (pi/7)]?
    # This forces them to be in specific sectors, which might be too restrictive if the optimal solution is not evenly spaced.
    # However, for a convex polygon, the angles must be sorted.
    # Let's use bounds [0, pi] and rely on the penalty.
    
    bounds = []
    for i in range(7):
        bounds.append((0.1, 5.0)) # Radii
    for i in range(7):
        bounds.append((0.01, np.pi - 0.01)) # Angles
        
    # Initial guess: Regular 14-gon
    # Angles for half: 0, pi/7, 2pi/7, ... 6pi/7
    init_theta = np.linspace(0, np.pi, 7, endpoint=False)
    # Shift slightly to avoid 0 and pi boundaries
    init_theta = init_theta + 0.01
    init_r = np.ones(7)
    
    init_params = np.concatenate((init_r, init_theta))
    
    best_params = init_params.copy()
    best_score_val = _score_sym(best_params)
    
    # Phase 1: Differential Evolution
    # DE is slow, so limit iterations.
    try:
        de_res = differential_evolution(
            _score_sym,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-6,
            polish=False, # We will polish manually
            init='sobol',
            popsize=15
        )
        
        if time.time() < deadline - 20:
            if de_res.fun < best_score_val:
                best_params = de_res.x
                best_score_val = de_res.fun
                
    except Exception:
        pass

    # Phase 2: Local Refinement (Powell or Nelder-Mead)
    # Powell is often better for this type of non-smooth problem with some structure.
    try:
        res2 = minimize(
            _score_sym,
            best_params,
            method='Powell',
            options={
                'maxiter': 5000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if time.time() < deadline - 5:
            if res2.fun < best_score_val:
                best_params = res2.x
                best_score_val = res2.fun
                
    except Exception:
        pass
        
    # Phase 3: Try Nelder-Mead on the best found
    if time.time() < deadline - 5:
        try:
            res3 = minimize(
                _score_sym,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if time.time() < deadline - 5:
                if res3.fun < best_score_val:
                    best_params = res3.x
                    best_score_val = res3.fun
                    
        except Exception:
            pass

    # Reconstruct points from best params
    r = best_params[:7]
    theta = best_params[7:]
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    P_half = np.column_stack((x, y))
    P_full = np.vstack((P_half, -P_half))
    
    # Normalize to unit area
    try:
        hull = ConvexHull(P_full)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon if something went wrong
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
        return final_points
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = P_full * scale
    else:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
# EVOLVE-BLOCK-END
