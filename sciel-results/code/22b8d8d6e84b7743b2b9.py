import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 14

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
        
        min_area = np.inf
        # Compute all triangle areas
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Area = 0.5 * |cross(p2-p1, p3-p1)|
                    v1 = p2 - p1
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_area:
                        min_area = area
        
        return min_area / hull_area
    except Exception:
        return 0.0

def _double_ring_points(r1, r2, rot):
    """
    Generate points for a double-ring configuration with 7-fold symmetry.
    Outer ring: 7 points at radius r1, angles k * 2*pi/7
    Inner ring: 7 points at radius r2, angles (k + rot) * 2*pi/7
    
    The 'rot' parameter controls the angular offset of the inner ring relative to the outer.
    The optimal rot is likely pi/7 (half the step) to maximize separation.
    """
    k = np.arange(7)
    angles_out = 2.0 * np.pi * k / 7.0
    angles_in = 2.0 * np.pi * (k + rot) / 7.0
    
    x_out = r1 * np.cos(angles_out)
    y_out = r1 * np.sin(angles_out)
    
    x_in = r2 * np.cos(angles_in)
    y_in = r2 * np.sin(angles_in)
    
    # Interleave or stack? The score is invariant to permutation.
    # Let's just stack them.
    points = np.column_stack((np.concatenate([x_out, x_in]), np.concatenate([y_out, y_in])))
    return points

def _optimize_double_ring():
    """
    Optimize the double-ring parameters (r1, r2, rot).
    Scale is invariant, so we can fix r1 = 1.0 and optimize r2 in [0, 1] and rot in [0, 1] (fraction of step).
    Actually, rot is an angle offset. Let's optimize rot in [0, 2*pi/7] due to symmetry.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    def objective(params):
        if time.time() > deadline:
            return 0.0
        
        r1 = params[0]
        r2 = params[1]
        rot = params[2]
        
        points = _double_ring_points(r1, r2, rot)
        return -_calculate_score(points)
    
    # Bounds:
    # r1: [0.5, 1.0] (must be positive, scale invariant so 1.0 is fine, but let's allow variation just in case? No, fix r1=1.0 to reduce dim)
    # If we fix r1=1.0, we only optimize r2 and rot.
    # But what if the optimal configuration has r1 != 1.0 relative to r2? 
    # Since scale is invariant, we can always normalize so max(r1, r2) = 1.
    # If r1 >= r2, we can set r1=1 and r2 in [0,1].
    # If r2 > r1, we can swap them? No, the rings are distinct.
    # However, the configuration is symmetric if we swap rings and adjust rot?
    # Let's just search r1 in [0.1, 1.0], r2 in [0.0, 1.0], rot in [0.0, 2*pi/7].
    
    bounds = [
        (0.1, 1.0),  # r1
        (0.0, 1.0),  # r2
        (0.0, 2.0 * np.pi / 7.0)  # rot
    ]
    
    # Use Differential Evolution on this small 3D space
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20, 
            tol=1e-8,
            polish=False,
            init='latinhypercube'
        )
        best_params = res.x
        best_score = -res.fun
    except Exception as e:
        # Fallback
        best_params = np.array([1.0, 0.5, np.pi/7.0])
        best_score = -objective(best_params)
        
    # Polish
    if time.time() < deadline - 5.0:
        try:
            res_polish = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if -res_polish.fun > best_score:
                best_params = res_polish.x
                best_score = -res_polish.fun
        except Exception:
            pass
            
    r1, r2, rot = best_params
    points = _double_ring_points(r1, r2, rot)
    
    return points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try double ring first
    points, score = _optimize_double_ring()
    
    # Final verification
    final_score = _calculate_score(points)
    if final_score <= 0.0:
        # Fallback to regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return points
