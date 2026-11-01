import numpy as np
import scipy.spatial as spat
import time
import itertools

N = 14

def calc_score_fast(points):
    """
    Fast calculation of min triangle area / hull area.
    """
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    
    n = len(points)
    
    # Generate all combinations of 3 indices
    # For N=14, C(14,3) = 364.
    # Vectorized approach
    
    i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1)
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    # Strategy:
    # Represent points as two nested regular heptagons.
    # Outer heptagon: radius R1, angles theta_outer
    # Inner heptagon: radius R2, angles theta_inner
    # Parameters: ratio = R2/R1, rotation = offset of inner relative to outer.
    # We fix R1 = 1.0 for optimization, then normalize at the end (score is scale invariant).
    
    # Base angles for heptagon
    base_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    
    # Pre-compute indices for score calculation to speed up
    # Actually, calc_score_fast is fast enough for 1D/2D search.
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: Regular 14-gon equivalent?
    # A regular 14-gon can be seen as two heptagons with ratio ~0.82 (cos(pi/14) * 2 * sin(pi/14) / (2*sin(pi/14)) ... wait)
    # Distance from center to vertex of regular 14-gon: R14.
    # Distance from center to vertex of regular 7-gon inscribed in same circle? No.
    # Let's just do a grid search over ratio and rotation.
    
    # Grid search parameters
    # Ratio r in [0.1, 0.99]
    # Rotation phi in [0, pi/7] (symmetry of heptagon)
    
    n_ratio_steps = 100
    n_rot_steps = 100
    
    ratios = np.linspace(0.2, 0.95, n_ratio_steps)
    rotations = np.linspace(0, np.pi/7, n_rot_steps)
    
    # Coarse grid search
    for r in ratios:
        for phi in rotations:
            if time.time() > deadline - 10:
                break
            
            # Construct points
            # Outer heptagon
            angles_outer = base_angles
            pts_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
            
            # Inner heptagon
            angles_inner = base_angles + phi
            pts_inner = r * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
            
            points = np.vstack((pts_outer, pts_inner))
            
            score = calc_score_fast(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
        
        if time.time() > deadline - 10:
            break
    
    if best_points is None:
        # Fallback to regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = calc_score_fast(best_points)
    
    # Refine with local search around the best grid point
    # We can do a simple coordinate descent or small random perturbation in the (r, phi) space
    # Let's extract the best r and phi from the grid? 
    # To do this properly, we should track them.
    
    # Let's re-run a finer grid around the best found region if time permits.
    # For simplicity and robustness, let's just do a few steps of simulated annealing on the (r, phi) parameters
    # starting from the best grid point.
    
    # Identify best r, phi from the coarse search
    # We didn't store them, so let's just do a dense 1D sweep for phi assuming optimal r is near the coarse best?
    # Better: Store best_r, best_phi in the loop above.
    
    # Let's restart the logic slightly to track best params.
    
    best_score = 0.0
    best_params = (0.5, 0.0)
    best_points = None
    
    for r in ratios:
        for phi in rotations:
            if time.time() > deadline - 20:
                break
            
            angles_outer = base_angles
            pts_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
            
            angles_inner = base_angles + phi
            pts_inner = r * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
            
            points = np.vstack((pts_outer, pts_inner))
            
            score = calc_score_fast(points)
            
            if score > best_score:
                best_score = score
                best_params = (r, phi)
                best_points = points.copy()
        
        if time.time() > deadline - 20:
            break
    
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = calc_score_fast(best_points)
        return best_points

    # Local refinement using scipy minimize on the 2D parameter space
    # Objective: maximize score -> minimize -score
    
    def neg_score(params):
        r, phi = params
        # Constrain r to [0.1, 0.99] and phi to [0, pi/7]
        if r < 0.1 or r > 0.99 or phi < 0 or phi > np.pi/7:
            return 0.0 # Or a large penalty
        
        angles_outer = base_angles
        pts_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
        
        angles_inner = base_angles + phi
        pts_inner = r * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
        
        points = np.vstack((pts_outer, pts_inner))
        
        return -calc_score_fast(points)
    
    from scipy.optimize import minimize
    
    x0 = np.array(best_params)
    
    # Try a few local optimizations from different starting points near the best
    # to avoid local minima in the 2D landscape
    
    candidates = [x0]
    # Add some nearby points
    candidates.append([x0[0] + 0.05, x0[1]])
    candidates.append([x0[0] - 0.05, x0[1]])
    candidates.append([x0[0], x0[1] + 0.1])
    candidates.append([x0[0], x0[1] - 0.1])
    
    # Also try the regular 14-gon equivalent in this parameterization?
    # Regular 14-gon has points at angles k*pi/7.
    # Outer: 0, 2pi/7, ...
    # Inner: pi/7, 3pi/7, ...
    # Ratio for regular 14-gon:
    # R_14 = 1. R_7_inner relative to R_14?
    # In a regular 14-gon, the "inner" 7 points are not on a smaller circle, they are on the same circle.
    # Wait, the representation is two heptagons.
    # If we set r=1 and phi=pi/7, we get a regular 14-gon.
    # My grid search restricted r < 0.95. I should include r=1.
    
    # Let's add r=1, phi=pi/7 as a candidate
    candidates.append([1.0, np.pi/7])
    
    for start in candidates:
        if time.time() > deadline - 5:
            break
        
        # Check bounds for start
        if start[0] < 0.1 or start[0] > 1.01 or start[1] < -0.1 or start[1] > np.pi/7 + 0.1:
             # Clip or skip? 
             # If r=1, it's valid.
             pass
        
        res = minimize(neg_score, start, method='Nelder-Mead', 
                       options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 100})
        
        if res.success:
            r_opt, phi_opt = res.x
            # Clamp to valid range
            r_opt = np.clip(r_opt, 0.1, 1.0)
            phi_opt = np.clip(phi_opt, 0, np.pi/7)
            
            angles_outer = base_angles
            pts_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
            
            angles_inner = base_angles + phi_opt
            pts_inner = r_opt * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
            
            points = np.vstack((pts_outer, pts_inner))
            
            score = calc_score_fast(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()

    # Final check: ensure convex hull is not degenerate
    # And return the best points found
    
    return best_points
