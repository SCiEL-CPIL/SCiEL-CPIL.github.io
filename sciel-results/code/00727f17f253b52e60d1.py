"""
Strategy: Constrained radial optimization.
We fix the angular positions of the 14 points to those of a regular 14-gon (or a slight perturbation thereof) 
and optimize only the radial distances from the origin. 
This reduces the dimensionality from 28 to 14 (or 15 with a global rotation), making the landscape smoother 
and allowing more robust convergence within the time budget. 
We also allow for a small angular jitter in the initial guess to escape local minima associated with strict symmetry, 
but the primary search is in radial space.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    """
    n = len(points)
    if n < 3:
        return float('inf')
    
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(n - 2):
        dx_j = x[i+1:] - x[i]
        dy_j = y[i+1:] - y[i]
        
        for j_idx in range(len(dx_j)):
            j = i + 1 + j_idx
            dx_ij = dx_j[j_idx]
            dy_ij = dy_j[j_idx]
            
            if j + 1 >= n:
                continue
            
            dx_ik = x[j+1:] - x[i]
            dy_ik = y[j+1:] - y[i]
            
            cross = dx_ij * dy_ik - dx_ik * dy_ij
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    if min_area < 1e-10:
                        return min_area
                        
    return min_area

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective_radial(radii, angles):
    """
    Objective function to minimize (negative score) for radial parameters.
    radii: array of shape (N,)
    angles: array of shape (N,)
    """
    if np.any(radii <= 0):
        return 1e10
        
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack([x, y])
    
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Base angles for regular 14-gon
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    best_score_val = 0.0
    best_radii = np.ones(N)
    best_angles = base_angles.copy()
    
    # Strategy:
    # 1. Try strict regular polygon (radii=1)
    # 2. Optimize radii with fixed angles
    # 3. Optimize radii with slightly perturbed angles (multi-start)
    
    # Initial: Regular polygon
    r_init = np.ones(N)
    a_init = base_angles.copy()
    score_init = -_objective_radial(r_init, a_init)
    best_score_val = score_init
    best_radii = r_init.copy()
    best_angles = a_init.copy()
    
    # Optimization 1: Fixed angles, optimize radii
    # Powell is good for low-dimensional, non-smooth (due to min) problems
    try:
        res1 = minimize(
            lambda r: _objective_radial(r, base_angles),
            r_init,
            method='Powell',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if res1.fun < _objective_radial(best_radii, best_angles):
            best_radii = res1.x
            best_score_val = -res1.fun
    except Exception:
        pass
    
    # Optimization 2: Multi-start with angular perturbations
    # We allow small angular deviations to break symmetry if it helps
    # We optimize both radii and angles? No, let's keep angles fixed but try different base rotations or jittered angles.
    # Actually, optimizing angles too increases dimension to 28 again. 
    # Let's stick to optimizing radii for a few different angular configurations.
    
    rng = np.random.default_rng(42)
    
    num_starts = 20
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Generate perturbed angles
        # Small Gaussian noise on angles
        noise_std = 0.05 + 0.1 * (i / num_starts)
        a_pert = base_angles + rng.normal(0, noise_std, size=N)
        
        # Sort angles to maintain convexity order? 
        # The problem doesn't strictly require points to be in angular order for the hull, 
        # but for "convex region" placement, usually we consider the convex hull of the set.
        # If angles are sorted, the polygon formed by connecting them in order is convex if radii are positive.
        # Let's sort the angles and corresponding radii indices to keep the topology simple.
        sort_idx = np.argsort(a_pert)
        a_sorted = a_pert[sort_idx]
        
        # Initial radii: 1.0
        r_init_pert = np.ones(N)
        
        # Optimize radii for this angular configuration
        try:
            res_pert = minimize(
                lambda r: _objective_radial(r, a_sorted),
                r_init_pert,
                method='Powell',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            current_score = -res_pert.fun
            if current_score > best_score_val:
                best_score_val = current_score
                best_radii = res_pert.x
                best_angles = a_sorted
        except Exception:
            pass
            
    # Final refinement on the best found configuration
    if time.time() < deadline - 5:
        try:
            # Use Nelder-Mead for final polish on the best radii
            # We need to handle the angles as constants here
            res_final = minimize(
                lambda r: _objective_radial(r, best_angles),
                best_radii,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_final.fun < _objective_radial(best_radii, best_angles):
                best_radii = res_final.x
                best_score_val = -res_final.fun
        except Exception:
            pass

    # Reconstruct points
    x = best_radii * np.cos(best_angles)
    y = best_radii * np.sin(best_angles)
    points = np.column_stack([x, y])
    
    # Normalize to unit area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
        return final_points
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = points * scale
    else:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
