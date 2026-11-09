"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points in the set.
    Points: (N, 2) array.
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            if j + 1 < n:
                pk = points[j + 1:]
                v1 = pj - pi
                v2 = pk - pi
                cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
                areas = 0.5 * np.abs(cross)
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    min_area = _min_triangle_area(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _score_from_angles(angles):
    """
    Calculate score for points on unit circle with given angles.
    """
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return _score(points)

def _objective_angles(angles):
    """
    Objective function for angle optimization.
    """
    return -_score_from_angles(angles)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_score = 0.0
    best_angles = None
    
    # Initial angles: regular polygon
    base_angles = 2 * np.pi * np.arange(N) / N
    
    # Try multiple perturbations of the regular polygon angles
    rng = np.random.RandomState(42)
    
    starting_angles_list = [base_angles.copy()]
    
    # Perturbed angles
    for _ in range(10):
        perturbed = base_angles + rng.normal(0, 0.1, N)
        starting_angles_list.append(perturbed)
        
    # Shifted regular polygon (rotation shouldn't matter, but just in case)
    starting_angles_list.append(base_angles + np.pi/N)
    
    # Alternating perturbations
    for amp in [0.05, 0.1, 0.2]:
        pert = base_angles.copy()
        for i in range(N):
            if i % 2 == 0:
                pert[i] += amp
            else:
                pert[i] -= amp
        starting_angles_list.append(pert)

    # Optimize each starting set of angles
    for idx, start_angles in enumerate(starting_angles_list):
        if time.time() > deadline:
            break
        
        try:
            res = minimize(_objective_angles, start_angles, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            score = -res.fun
            angles = res.x
            
            if score > best_score:
                best_score = score
                best_angles = angles.copy()
        except Exception:
            pass
    
    # If we have a good solution, try to refine it further
    if best_angles is not None and time.time() < deadline - 10:
        try:
            res_refine = minimize(_objective_angles, best_angles, method='Nelder-Mead',
                                  options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
            score_refine = -res_refine.fun
            angles_refine = res_refine.x
            
            if score_refine > best_score:
                best_score = score_refine
                best_angles = angles_refine.copy()
        except Exception:
            pass
            
    # Convert best angles to points
    if best_angles is not None:
        best_points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    else:
        # Fallback to regular polygon
        best_points = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
        
    return best_points
# EVOLVE-BLOCK-END
