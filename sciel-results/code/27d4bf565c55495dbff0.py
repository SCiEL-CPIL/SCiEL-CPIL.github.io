"""Heilbronn convex regions: maximize minimum triangle area over all triples."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
import itertools

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _generate_two_shell_config(r_inner, angle_offset):
    """
    Generate a two-shell configuration with 13 points.
    - 7 points on outer shell (regular heptagon)
    - 6 points on inner shell (regular hexagon)
    
    Parameters:
    - r_inner: radius of inner shell (outer shell has radius 1.0)
    - angle_offset: rotation angle of inner shell relative to outer
    """
    n_outer = 7
    n_inner = 6
    
    # Outer shell: regular heptagon
    angles_outer = 2 * np.pi * np.arange(n_outer) / n_outer
    outer_points = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    # Inner shell: regular hexagon with rotation offset
    angles_inner = 2 * np.pi * np.arange(n_inner) / n_inner + angle_offset
    inner_points = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    # Combine all points
    all_points = np.vstack([outer_points, inner_points])
    return all_points

def _two_shell_objective(params):
    """Objective function for two-shell optimization (negative score)."""
    r_inner, angle_offset = params
    # Constrain r_inner to be positive and reasonable
    r_inner = max(0.1, min(0.95, r_inner))
    points = _generate_two_shell_config(r_inner, angle_offset)
    score = _score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with a reasonable initial guess
    best_params = np.array([0.5, 0.0])
    best_score = _score(_generate_two_shell_config(0.5, 0.0))
    
    # Try multiple initial configurations for two-shell optimization
    initial_guesses = [
        [0.3, 0.0],
        [0.4, 0.0],
        [0.5, 0.0],
        [0.6, 0.0],
        [0.7, 0.0],
        [0.5, np.pi/6],
        [0.5, np.pi/7],
        [0.5, np.pi/12],
        [0.6, np.pi/7],
        [0.4, np.pi/6],
    ]
    
    for init in initial_guesses:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                _two_shell_objective,
                init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            candidate_score = -result.fun
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    # Also try a three-shell configuration as an alternative
    def _generate_three_shell_config(r1, r2, angle1, angle2):
        """
        Three-shell configuration:
        - 5 points on outer shell
        - 4 points on middle shell  
        - 4 points on inner shell
        """
        n_outer, n_mid, n_inner = 5, 4, 4
        
        angles_outer = 2 * np.pi * np.arange(n_outer) / n_outer
        outer_points = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        
        angles_mid = 2 * np.pi * np.arange(n_mid) / n_mid + angle1
        mid_points = r1 * np.column_stack([np.cos(angles_mid), np.sin(angles_mid)])
        
        angles_inner = 2 * np.pi * np.arange(n_inner) / n_inner + angle2
        inner_points = r2 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        
        all_points = np.vstack([outer_points, mid_points, inner_points])
        return all_points
    
    def _three_shell_objective(params):
        r1, r2, angle1, angle2 = params
        r1 = max(0.1, min(0.7, r1))
        r2 = max(0.1, min(0.5, r2))
        points = _generate_three_shell_config(r1, r2, angle1, angle2)
        score = _score(points)
        return -score
    
    if time.time() < deadline - 15:
        three_shell_inits = [
            [0.5, 0.3, 0.0, 0.0],
            [0.6, 0.3, np.pi/4, 0.0],
            [0.5, 0.35, np.pi/5, np.pi/6],
        ]
        
        for init in three_shell_inits:
            if time.time() > deadline - 10:
                break
            
            try:
                result = minimize(
                    _three_shell_objective,
                    init,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                
                candidate_score = -result.fun
                if candidate_score > best_score:
                    best_score = candidate_score
                    # Store as a flag that we found a better three-shell config
                    # We'll reconstruct it at the end
            
            except Exception:
                continue
    
    # Reconstruct the best two-shell configuration
    r_inner, angle_offset = best_params
    r_inner = max(0.1, min(0.95, r_inner))
    final_points = _generate_two_shell_config(r_inner, angle_offset)
    
    return final_points
# EVOLVE-BLOCK-END
