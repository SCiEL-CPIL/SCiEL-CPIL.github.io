import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools
import sys

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """
    Fast calculation of minimum triangle area using vectorized operations.
    Points: (N, 2) array. Returns minimum area (not doubled).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            v1 = points[j] - pi
            v2s = points[j + 1:] - pi
            cross_vals = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            if len(areas) > 0:
                cur_min = np.min(areas)
                if cur_min < min_area:
                    min_area = cur_min
    return min_area

def _score(points):
    """
    Score = Min Triangle Area / Convex Hull Area.
    """
    try:
        if np.allclose(points, points[0]):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area < 1e-12:
            return 0.0
            
        min_tri_area = _min_triangle_area_fast(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_points = points.copy()
            best_score = score

    # Strategy: Circular Packing via Deterministic Grid Search
    # We restrict points to the unit circle. The convex hull is the polygon formed by these points.
    # The problem reduces to finding 14 angles theta_i that maximize min_tri_area / hull_area.
    # Since the hull is fixed to the convex hull of the points on the circle, and the points are on the circle,
    # the hull area is determined by the angles.
    
    # The regular 14-gon is a strong baseline.
    # We perform a grid search around the regular configuration.
    # Due to symmetry, we can fix theta_0 = 0 and optimize the relative angles.
    # However, a full grid search on 13 variables is too large.
    
    # Instead, we use a "local" grid search strategy:
    # 1. Start with regular polygon.
    # 2. Perturb each angle individually within a small range and keep the best.
    # 3. Repeat this process (coordinate ascent with discrete steps) until convergence or time limit.
    
    # Initialize with regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    current_score = _score(points)
    update_best(points, current_score)
    
    # Adaptive step size for the grid search
    # Start with a coarse step, refine if no improvement
    initial_step = 0.05 # ~3 degrees
    current_step = initial_step
    no_improvement_count = 0
    max_no_improvement = 20
    
    # We will iterate through each angle and try shifting it by +/- step
    # To maintain the "circular" nature, we just adjust the angle.
    # Note: The order of angles matters for the hull if they cross, but for a convex polygon
    # inscribed in a circle, the hull is just the polygon connecting them in angular order.
    # We assume the points remain in sorted angular order. If a point crosses another, 
    # the hull topology changes, but the set of points is the same. 
    # Actually, for points on a circle, the convex hull is always the polygon formed by the points 
    # in their circular order. So we can just sort the angles to ensure valid hull calculation 
    # or just rely on ConvexHull which handles it.
    
    # Optimization: 
    # Since all points are on the unit circle, the hull area is fixed by the set of angles.
    # The min triangle area is also fixed by the set of angles.
    
    # Let's use a smarter approach: 
    # The optimal configuration for "maximize min triangle area" on a circle is likely 
    # close to regular, but might have some "clumping" to avoid small triangles?
    # Actually, small triangles are formed by close points. So we want to maximize the minimum gap?
    # No, a triangle is formed by 3 points. The area is 0.5 * R^2 * |sin(a-b) + sin(b-c) + sin(c-a)|.
    # For small gaps, the area is small. So we want to avoid having 3 points close together.
    
    # Coordinate Ascent with Discrete Steps:
    while time.time() < deadline and no_improvement_count < max_no_improvement:
        improved = False
        for i in range(N):
            # Try shifting angle i
            original_angle = angles[i]
            # Try a few discrete shifts
            shifts = [current_step, -current_step, 2*current_step, -2*current_step, 0.5*current_step, -0.5*current_step]
            for shift in shifts:
                new_angle = original_angle + shift
                # Keep angles in [0, 2pi) for consistency, though not strictly necessary for cos/sin
                angles[i] = new_angle % (2 * np.pi)
                
                # Sort angles to maintain circular order? 
                # ConvexHull handles unsorted points, but for consistent indexing in future steps,
                # it's better to keep them sorted or just track the set.
                # Let's just compute score on the unsorted array, ConvexHull will sort internally.
                points = np.column_stack([np.cos(angles), np.sin(angles)])
                score = _score(points)
                
                if score > current_score:
                    current_score = score
                    update_best(points, current_score)
                    improved = True
                    # If improved, keep the new angle and stop trying other shifts for this i?
                    # Or continue to find the best shift for this i?
                    # Let's find the best shift for this i in one pass.
                    break 
                else:
                    # Revert
                    angles[i] = original_angle
            
            # If we found an improvement in this iteration, we restart the loop over angles?
            # Or just continue? Coordinate ascent usually continues.
            
        if not improved:
            no_improvement_count += 1
            # Reduce step size if stuck
            current_step *= 0.7
            if current_step < 1e-5:
                break
        else:
            no_improvement_count = 0
            # Reset step size? Or keep it?
            # Keep it to allow exploration.

    # Final Refinement: 
    # If we have time, do a finer grid search around the best found angles.
    if best_points is not None and time.time() < deadline - 5:
        # Extract angles from best_points
        x, y = best_points[:, 0], best_points[:, 1]
        best_angles = np.arctan2(y, x)
        
        # Fine-tune with very small steps
        fine_step = 1e-4
        for _ in range(10): # 10 passes of fine tuning
            if time.time() > deadline - 2:
                break
            improved_fine = False
            for i in range(N):
                original_angle = best_angles[i]
                for shift in [fine_step, -fine_step]:
                    best_angles[i] = (original_angle + shift) % (2 * np.pi)
                    points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
                    score = _score(points)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        improved_fine = True
                        break
                    else:
                        best_angles[i] = original_angle
            if not improved_fine:
                break

    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
