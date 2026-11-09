import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _get_initial_positions():
    """
    Generate positions for a 1-2-3-3-2 hexagonal cluster.
    """
    positions = []
    # Row 0: 1 hex at (0,0)
    positions.append((0.0, 0.0))
    
    # Row 1: 2 hexes
    # y offset: 1.5
    # x offsets: +/- sqrt(3)/2
    positions.append((-np.sqrt(3)/2, 1.5))
    positions.append((np.sqrt(3)/2, 1.5))
    
    # Row 2: 3 hexes
    # y offset: 3.0
    # x offsets: -sqrt(3), 0, sqrt(3)
    positions.append((-np.sqrt(3), 3.0))
    positions.append((0.0, 3.0))
    positions.append((np.sqrt(3), 3.0))
    
    # Row 3: 3 hexes
    # y offset: -3.0
    # x offsets: -sqrt(3), 0, sqrt(3)
    positions.append((-np.sqrt(3), -3.0))
    positions.append((0.0, -3.0))
    positions.append((np.sqrt(3), -3.0))
    
    # Row 4: 2 hexes
    # y offset: -1.5
    # x offsets: +/- sqrt(3)/2
    positions.append((-np.sqrt(3)/2, -1.5))
    positions.append((np.sqrt(3)/2, -1.5))
    
    return positions

def _calc_required_side(positions):
    """
    Calculate the minimum side length of the outer hexagon
    required to contain all hexagons at given centers.
    Assumes inner hexagons are axis-aligned (angle 0).
    """
    max_val = 0.0
    for cx, cy in positions:
        # The vertices of a unit hexagon at (cx, cy) are at distance 1 from (cx, cy).
        # We need to check the max projection of the *vertices* onto the normals.
        # Instead of iterating 6 vertices * 11 hexagons, we can approximate
        # or just use the center + max radius.
        # However, the shape is not a circle.
        # Let's iterate vertices for accuracy.
        
        # Vertices of unit hexagon (angle 0)
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        verts = [
            (1.0, 0.0),
            (0.5, np.sqrt(3)/2),
            (-0.5, np.sqrt(3)/2),
            (-1.0, 0.0),
            (-0.5, -np.sqrt(3)/2),
            (0.5, -np.sqrt(3)/2)
        ]
        
        for vx, vy in verts:
            px = cx + vx
            py = cy + vy
            
            # Normals of outer hexagon sides (apothem direction)
            # n0: (sqrt(3)/2, 1/2)
            # n1: (0, 1)
            # n2: (-sqrt(3)/2, 1/2)
            # n3: (-sqrt(3)/2, -1/2)
            # n4: (0, -1)
            # n5: (sqrt(3)/2, -1/2)
            
            # Projections
            p0 = abs(np.sqrt(3)/2 * px + 1/2 * py)
            p1 = abs(py)
            p2 = abs(-np.sqrt(3)/2 * px + 1/2 * py)
            
            # Max projection for this vertex
            max_proj = max(p0, p1, p2)
            
            if max_proj > max_val:
                max_val = max_proj
                
    # L = max_proj / cos(30) = max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3)
    return 2.0 * max_val / np.sqrt(3)

def _overlap_penalty(positions):
    """
    Penalty for overlapping inner hexagons.
    Assumes axis-aligned. Overlap if center distance < sqrt(3).
    """
    penalty = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < np.sqrt(3):
                penalty += (np.sqrt(3) - dist)**2
    return penalty

def construct():
    start_time = time.time()
    
    # Initial configuration
    pos_list = _get_initial_positions()
    x0 = np.array([coord for p in pos_list for coord in p])
    
    # Objective function
    def objective(x):
        # Reshape x to positions
        pos = [(x[i], x[i+1]) for i in range(0, 2*N, 2)]
        
        # Time check
        if time.time() - start_time > 140:
            return 1e10
            
        L = _calc_required_side(pos)
        pen = _overlap_penalty(pos)
        
        # We want to minimize L, but avoid overlap.
        # If there is overlap, the configuration is invalid.
        # We add a large penalty to push it out.
        return L + 10.0 * pen

    # Optimization
    # Nelder-Mead is good for this small dimensional problem (22D)
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
        best_x = res.x
    except Exception:
        best_x = x0

    # Convert back to list of tuples
    best_positions = [(best_x[i], best_x[i+1]) for i in range(0, 2*N, 2)]
    
    # Calculate final L
    final_L = _calc_required_side(best_positions)
    
    # Check if overlap is still present (should be minimal)
    # If overlap is significant, fall back to initial
    if _overlap_penalty(best_positions) > 0.01:
        best_positions = pos_list
        final_L = _calc_required_side(best_positions)

    # Format output
    inner = []
    for x, y in best_positions:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), final_L, 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to make sure the entry point is correct.
# The task requires: construct() -> (inner, center, side, angle)
# My construct returns exactly that.
# EVOLVE-BLOCK-END
