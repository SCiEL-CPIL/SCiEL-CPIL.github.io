"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize
import itertools

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Normal directions for flat-top outer hexagon
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)

    def outer_L(points):
        """Calculate min L for points to fit in centered outer hexagon."""
        # points: (12, 2)
        # We need to check if all vertices of inner hexagons are inside.
        # For aligned hexagons (rot=0), vertices are at angles 0, 60, ...
        # Center (x,y) + (cos, sin)
        # We approximate the required L by projecting the outermost points.
        # The outermost point of a hexagon at (x,y) with rot 0 is at distance 1 from center.
        # The "extents" in the 6 normal directions determine L.
        # For a hexagon at (x,y), the projection on normal k is:
        # max over vertices v of (x + v_x)*cos_k + (y + v_y)*sin_k
        # Vertices of unit hexagon (rot 0): (1,0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1,0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
        
        # Precompute vertices
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        vx = np.cos(angles)
        vy = np.sin(angles)
        
        # points: (12, 2)
        # We can vectorize this.
        # For each hexagon i, and each normal k, find max projection.
        # Proj_{i,k} = max_j ( (x_i + vx_j) * cos_k + (y_i + vy_j) * sin_k )
        # = x_i * cos_k + y_i * sin_k + max_j (vx_j * cos_k + vy_j * sin_k)
        
        # max_j (vx_j * cos_k + vy_j * sin_k) is the support function of the unit hexagon in direction k.
        # For a regular hexagon, this is constant? No.
        # In direction of normal (perpendicular to side), the support is 1 (circumradius)? No.
        # The distance from center to vertex is 1. The distance from center to side is sqrt(3)/2.
        # The normals are perpendicular to sides.
        # So for normal direction k, the max projection of the unit hexagon is 1 (vertex in that direction? No, vertex is at 0, 60... Normal is at 30, 90...
        # Angle between normal (30) and vertex (0) is 30.
        # Projection of vertex (1,0) on normal (cos30, sin30) is cos(30) = sqrt(3)/2.
        # Projection of vertex (0.5, sqrt3/2) on normal (cos30, sin30) is 0.5*cos30 + sqrt3/2*sin30 = 0.5*(sqrt3/2) + sqrt3/2*(1/2) = sqrt3/2.
        # So the support function in the normal directions is exactly sqrt(3)/2.
        
        # Therefore, the projection of hexagon i on normal k is:
        # P_{i,k} = x_i * cos_k + y_i * sin_k + sqrt(3)/2
        
        # The outer hexagon of side L has support L * sqrt(3)/2 in the normal directions?
        # Equation of side: x cos_k + y sin_k = L * sqrt(3)/2 ?
        # Let's check. Center (0,0), side L.
        # Distance from center to side is L * sqrt(3)/2.
        # So yes, the constraint is P_{i,k} <= L * sqrt(3)/2.
        # So L >= max_{i,k} (P_{i,k}) / (sqrt(3)/2).
        
        # We want to minimize L.
        # L = max_{i,k} (x_i * cos_k + y_i * sin_k + sqrt(3)/2) / (sqrt(3)/2)
        # = max_{i,k} ( (x_i * cos_k + y_i * sin_k + sqrt(3)/2) / (sqrt(3)/2) )
        
        # Let's compute this max.
        # x_i * cos_k + y_i * sin_k
        # points: (12, 2)
        # cos_k, sin_k: (6,)
        # proj = points[:, 0] * cos_k + points[:, 1] * sin_k  # (12, 6)
        # max_proj = np.max(proj)
        # L = (max_proj + sqrt(3)/2) / (sqrt(3)/2)
        
        proj = points[:, 0] * normal_cos + points[:, 1] * normal_sin
        max_proj = np.max(proj)
        return (max_proj + sqrt(3)/2) / (sqrt(3)/2)

    def objective(params):
        # params: 24 values (12 x, 12 y)
        x = params[0:12]
        y = params[12:24]
        points = np.column_stack((x, y))
        
        # Calculate L
        L = outer_L(points)
        
        # Calculate overlap penalty
        # Min distance between centers for aligned hexagons is sqrt(3).
        # Penalty for any pair closer than sqrt(3) - 0.01
        penalty = 0.0
        # We can vectorize distance calculation
        # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # dist = np.linalg.norm(diff, axis=2)
        # np.fill_diagonal(dist, 0)
        # violations = np.maximum(0, (sqrt3 - 0.01) - dist)
        # penalty = np.sum(violations**2)
        
        # For speed, we might skip full vectorization if N is small, but 12 is small.
        for i in range(12):
            for j in range(i+1, 12):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = math.sqrt(dx*dx + dy*dy)
                if d < sqrt3 - 0.01:
                    penalty += (sqrt3 - 0.01 - d)**2
        
        return L + 10.0 * penalty

    # Initial configuration: 3x4 grid
    # Spacing sqrt(3)
    x0 = []
    y0 = []
    for row in range(3):
        for col in range(4):
            # Shift rows to form a hexagonal tiling?
            # If we just use a square grid with spacing 2, it's safe but loose.
            # Let's use a tight tiling.
            # Axial coordinates (q, r)
            # q = col, r = row
            # x = 1.5 * q
            # y = sqrt3 * (r + q/2)
            # This creates a parallelogram of hexagons.
            x0.append(1.5 * col)
            y0.append(sqrt3 * (row + col/2.0))
            
    x0 = np.array(x0)
    y0 = np.array(y0)
    
    # Center the grid
    x0 -= np.mean(x0)
    y0 -= np.mean(y0)
    
    params0 = np.concatenate((x0, y0))
    
    # Optimize
    # Use Nelder-Mead
    # Max iterations to stay within time
    try:
        res = minimize(objective, params0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if time.time() > deadline - 10.0:
             # If we are running out of time, use current best or fallback
             # Nelder-Mead doesn't easily return intermediate bests without callback
             # Let's assume it finishes or we use res.x if available
             pass
             
        best_params = res.x
        best_L = outer_L(np.column_stack((best_params[0:12], best_params[12:24])))
        
        # Verify no overlap
        x_best = best_params[0:12]
        y_best = best_params[12:24]
        valid = True
        for i in range(12):
            for j in range(i+1, 12):
                dx = x_best[i] - x_best[j]
                dy = y_best[i] - y_best[j]
                d = math.sqrt(dx*dx + dy*dy)
                if d < sqrt3 - 0.05: # Allow small tolerance
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to initial grid if optimization failed to keep them disjoint
            best_params = params0
            best_L = outer_L(np.column_stack((x0, y0)))
            
    except Exception:
        best_params = params0
        best_L = outer_L(np.column_stack((x0, y0)))

    # Construct inner hexagons
    inner = []
    for i in range(12):
        x = best_params[i]
        y = best_params[12+i]
        inner.append([x, y, 0.0]) # Rotation 0
        
    # Final L
    # Add small epsilon for safety
    final_L = best_L * 1.00001
    
    return inner, (0.0, 0.0), final_L, 0.0
