import numpy as np
import time

N = 12
SQRT3 = np.sqrt(3.0)

def _hex_verts(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _min_outer_side(positions, angles):
    """
    Find the minimum side length S of a regular hexagon (centered at origin)
    that contains all given hexagons.
    positions: (N, 2) array of centers
    angles: (N,) array of angles in radians
    """
    # We will search for the optimal outer angle and S.
    # Since the configuration is fixed, we can scan the outer angle.
    best_S = float('inf')
    
    # Scan outer angle from 0 to 60 degrees
    # The symmetry of the hexagon is 60 degrees.
    # We can use a coarse grid and then refine.
    for outer_ang_deg in np.linspace(0, 60, 61):
        outer_ang_rad = np.radians(outer_ang_deg)
        # Normals of the outer hexagon
        normals = outer_ang_rad + np.radians(60 * np.arange(6))
        on_cos = np.cos(normals)
        on_sin = np.sin(normals)
        
        # For each inner hex, compute the required S
        # S = max over all normals i and all hexes k of:
        #   (dot(center_k, normal_i) + support_k(normal_i)) / apothem_factor?
        #   Actually, the condition is:
        #   dot(center, n) + support(angle_k, n) <= S * apothem_outer?
        #   No, for a regular hexagon of side S, the distance from center to edge (inradius) is S * sqrt(3) / 2.
        #   So the condition is: dot(center, n) + support <= S * sqrt(3) / 2.
        #   So S >= 2 * (dot(center, n) + support) / sqrt(3).
        
        max_val = 0.0
        for k in range(N):
            cx, cy = positions[k]
            ang_k = angles[k]
            dot_c = cx * on_cos + cy * on_sin
            # Support of unit hex at angle ang_k in direction normals
            # support = max_m cos(ang_k + 60m - normal_i)
            # We need max over i and m.
            # cos_vals shape (6, 6)
            cos_vals = np.cos(ang_k + np.radians(60 * np.arange(6))[:, None] - normals[None, :])
            supports = np.max(cos_vals, axis=0) # max over m for each i
            needed = dot_c + supports
            max_needed_for_k = np.max(needed)
            if max_needed_for_k > max_val:
                max_val = max_needed_for_k
        
        S_candidate = 2.0 * max_val / SQRT3
        if S_candidate < best_S:
            best_S = S_candidate
            
    return best_S

def construct():
    # Define a 3x4 grid of hexagons
    positions = []
    angles = []
    
    # Hex grid coordinates (q, r)
    # q: 0..3, r: 0..2
    for r in range(3):
        for q in range(4):
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            positions.append([x, y])
            angles.append(0.0) # All aligned
            
    positions = np.array(positions)
    angles = np.array(angles)
    
    # Center the configuration
    center = np.mean(positions, axis=0)
    positions -= center
    
    # Compute minimal S
    S = _min_outer_side(positions, angles)
    
    # Add a small safety margin
    S *= 1.01
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([float(positions[i][0]), float(positions[i][1]), float(np.degrees(angles[i]))])
    
    # Outer hexagon center is (0,0) because we centered the inner ones
    # Outer angle: We need to return the angle that minimizes S.
    # The function _min_outer_side found the S but not the angle.
    # Let's re-run to find the angle.
    
    best_S = float('inf')
    best_ang = 0.0
    for outer_ang_deg in np.linspace(0, 60, 61):
        outer_ang_rad = np.radians(outer_ang_deg)
        normals = outer_ang_rad + np.radians(60 * np.arange(6))
        on_cos = np.cos(normals)
        on_sin = np.sin(normals)
        
        max_val = 0.0
        for k in range(N):
            cx, cy = positions[k]
            ang_k = angles[k]
            dot_c = cx * on_cos + cy * on_sin
            cos_vals = np.cos(ang_k + np.radians(60 * np.arange(6))[:, None] - normals[None, :])
            supports = np.max(cos_vals, axis=0)
            needed = dot_c + supports
            max_needed_for_k = np.max(needed)
            if max_needed_for_k > max_val:
                max_val = max_needed_for_k
        
        S_candidate = 2.0 * max_val / SQRT3
        if S_candidate < best_S:
            best_S = S_candidate
            best_ang = outer_ang_deg
            
    return inner, (0.0, 0.0), float(best_S * 1.01), float(best_ang)
