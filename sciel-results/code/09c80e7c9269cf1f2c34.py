import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600 - 15  # Safety margin

    # Precompute base angles and normals for unit hexagon (flat-top)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    # Normals (perpendicular to edges) are at 30, 90, 150, 210, 270, 330 degrees
    base_angles = np.deg2rad(np.arange(30, 360, 60))
    base_normals = np.column_stack((np.cos(base_angles), np.sin(base_angles)))  # (6, 2)
    
    # Outer hexagon is flat-top, centered at origin.
    # Its normals are the same as base_normals.
    outer_normals = base_normals

    def hex_vertices(cx, cy, s, a_deg):
        a_rad = np.deg2rad(a_deg)
        angles = a_rad + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack((
            cx + s * np.cos(angles),
            cy + s * np.sin(angles)
        ))

    def get_normals(a_deg):
        """Get the 6 edge normals for a hexagon with given angle."""
        a_rad = np.deg2rad(a_deg)
        c, s = np.cos(a_rad), np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        return base_normals @ R.T

    def calc_overlap_penalty(v1, n1, v2, n2):
        """
        Calculate the minimum overlap depth if two hexagons intersect.
        Returns 0 if separated.
        v1, v2: (6, 2) vertex arrays
        n1, n2: (6, 2) normal arrays
        """
        min_overlap = np.inf
        all_overlap = True
        
        # Check all 12 normals
        for i in range(6):
            n = n1[i]
            p1 = v1 @ n
            p2 = v2 @ n
            min1, max1 = p1.min(), p1.max()
            min2, max2 = p2.min(), p2.max()
            
            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return 0.0 # Separated
            
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                 overlap = 0
            if overlap < min_overlap:
                min_overlap = overlap
        
        for i in range(6):
            n = n2[i]
            p1 = v1 @ n
            p2 = v2 @ n
            min1, max1 = p1.min(), p1.max()
            min2, max2 = p2.min(), p2.max()
            
            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return 0.0 # Separated
            
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                 overlap = 0
            if overlap < min_overlap:
                min_overlap = overlap
        
        return min_overlap

    def get_min_side(positions, angles):
        """
        Compute the minimum side length of the outer hexagon.
        """
        max_proj = 0.0
        for i in range(11):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            projs = verts @ outer_normals.T  # (6, 6)
            max_proj = max(max_proj, np.max(projs))
        
        # Apothem = S * sqrt(3) / 2
        # max_proj <= Apothem
        # S >= 2 * max_proj / sqrt(3)
        S = 2 * max_proj / np.sqrt(3)
        return S

    def objective(x):
        if time.time() > deadline:
            return 1e10
        
        positions = x[:22].reshape(11, 2)
        angles = x[22:33]
        
        # Calculate min side
        S = get_min_side(positions, angles)
        
        # Check overlaps and compute penalty
        penalty = 0.0
        for i in range(11):
            v1 = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            n1 = get_normals(angles[i])
            for j in range(i+1, 11):
                v2 = hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                n2 = get_normals(angles[j])
                overlap = calc_overlap_penalty(v1, n1, v2, n2)
                if overlap > 0:
                    penalty += overlap ** 2
        
        return S + 1000 * penalty

    # Initial guess: 7 flower + 4 corners
    sqrt3 = np.sqrt(3)
    init_pos = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        [3, sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [3, -sqrt3]
    ])
    init_angles = np.zeros(11)
    
    x0 = np.concatenate([init_pos.flatten(), init_angles])
    
    best_x = x0
    best_val = objective(x0)
    
    # Optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < best_val:
            best_x = result.x
            best_val = result.fun
    except Exception:
        pass

    # Second pass with refined initial point
    if time.time() < deadline - 60:
        try:
            result2 = minimize(objective, best_x, method='Nelder-Mead',
                               options={'maxiter': 50000, 'xatol': 1e-9, 'fatol': 1e-11})
            if result2.fun < best_val:
                best_x = result2.x
                best_val = result2.fun
        except Exception:
            pass

    # Extract best result
    positions = best_x[:22].reshape(11, 2)
    angles = best_x[22:33]
    
    # Final validation
    S = get_min_side(positions, angles)
    S = max(S, np.sqrt(11))
    
    # Check overlaps again with final positions
    valid = True
    for i in range(11):
        v1 = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        n1 = get_normals(angles[i])
        for j in range(i+1, 11):
            v2 = hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
            n2 = get_normals(angles[j])
            if calc_overlap_penalty(v1, n1, v2, n2) > 1e-6:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to seed
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    inner_hex_data = np.column_stack((positions, angles))
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = S

    return inner_hex_data, outer_hex_data, outer_hex_side_length
