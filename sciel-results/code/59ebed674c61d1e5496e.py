import numpy as np
from scipy.optimize import minimize
import time
import itertools

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600 - 20  # Safety margin

    # Precompute base angles for unit hexagon normals
    # For a regular hexagon with side 1, the normals to edges are at 30, 90, 150, 210, 270, 330 degrees
    # relative to the hexagon's orientation.
    # If the hexagon is rotated by angle 'a', the normals are at a + 30, a + 90, ...
    
    def get_hex_vertices(cx, cy, s, angle_deg):
        """Returns (6, 2) array of vertices."""
        a_rad = np.deg2rad(angle_deg)
        angles = a_rad + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack((
            cx + s * np.cos(angles),
            cy + s * np.sin(angles)
        ))

    def get_normals(angle_deg):
        """Returns (6, 2) array of outward normals for a hexagon with given angle."""
        a_rad = np.deg2rad(angle_deg)
        # Normals are at 30, 90, 150, 210, 270, 330 degrees relative to 0.
        # So absolute angles are a + 30, a + 90, ...
        normal_angles = a_rad + np.deg2rad(np.arange(30, 360, 60))
        return np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))

    def is_separated(v1, v2, n1, n2, tol=1e-6):
        """
        Check if two hexagons are separated.
        v1, v2: (6, 2) vertices.
        n1, n2: (6, 2) normals.
        Returns True if separated (no intersection).
        """
        # Combine normals
        normals = np.vstack([n1, n2])
        
        # Project vertices onto normals
        # p1 = v1 @ n.T -> (6, 12)
        # p2 = v2 @ n.T -> (6, 12)
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = np.min(p1, axis=0)
        max1 = np.max(p1, axis=0)
        min2 = np.min(p2, axis=0)
        max2 = np.max(p2, axis=0)
        
        # Separation if max1 < min2 - tol OR max2 < min1 - tol for ANY normal
        # We want to return True if they ARE separated.
        # So we check if there exists a normal where they are apart.
        sep1 = max1 < min2 - tol
        sep2 = max2 < min1 - tol
        
        if np.any(sep1) or np.any(sep2):
            return True
        
        return False

    def compute_side(positions, angles):
        """
        Compute the minimum side length S of the outer hexagon (centered at 0,0, angle 0)
        that contains all inner hexagons.
        """
        # Outer hexagon angle is 0. Normals at 30, 90, ...
        outer_normals = get_normals(0.0) # (6, 2)
        
        max_proj = 0.0
        for i in range(11):
            verts = get_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            # Project onto outer normals
            projs = verts @ outer_normals.T
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
        
        # Calculate side length
        S = compute_side(positions, angles)
        
        # Check overlaps and containment
        penalty = 0.0
        
        # Pre-compute vertices and normals for all hexes to save time?
        # It's only 11 hexes, so 55 pairs. 
        verts = []
        norms = []
        for i in range(11):
            verts.append(get_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
            norms.append(get_normals(angles[i]))
            
        # Overlap check
        for i in range(11):
            for j in range(i + 1, 11):
                if not is_separated(verts[i], verts[j], norms[i], norms[j]):
                    # Penalty for overlap
                    # Estimate distance between centers
                    d = np.linalg.norm(positions[i] - positions[j])
                    # If d is small, penalty is high.
                    # A unit hexagon has diameter 2. So if d < 2, they likely overlap.
                    # We use a smooth penalty.
                    penalty += max(0, 2.0 - d) ** 2
                    
        return S + 100 * penalty

    def is_valid(positions, angles):
        """
        Strictly check if the configuration is valid.
        1. S >= sqrt(11)
        2. No overlaps
        3. Containment
        """
        S = compute_side(positions, angles)
        if S < np.sqrt(11) - 1e-6:
            return False, S
            
        verts = []
        norms = []
        for i in range(11):
            verts.append(get_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
            norms.append(get_normals(angles[i]))
            
        for i in range(11):
            for j in range(i + 1, 11):
                if not is_separated(verts[i], verts[j], norms[i], norms[j], tol=1e-6):
                    return False, S
                    
        return True, S

    # Initial guess
    sqrt3 = np.sqrt(3)
    # 7 hexes in center flower
    init_pos = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        # 4 more hexes in corners? 
        # Let's try a symmetric arrangement
        [3, sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [3, -sqrt3]
    ])
    init_angles = np.zeros(11)
    
    x0 = np.concatenate([init_pos.flatten(), init_angles])
    
    best_x = x0
    best_S, valid = is_valid(best_x[:22].reshape(11, 2), best_x[22:33])
    if not valid:
        # Fallback to a known valid packing (seed-like)
        # This is a rough fallback, but we want to improve.
        # Let's use a grid or something simple if optimization fails.
        best_S = np.inf # Force update
        
    # Optimization Loop
    # Use multiple restarts with different initial conditions
    
    num_restarts = 20
    rng = np.random.default_rng(42)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            x0 = np.concatenate([init_pos.flatten(), init_angles])
        else:
            # Random perturbation
            x0 = best_x + rng.normal(0, 0.5, size=33)
            
        # Run Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < 1e9:
                x_candidate = res.x
                S_candidate, valid_candidate = is_valid(x_candidate[:22].reshape(11, 2), x_candidate[22:33])
                
                if valid_candidate and S_candidate < best_S:
                    best_S = S_candidate
                    best_x = x_candidate
        except Exception:
            pass

    # Final check
    positions = best_x[:22].reshape(11, 2)
    angles = best_x[22:33]
    
    valid_final, S_final = is_valid(positions, angles)
    
    if not valid_final:
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
    outer_hex_side_length = S_final

    return inner_hex_data, outer_hex_data, outer_hex_side_length
