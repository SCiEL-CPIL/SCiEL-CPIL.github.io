import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 550  # Leave 50s margin

    def get_hex_vertices(cx, cy, s, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        angles = angle_rad + np.arange(6) * np.pi / 3
        xs = cx + s * np.cos(angles)
        ys = cy + s * np.sin(angles)
        return np.column_stack((xs, ys))

    def calculate_side_length(positions, angles):
        # positions: (11, 2), angles: (11,)
        # Outer hexagon centered at (0,0), angle 0
        # Normals for outer hexagon (flat top orientation, angle 0)
        # Vertices at 0, 60, ...
        # Normals at 30, 90, ... ? No.
        # If vertices are at 0, 60, 120...
        # Edge from 0 to 60. Midpoint angle 30. Normal is 30?
        # Let's stick to the definition:
        # Hexagon (cx, cy, s, a) vertices at a + 60i.
        # If a=0, vertices at 0, 60, 120, 180, 240, 300.
        # Edge 0-60. Normal points out.
        # The apothem direction for edge 0-60 is 30 degrees.
        # So normals are at 30, 90, 150, 210, 270, 330.
        
        normals = np.array([
            [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))],
            [np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))],
            [np.cos(np.deg2rad(150)), np.sin(np.deg2rad(150))],
            [np.cos(np.deg2rad(210)), np.sin(np.deg2rad(210))],
            [np.cos(np.deg2rad(270)), np.sin(np.deg2rad(270))],
            [np.cos(np.deg2rad(330)), np.sin(np.deg2rad(330))]
        ])
        
        max_proj = 0.0
        for i in range(11):
            verts = get_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            # Project all vertices onto all 6 normals
            projs = verts @ normals.T # (6, 6)
            max_proj = max(max_proj, np.max(projs))
            
        # max_proj is the required apothem A
        # S = 2 * A / sqrt(3)
        return 2 * max_proj / np.sqrt(3)

    def overlap_penalty(positions, angles):
        penalty = 0.0
        n = 11
        verts_list = []
        for i in range(n):
            verts_list.append(get_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
        
        for i in range(n):
            for j in range(i+1, n):
                v1 = verts_list[i]
                v2 = verts_list[j]
                
                # Get edge normals for hex 1 and hex 2
                # Normals of a hexagon at angle a are a + 30 + 60k
                a1 = np.deg2rad(angles[i])
                a2 = np.deg2rad(angles[j])
                
                n1 = a1 + np.pi/6 + np.arange(6) * np.pi/3
                n2 = a2 + np.pi/6 + np.arange(6) * np.pi/3
                
                all_normals = np.concatenate((n1, n2))
                
                separated = False
                min_overlap = float('inf')
                
                for ang in all_normals:
                    n = np.array([np.cos(ang), np.sin(ang)])
                    p1 = v1 @ n
                    p2 = v2 @ n
                    
                    min1, max1 = np.min(p1), np.max(p1)
                    min2, max2 = np.min(p2), np.max(p2)
                    
                    if max1 < min2 or max2 < min1:
                        separated = True
                        break
                    
                    # Overlap length on this axis
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap < min_overlap:
                        min_overlap = overlap
                
                if not separated:
                    # They overlap. Penalty is the minimum overlap depth
                    # This is a rough measure.
                    penalty += min_overlap
        
        return penalty

    def objective(vars):
        positions = vars[:22].reshape(11, 2)
        angles = vars[22:]
        
        # Fix first hexagon at 0,0,0 to break symmetry
        # But we are optimizing all. Let's just let it optimize.
        # To ensure stability, we can penalize movement of center?
        # No, just optimize all.
        
        s = calculate_side_length(positions, angles)
        p = overlap_penalty(positions, angles)
        
        # Large weight for penalty to ensure validity
        return s + 100.0 * p

    # Initial guess: Grid from seed
    init_positions = np.array([
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.17],
        [1.25, 2.17],
        [-1.25, -2.17],
        [1.25, -2.17],
        [-3.75, 2.17],
        [3.75, 2.17],
        [-3.75, -2.17],
        [3.75, -2.17]
    ])
    init_angles = np.zeros(11)
    
    init_vars = np.concatenate((init_positions.flatten(), init_angles))
    
    best_vars = init_vars
    best_score = objective(init_vars)
    
    # Optimization loop
    options = {
        'maxiter': 500,
        'xatol': 1e-6,
        'fatol': 1e-6,
        'disp': False
    }
    
    # Try Nelder-Mead
    while time.time() < deadline:
        res = minimize(objective, best_vars, method='Nelder-Mead', options=options)
        
        if res.fun < best_score:
            best_vars = res.x
            best_score = res.fun
        
        # Check time
        if time.time() > deadline:
            break
            
        # Perturb to escape local minima
        # Only perturb if we have time
        if time.time() > deadline - 100:
            break
            
        perturbation = np.random.normal(0, 0.1, size=33)
        # Keep center fixed? No, let it move.
        candidate = best_vars + perturbation
        # Ensure candidate is reasonable
        cand_score = objective(candidate)
        if cand_score < best_score + 10: # Allow some uphill moves
             best_vars = candidate
             best_score = cand_score
             
    # Extract best result
    positions = best_vars[:22].reshape(11, 2)
    angles = best_vars[22:]
    
    # Re-center the packing to (0,0) to match outer hexagon center
    # The outer hexagon is defined at (0,0).
    # The inner positions are absolute.
    # We should shift inner positions so that their centroid is (0,0)?
    # No, the outer hexagon is fixed at (0,0). The inner hexagons must be inside it.
    # The optimization minimized S based on the extent from (0,0).
    # So the current positions are relative to (0,0).
    
    inner_hex_data = np.column_stack((positions, angles))
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = calculate_side_length(positions, angles)
    
    # Final check: If penalty is not 0, the solution is invalid.
    # We must ensure the returned solution is valid.
    # If the optimizer got stuck in an invalid state, we should fall back to the seed or a known valid state.
    final_penalty = overlap_penalty(positions, angles)
    if final_penalty > 1e-4:
        # Fallback to seed
        inner_hex_data = np.array([
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
            [3.75, -2.17, 0]
        ])
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        outer_hex_side_length = 8.0
    else:
        # Recalculate side length accurately
        outer_hex_side_length = calculate_side_length(positions, angles)

    return inner_hex_data, outer_hex_data, outer_hex_side_length
