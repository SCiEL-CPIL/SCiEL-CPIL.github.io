import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    # Parameters
    n_hex = 11
    eps = 1e-6
    
    # Initial guess: The current "flower" pattern
    # Ring 0: 1
    # Ring 1: 6
    # Ring 2: 4
    x0 = np.zeros((n_hex, 3))
    
    # Center
    x0[0] = [0, 0, 30.0]
    
    # Ring 1
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        x0[k+1] = [r * np.cos(angle), r * np.sin(angle), 30.0]
        
    # Ring 2
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0
        x0[k+7] = [r * np.cos(angle), r * np.sin(angle), 30.0]

    # Function to calculate overlap penalty and side length
    def calculate_side_and_penalty(positions):
        # positions: (n, 3) [x, y, angle_deg]
        x = positions[:, 0]
        y = positions[:, 1]
        angles = positions[:, 2]
        
        # 1. Check for overlaps
        # Approximate hexagon as circle of radius r_hex = sqrt(3)/2 * 2? No.
        # Min distance between centers is sqrt(3).
        # We check all pairs.
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                # If dist < sqrt(3), they overlap (for same orientation)
                # For different orientations, it's complex. 
                # Assume small rotations or use conservative bound.
                # For SA, use simple distance penalty.
                if dist < np.sqrt(3) - eps:
                    penalty += (np.sqrt(3) - dist) ** 2 * 1000
        
        # 2. Calculate side length
        # Find max extent
        # For each hex, compute vertices
        # Simplified: Use max x/y of centers + max offset
        # Max offset for Pointy Top (angle 30 in our convention? Let's assume angle 0 is Pointy for calc ease, 
        # but we pass 30 to evaluator. 
        # Evaluator convention: Angle 30 -> Pointy Top? 
        # Previous code: angle 30, max_y=1, max_x=sqrt(3)/2. This is Pointy Top.
        # So we treat angle 30 as Pointy Top.
        
        max_x = 0
        max_y = 0
        
        for i in range(n_hex):
            cx = x[i]
            cy = y[i]
            # Pointy Top extents
            cx_max = cx + np.sqrt(3)/2
            cy_max = cy + 1.0
            cx_min = cx - np.sqrt(3)/2
            cy_min = cy - 1.0
            
            max_x = max(max_x, cx_max, -cx_min)
            max_y = max(max_y, cy_max, -cy_min)
            
        # Outer hexagon (Pointy Top) constraints:
        # 2*S >= 2*max_y => S >= max_y
        # sqrt(3)*S >= 2*max_x => S >= 2*max_x/sqrt(3)
        
        s1 = max_y
        s2 = 2 * max_x / np.sqrt(3)
        side = max(s1, s2)
        
        return side, penalty

    # Optimization
    best_positions = x0.copy()
    best_side, best_penalty = calculate_side_and_penalty(x0)
    best_score = best_side + best_penalty
    
    # Simple SA loop
    current_positions = x0.copy()
    current_side, current_penalty = calculate_side_and_penalty(current_positions)
    current_score = current_side + current_penalty
    
    T = 1.0
    step = 0.5
    
    time_limit = 30.0 # Run for 30s
    
    while time.time() - start_time < time_limit:
        # Generate new position
        new_positions = current_positions.copy()
        idx = np.random.randint(n_hex)
        new_positions[idx, 0] += np.random.uniform(-step, step)
        new_positions[idx, 1] += np.random.uniform(-step, step)
        # Keep angle fixed for now to simplify? 
        # Or allow rotation?
        # new_positions[idx, 2] += np.random.uniform(-10, 10)
        
        new_side, new_penalty = calculate_side_and_penalty(new_positions)
        new_score = new_side + new_penalty
        
        if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / T):
            current_positions = new_positions
            current_side = new_side
            current_penalty = new_penalty
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_positions = current_positions.copy()
                best_side = current_side
                
        T *= 0.99
        if T < 0.01:
            T = 0.01
        
        # Adaptive step
        if best_score < 3.99:
             step *= 0.95

    # Final Polish: Local optimization on best_positions
    # Use scipy minimize on the penalty function
    
    def objective(pos):
        s, p = calculate_side_and_penalty(pos)
        return s + p*1000

    res = minimize(objective, best_positions, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if res.fun < best_score:
        best_positions = res.x
        best_side, _ = calculate_side_and_penalty(res.x)

    # Construct return value
    inner = []
    for i in range(n_hex):
        inner.append([best_positions[i, 0], best_positions[i, 1], 30.0])
    
    center = (0.0, 0.0)
    side = best_side
    angle = 30.0
    
    return inner, center, side, angle
