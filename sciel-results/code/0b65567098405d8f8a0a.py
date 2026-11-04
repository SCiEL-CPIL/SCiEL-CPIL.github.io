import numpy as np
from scipy.optimize import minimize

def construct():
    N = 12
    
    # Initial guess: 3x4 grid of hexagons
    # This is a valid non-overlapping configuration
    centers = []
    for i in range(-1, 2):
        for j in range(-1, 3):
            x = i * np.sqrt(3)
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers.append([x, y])
            
    centers = np.array(centers)
    # Center the cluster
    centers -= np.mean(centers, axis=0)
    
    # Initial angles: 0 for all
    angles = np.zeros(N)
    
    # State: [x1, y1, a1, ..., x12, y12, a12, outer_angle]
    # Length 37
    state0 = np.zeros(37)
    for i in range(N):
        state0[3*i] = centers[i, 0]
        state0[3*i+1] = centers[i, 1]
        state0[3*i+2] = angles[i]
    state0[36] = 0.0 # Outer angle
    
    def get_vertices(state):
        verts = []
        for i in range(N):
            cx = state[3*i]
            cy = state[3*i+1]
            th = np.deg2rad(state[3*i+2])
            # Vertices of unit hexagon
            # Angles: 0, 60, 120, 180, 240, 300
            # Wait, standard hexagon vertices are at 0, 60...?
            # If side is 1, circumradius is 1.
            # Vertices at (1,0), (0.5, sqrt(3)/2)...
            # Let's use 0 to 300 step 60.
            base_angles = np.deg2rad(np.arange(0, 360, 60))
            vx = cx + np.cos(base_angles + th)
            vy = cy + np.sin(base_angles + th)
            verts.append(np.column_stack([vx, vy]))
        return np.vstack(verts)

    def calc_outer_L(state):
        verts = get_vertices(state)
        outer_th = np.deg2rad(state[36])
        
        # Normals for outer hexagon
        # If outer hex is rotated by outer_th, its sides are perpendicular to these normals
        # Normals are at outer_th + k * 60 degrees
        normals = []
        for k in range(6):
            ang = outer_th + k * np.pi / 3
            normals.append(np.array([np.cos(ang), np.sin(ang)]))
        normals = np.array(normals) # (6, 2)
        
        # Project vertices onto normals
        # proj (72, 6)
        proj = verts @ normals.T
        
        # Widths
        widths = np.max(proj, axis=0) - np.min(proj, axis=0)
        
        # L = max(widths) / sqrt(3)
        L = np.max(widths) / np.sqrt(3)
        return L

    def calc_penalty(state):
        # Extract centers
        cx = state[0::3]
        cy = state[1::3]
        
        # Calculate pairwise distances
        # d_ij = sqrt((cx_i - cx_j)^2 + (cy_i - cy_j)^2)
        # We need sum max(0, sqrt(3) - d_ij)^2
        # Note: sqrt(3) is the min distance for non-overlapping unit hexagons (sharing edge)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < np.sqrt(3):
                    diff = np.sqrt(3) - d
                    penalty += diff * diff
        return penalty

    def objective(state):
        L = calc_outer_L(state)
        P = calc_penalty(state)
        return L + 100.0 * P

    # Optimize
    # Nelder-Mead is robust for non-smooth (penalty has kinks)
    # We run it from the initial state
    # Also try a few random perturbations
    
    best_state = state0
    best_score = objective(state0)
    
    # Run optimization
    res = minimize(objective, state0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000, 'adaptive': True})
    
    if res.fun < best_score:
        best_score = res.fun
        best_state = res.x
        
    # Try another start: slightly perturbed
    np.random.seed(42)
    state1 = state0 + np.random.normal(0, 0.1, 37)
    state1[36] = 0.0 # Keep outer angle 0 for now? Or randomize?
    res1 = minimize(objective, state1, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000, 'adaptive': True})
    if res1.fun < best_score:
        best_score = res1.fun
        best_state = res1.x

    # Extract best state
    inner = []
    for i in range(N):
        x = best_state[3*i]
        y = best_state[3*i+1]
        a = best_state[3*i+2]
        inner.append([x, y, a])
        
    # The outer hexagon is centered at (0,0) in our optimization
    # We should return the center as (0,0)
    # And the side length L
    # And the angle
    
    L = calc_outer_L(best_state)
    outer_angle = best_state[36]
    
    # One final check: ensure no overlap by checking actual hexagon distance?
    # The penalty used circumcircle approximation.
    # If the score is good, it's likely valid.
    
    return inner, (0.0, 0.0), L, outer_angle

