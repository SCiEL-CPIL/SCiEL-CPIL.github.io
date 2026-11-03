import numpy as np
import time
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin before 120s limit
    
    # Unit hexagon vertices (centered at origin, flat side up? No, pointy top? 
    # Standard: vertices at angles 0, 60, 120...
    # If we use 0, 60... the width is 2 (vertex to vertex) and height is sqrt(3).
    # Let's use vertices at angles 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3.
    # This makes the "width" (x-extent) 2 and "height" (y-extent) sqrt(3).
    # This is a "pointy" hexagon horizontally? No, vertex at 0 is (1,0).
    # So it points left and right.
    hex_angles = np.arange(6) * (np.pi / 3)
    hex_verts = np.vstack((np.cos(hex_angles), np.sin(hex_angles))).T
    
    # Initial positions: Compact cluster
    # 1 center, 6 ring, 5 next ring
    # Distances: Ring 1 at sqrt(3), Ring 2 at ~2*sqrt(3) or similar
    sqrt3 = math.sqrt(3)
    
    # Generate a hexagonal grid of points
    # We will pick 12 points from a larger grid
    # Grid spacing is sqrt(3) in x, 1.5 in y?
    # Let's just place them manually in a tight circle
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Center
    positions[0] = [0.0, 0.0]
    
    # Ring 1 (6 hexes)
    for i in range(6):
        ang = i * (np.pi / 3)
        positions[i+1] = [sqrt3 * math.cos(ang), sqrt3 * math.sin(ang)]
        angles[i+1] = 0.0 # Aligned initially
        
    # Ring 2 (5 hexes) - Place them in the gaps or further out
    # To keep it compact, place them at distance ~2.598 (1.5 * sqrt(3)) ?
    # Or just push them out and let SA pull them in.
    # Let's place them at radius 2.5 * sqrt(3) / 2 ? No.
    # Let's place them at angles offset by 30 degrees from Ring 1, at radius ~2.5
    for i in range(5):
        ang = (i * 2 * np.pi / 5) + np.pi/6 # Offset
        r = 2.5 * sqrt3 / 2 # ~2.16
        # Actually, let's just put them further out to avoid initial overlap
        r = 2.0 * sqrt3
        positions[i+7] = [r * math.cos(ang), r * math.sin(ang)]
        angles[i+7] = 0.0
        
    # Outer hex rotation
    phi = 0.0
    
    best_state = (positions.copy(), angles.copy(), phi)
    best_score = float('inf')
    
    # SA Parameters
    T = 1.0
    cooling = 0.995
    min_T = 1e-4
    step_size = 0.5
    
    # Precompute outer normals base (without phi)
    # Outer hex normals are at angles phi, phi+60, ...
    
    def get_vertices(pos, ang):
        # Rotate hex_verts by ang
        c, s = math.cos(ang), math.sin(ang)
        R = np.array([[c, -s], [s, c]])
        verts = hex_verts @ R.T
        return verts + pos
    
    def evaluate(pos, ang, phi):
        # Calculate S_req
        # Outer normals
        n_angles = np.arange(6) * (np.pi / 3) + phi
        normals = np.vstack((np.cos(n_angles), np.sin(n_angles))).T
        
        max_dot = 0.0
        for i in range(N):
            verts = get_vertices(pos[i], ang[i])
            # Dot products of all 6 verts with all 6 normals
            # verts: 6x2, normals: 6x2
            dots = verts @ normals.T
            max_dot = max(max_dot, np.max(dots))
        
        S_req = (2.0 / sqrt3) * max_dot
        
        # Calculate Penalty for overlap
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < sqrt3:
                    penalty += (sqrt3 - d) ** 2
        
        return S_req + 10.0 * penalty, S_req

    current_pos = positions.copy()
    current_ang = angles.copy()
    current_phi = phi
    current_cost, current_S = evaluate(current_pos, current_ang, current_phi)
    
    if current_S < best_score:
        best_score = current_S
        best_state = (current_pos.copy(), current_ang.copy(), current_phi)
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
            
        # Perturb
        new_pos = current_pos.copy()
        new_ang = current_ang.copy()
        new_phi = current_phi
        
        # Randomly perturb a few positions
        for _ in range(2):
            idx = np.random.randint(N)
            new_pos[idx] += np.random.normal(0, step_size, 2)
            
        # Randomly perturb angles
        for _ in range(1):
            idx = np.random.randint(N)
            new_ang[idx] += np.random.normal(0, 0.1)
            
        # Perturb outer rotation
        new_phi += np.random.normal(0, 0.05)
        
        new_cost, new_S = evaluate(new_pos, new_ang, new_phi)
        
        # Acceptance
        delta = new_cost - current_cost
        if delta < 0 or np.random.rand() < math.exp(-delta / T):
            current_pos = new_pos
            current_ang = new_ang
            current_phi = new_phi
            current_cost = new_cost
            
            if new_S < best_score:
                best_score = new_S
                best_state = (current_pos.copy(), current_ang.copy(), current_phi)
        
        T *= cooling
        
        # Optional: Reduce step size slowly
        if T < 0.1:
            step_size = 0.1
        elif T < 0.01:
            step_size = 0.05

    # Final Polish: Optimize phi for the best state found
    best_pos, best_ang, best_phi = best_state
    
    # Fine tune phi
    for _ in range(100):
        min_S = float('inf')
        min_phi = best_phi
        for p in np.linspace(best_phi - 0.1, best_phi + 0.1, 20):
            _, S_val = evaluate(best_pos, best_ang, p)
            if S_val < min_S:
                min_S = S_val
                min_phi = p
        if min_phi == best_phi:
            break
        best_phi = min_phi

    # Extract final results
    inner = []
    for i in range(N):
        x, y = best_pos[i]
        ang_deg = math.degrees(best_ang[i]) % 360
        inner.append([float(x), float(y), float(ang_deg)])
    
    center = [0.0, 0.0]
    side = best_score
    angle_deg = math.degrees(best_phi) % 360
    
    return inner, center, float(side), float(angle_deg)
# EVOLVE-BLOCK-END
