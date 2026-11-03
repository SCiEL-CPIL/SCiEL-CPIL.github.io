import numpy as np
from scipy.optimize import minimize

N = 11

def get_inner_hex_vertices(cx, cy, angle):
    # Pointy-top hexagon at angle=0 has vertices at 30, 90, 150... degrees
    # Standard math angle: 0 is East.
    # Let's use angles: 30, 90, 150, 210, 270, 330 degrees for pointy top.
    # This corresponds to pi/6 + k*pi/3.
    base_angles = np.pi / 6 + np.pi / 3 * np.arange(6)
    angles = base_angles + angle
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def get_outer_hex_vertices(S, angle):
    # Outer hexagon side length S.
    # Circumradius R = S.
    # Vertices at angles: 0, 60, 120... for flat-top? Or 30, 90... for pointy-top?
    # The problem says "outer hexagon center, side length and rotation".
    # Usually, if inner are rotated, outer can be too.
    # Let's assume outer hexagon is defined by its vertices.
    # If rotation is 0, let's align it such that sides are horizontal/vertical?
    # A regular hexagon with side S has circumradius S.
    # Let's use the same orientation convention: pointy-top at angle=0.
    # Vertices at 30, 90, ...
    base_angles = np.pi / 6 + np.pi / 3 * np.arange(6)
    angles = base_angles + angle
    x = S * np.cos(angles)
    y = S * np.sin(angles)
    return np.column_stack((x, y))

def point_to_segment_distance(p, a, b):
    # p, a, b are (2,) arrays
    ab = b - a
    ap = p - a
    len_sq = np.dot(ab, ab)
    if len_sq == 0:
        return np.linalg.norm(ap)
    t = np.dot(ap, ab) / len_sq
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def min_distance_between_hexagons(verts1, verts2):
    # verts1, verts2 are (6, 2) arrays
    min_d = float('inf')
    # Check vertices of 1 to edges of 2
    for i in range(6):
        p = verts1[i]
        a = verts2[i]
        b = verts2[(i + 1) % 6]
        d = point_to_segment_distance(p, a, b)
        if d < min_d:
            min_d = d
    # Check vertices of 2 to edges of 1
    for i in range(6):
        p = verts2[i]
        a = verts1[i]
        b = verts1[(i + 1) % 6]
        d = point_to_segment_distance(p, a, b)
        if d < min_d:
            min_d = d
    return min_d

def min_distance_to_outer_boundary(verts, outer_verts):
    # outer_verts is (6, 2)
    min_d = float('inf')
    for i in range(6):
        p = verts[i]
        a = outer_verts[i]
        b = outer_verts[(i + 1) % 6]
        d = point_to_segment_distance(p, a, b)
        if d < min_d:
            min_d = d
    return min_d

def construct():
    # Variables: x1, y1, a1, ..., xN, yN, aN, S
    # Total 3*N + 1 variables
    
    def objective(params):
        S = params[-1]
        if S < 1.0: # Minimum possible is slightly more than 1
            S = 1.0
        
        # Extract inner hex params
        centers = params[0:2*N:3]
        cys = params[1:2*N+1:3]
        angles = params[2:2*N+2:3]
        outer_angle = 0.0 # We can fix outer angle to 0 due to symmetry, or optimize it. 
                          # Optimizing it adds 1 variable. Let's fix it to 0 to reduce complexity, 
                          # as rotating the whole system is equivalent.
        
        # Calculate vertices for all inner hexagons
        inner_verts = []
        for i in range(N):
            v = get_inner_hex_vertices(centers[i], cys[i], angles[i])
            inner_verts.append(v)
            
        outer_verts = get_outer_hex_vertices(S, outer_angle)
        
        penalty = 0.0
        clearance = 0.0
        
        # Check non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                d = min_distance_between_hexagons(inner_verts[i], inner_verts[j])
                if d < 0:
                    # Smooth penalty for violation
                    penalty += d * d
                else:
                    # We want to maximize minimum clearance, so we can add a small reward for clearance?
                    # No, objective is S. Constraints are d >= 0.
                    # If we use penalty method, we only penalize violations.
                    pass
                    
        # Check containment
        for i in range(N):
            d = min_distance_to_outer_boundary(inner_verts[i], outer_verts)
            if d < 0:
                penalty += d * d
        
        # Objective: Minimize S + Penalty
        # To encourage compactness, we can add a small term for sum of distances to center?
        # No, minimizing S naturally pulls them in.
        return S + 100.0 * penalty

    # Initial guesses
    # Start with a hexagonal lattice arrangement
    # Center at (0,0)
    # 1 center, 6 surrounding, 4 more?
    # Let's try a grid-like start or random starts
    
    best_S = float('inf')
    best_params = None
    
    # Multi-start
    num_starts = 5
    
    for start_idx in range(num_starts):
        if start_idx == 0:
            # Hexagonal lattice
            # 1 center
            # 6 neighbors at distance 2 (side 1 hexes touch at distance 2? No, distance between centers of touching hexes is 2 * inradius = sqrt(3)? 
            # Side 1 hexagon. Inradius = sqrt(3)/2.
            # Distance between centers of touching hexes = 2 * inradius = sqrt(3) ~ 1.732.
            # Let's use distance d = sqrt(3) + small epsilon.
            d = np.sqrt(3) + 0.1
            
            init_centers_x = [0.0]
            init_centers_y = [0.0]
            init_angles = [0.0]
            
            # 6 neighbors
            for k in range(6):
                ang = k * np.pi / 3
                init_centers_x.append(d * np.cos(ang))
                init_centers_y.append(d * np.sin(ang))
                init_angles.append(0.0)
                
            # 4 more to make 11?
            # Place them further out?
            # Or just start with 11 random-ish points in a circle
            # Let's add 4 more at radius 2*d
            for k in range(4):
                ang = (k + 0.5) * np.pi / 2 # 45, 135, 225, 315
                init_centers_x.append(2.2 * d * np.cos(ang))
                init_centers_y.append(2.2 * d * np.sin(ang))
                init_angles.append(0.0)
                
            x0 = np.zeros(3 * N + 1)
            for i in range(N):
                x0[3*i] = init_centers_x[i]
                x0[3*i+1] = init_centers_y[i]
                x0[3*i+2] = init_angles[i]
            x0[-1] = 5.0 # Initial S
        else:
            # Random start
            np.random.seed(start_idx + 100)
            x0 = np.zeros(3 * N + 1)
            for i in range(N):
                # Random position in a circle of radius 3
                r = np.sqrt(np.random.random()) * 3
                theta = np.random.random() * 2 * np.pi
                x0[3*i] = r * np.cos(theta)
                x0[3*i+1] = r * np.sin(theta)
                x0[3*i+2] = np.random.random() * np.pi
            x0[-1] = 5.0

        # Optimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8})
            if res.success or res.fun < best_S:
                # Check if solution is valid (low penalty)
                S_val = res.x[-1]
                # Re-evaluate penalty to ensure validity
                # If penalty is high, it's not a valid packing
                # We can extract penalty from res.fun - S_val
                penalty_val = (res.fun - S_val) / 100.0
                if penalty_val < 1e-3: # Valid solution
                    if S_val < best_S:
                        best_S = S_val
                        best_params = res.x
        except Exception as e:
            pass

    if best_params is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct return values
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        ang = best_params[3*i+2]
        inner.append([x, y, np.degrees(ang)])
    
    outer_center = (0.0, 0.0)
    outer_side = best_params[-1]
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# The task requires keeping the function names and signatures.
# The seed program has `construct()` inside the block.
# I will place the entire logic inside the block to be safe, or just `construct` if imports are allowed outside.
# The prompt says "Keep the function names and signatures... of the seed program".
# The seed program defines N and construct.
# I will put imports and helper functions outside the block if possible, or inside if the block is the only place to edit.
# "Rewrite the program from scratch... Output the COMPLETE program".
# So I can structure the file however I want, as long as `construct()` exists and returns the right tuple.
# EVOLVE-BLOCK-END
