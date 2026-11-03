import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    # Hexagon geometry
    # Inner hexagon side length 1
    # Vertices of a unit hexagon with rotation 0
    # Angle 0: vertex at (1, 0)
    base_verts = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])
    
    # Core centers: 1 center + 6 neighbors
    # Distance between centers in honeycomb is SQRT3
    d = SQRT3
    core_centers = np.array([[0.0, 0.0]])
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        core_centers = np.vstack([core_centers, [[x, y]]])
        
    # Core rotations: all 0
    core_rots = np.zeros(7)
    
    # Ring 2 centers
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (SQRT3 / 2.0) * r
                ring2_centers.append([x, y])
    ring2_centers = np.array(ring2_centers)
    
    # Objective function
    # vars: [x7, y7, x8, y8, x9, y9, x10, y10, phi7, phi8, phi9, phi10, theta]
    # 8 pos + 4 rot + 1 theta = 13 vars
    
    def get_max_projection(centers, rots, theta):
        # centers: (11, 2)
        # rots: (11,)
        # theta: scalar
        
        # Generate all vertices
        # For each hex i, vertices are centers[i] + R(rots[i]) @ base_verts
        # R(phi) = [[cos, -sin], [sin, cos]]
        
        # We can compute this efficiently
        # But 11 hexagons is small, so simple loop or vectorized stack is fine.
        
        all_verts = []
        for i in range(11):
            cx, cy = centers[i]
            phi = rots[i]
            c, s = math.cos(phi), math.sin(phi)
            # Rotate base_verts
            # v_x' = c*v_x - s*v_y
            # v_y' = s*v_x + c*v_y
            vx = base_verts[:, 0]
            vy = base_verts[:, 1]
            new_vx = c * vx - s * vy
            new_vy = s * vx + c * vy
            verts = np.column_stack([cx + new_vx, cy + new_vy])
            all_verts.append(verts)
            
        all_verts = np.vstack(all_verts) # (66, 2)
        
        # Container normals
        # Normals at theta + k*pi/3
        k_angles = np.arange(6) * math.pi / 3
        normal_angles = theta + k_angles
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)]) # (6, 2)
        
        # Dot products: (66, 2) @ (2, 6) -> (66, 6)
        dots = all_verts @ normals.T
        abs_dots = np.abs(dots)
        
        # Max over points and normals
        max_proj = np.max(abs_dots)
        
        # S = 2/sqrt(3) * max_proj
        return (2.0 / SQRT3) * max_proj

    def objective(vars):
        # Check time
        if time.time() > deadline - safety_margin:
            return 1e9
            
        # Unpack
        pos = vars[:8].reshape(4, 2)
        rots_outer = vars[8:12]
        theta = vars[12]
        
        # Construct full centers and rots
        # Core is fixed
        centers = np.vstack([core_centers, pos])
        rots = np.concatenate([core_rots, rots_outer])
        
        # Check overlaps
        # We need to check distance between all pairs of centers
        # If any distance < SQRT3 - epsilon, penalize
        # 11 centers -> 55 pairs
        # This is fast
        
        min_dist = SQRT3
        # Optimization: only check outer-outer and outer-core?
        # Core-core is fixed and valid.
        # So check:
        # 1. Outer-Outer (4 choose 2 = 6 pairs)
        # 2. Outer-Core (4 * 7 = 28 pairs)
        
        # Outer-Outer
        for i in range(4):
            for j in range(i+1, 4):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < (min_dist - 1e-5)**2:
                    # Penalty
                    return 1e6 + (min_dist - math.sqrt(dist_sq))**2
                    
        # Outer-Core
        for i in range(4):
            for j in range(7):
                dx = pos[i, 0] - core_centers[j, 0]
                dy = pos[i, 1] - core_centers[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < (min_dist - 1e-5)**2:
                    return 1e6 + (min_dist - math.sqrt(dist_sq))**2

        # Calculate S
        S = get_max_projection(centers, rots, theta)
        return S

    # Initializations
    # We will try a few different sets of 4 ring-2 hexagons
    # and slightly perturb them.
    
    best_S = float('inf')
    best_vars = None
    
    # Generate a list of initial guesses
    # 1. Consecutive 4: 0, 1, 2, 3
    # 2. 0, 1, 2, 4
    # 3. 0, 1, 3, 4
    # 4. 0, 2, 4, 6 (symmetric pairs)
    # 5. 0, 1, 5, 6
    # 6. Random 4
    
    init_sets = [
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 2, 4, 6],
        [0, 1, 5, 6],
        [0, 3, 6, 9],
        [0, 1, 2, 8],
        [0, 2, 5, 7],
        [0, 1, 4, 5],
        [0, 2, 3, 4]
    ]
    
    for idx_set in init_sets:
        if time.time() > deadline - safety_margin:
            break
            
        # Initial positions
        init_pos = ring2_centers[idx_set]
        # Add small noise to break symmetry and help optimizer
        noise = np.random.normal(0, 0.1, init_pos.shape)
        init_pos = init_pos + noise
        
        # Initial rotations: 0
        init_rots = np.zeros(4)
        
        # Initial theta: 0
        init_theta = 0.0
        
        init_vars = np.concatenate([init_pos.flatten(), init_rots, [init_theta]])
        
        # Run optimizer
        # Nelder-Mead
        # maxiter: enough to converge
        # tol: tight
        try:
            res = minimize(objective, init_vars, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < best_S:
                best_S = res.fun
                best_vars = res.x
        except Exception as e:
            pass

    # If no valid config found, fallback
    if best_vars is None:
        # Return a valid but large config
        # Just use the seed logic or a large S
        inner = [[0.0, 0.0, 0.0]]
        for i in range(6):
            angle = i * math.pi / 3
            x = d * math.cos(angle)
            y = d * math.sin(angle)
            inner.append([x, y, 0.0])
        # Add 4 more far away? No, just return a large S
        # Actually, we can just return the seed result if this fails, but let's make a valid one
        # Using the first init_set without optimization
        pos = ring2_centers[[0, 1, 2, 3]]
        centers = np.vstack([core_centers, pos])
        rots = np.concatenate([core_rots, np.zeros(4)])
        S = get_max_projection(centers, rots, 0.0)
        inner = []
        for i in range(11):
            inner.append([centers[i, 0], centers[i, 1], rots[i]])
        return inner, (0.0, 0.0), S, 0.0

    # Extract best result
    pos = best_vars[:8].reshape(4, 2)
    rots_outer = best_vars[8:12]
    theta = best_vars[12]
    
    centers = np.vstack([core_centers, pos])
    rots = np.concatenate([core_rots, rots_outer])
    
    # Final S calculation
    final_S = get_max_projection(centers, rots, theta)
    
    inner = []
    for i in range(11):
        inner.append([centers[i, 0], centers[i, 1], math.degrees(rots[i])])
        
    return inner, (0.0, 0.0), final_S, math.degrees(theta)

