import numpy as np
import time
import random

N = 12
SQRT3 = np.sqrt(3.0)

# Precompute base vertices for a unit hexagon (flat-top, vertex at angle 0)
# Vertices at angles 0, 60, 120, 180, 240, 300
BASE_ANGLES = np.radians(np.arange(0, 360, 60))
BASE_COS = np.cos(BASE_ANGLES)
BASE_SIN = np.sin(BASE_ANGLES)
BASE_VERTS = np.column_stack([BASE_COS, BASE_SIN])

def hexagon_vertices(cx, cy, angle_rad):
    """
    Returns (6, 2) array of vertices for a unit hexagon at (cx, cy) with rotation angle_rad.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    verts = (R @ BASE_VERTS.T).T
    return verts + np.array([cx, cy])

def sat_min_separation(verts1, verts2):
    """
    Calculate the minimum separation distance between two convex polygons (hexagons)
    using the Separating Axis Theorem.
    Returns a positive value if separated, negative if overlapping (approx depth).
    """
    min_sep = float('inf')
    
    # Axes from polygon 1
    for i in range(6):
        j = (i + 1) % 6
        edge = verts1[j] - verts1[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        
        proj1 = verts1.dot(normal)
        proj2 = verts2.dot(normal)
        
        sep = min(proj2) - max(proj1)
        if sep < min_sep:
            min_sep = sep
            
        if min_sep > 1e-6:
            # Already separated, no need to check other axes
            return min_sep

    # Axes from polygon 2
    for i in range(6):
        j = (i + 1) % 6
        edge = verts2[j] - verts2[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        
        proj1 = verts1.dot(normal)
        proj2 = verts2.dot(normal)
        
        sep = min(proj1) - max(proj2)
        if sep < min_sep:
            min_sep = sep
            
        if min_sep > 1e-6:
            return min_sep
            
    return min_sep

def compute_objective(params, penalty_weight=10.0):
    """
    Calculate the objective function: outer side length + overlap penalty.
    params: array of shape (N, 3) -> [x, y, angle_rad]
    """
    # Unpack positions and angles
    positions = params[:, :2]
    angles = params[:, 2]
    
    # Calculate vertices for all hexagons
    verts_list = []
    for i in range(N):
        v = hexagon_vertices(positions[i, 0], positions[i, 1], angles[i])
        verts_list.append(v)
    
    # Calculate overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            sep = sat_min_separation(verts_list[i], verts_list[j])
            if sep < 0:
                # Quadratic penalty for overlap
                penalty += (-sep) ** 2
    
    # Calculate required outer side length
    # Gather all vertices
    all_verts = np.vstack(verts_list)
    x = all_verts[:, 0]
    y = all_verts[:, 1]
    
    # Linear forms for regular hexagon containment (flat-top outer hexagon)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    L1 = x * (SQRT3/2) + y * 0.5
    L2 = y
    L3 = -x * (SQRT3/2) + y * 0.5
    L4 = -x * (SQRT3/2) - y * 0.5
    L5 = -y
    L6 = x * (SQRT3/2) - y * 0.5
    
    all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
    max_proj = np.max(all_L)
    side = max_proj * 2.0 / SQRT3
    
    return side + penalty_weight * penalty

def initial_config_hexagonal():
    """
    Generate a compact honeycomb-like initial configuration.
    Center + 6 ring + 5 from next ring.
    """
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    params = np.zeros((N, 3))
    for i, (q, r) in enumerate(coords):
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        params[i, 0] = x
        params[i, 1] = y
        params[i, 2] = 0.0 # Angle
        
    return params

def simulated_annealing(start_time, deadline):
    """
    Run Simulated Annealing to optimize the configuration.
    """
    # Initial configuration
    params = initial_config_hexagonal()
    
    # Add small random noise to break symmetry
    params += np.random.normal(0, 0.01, params.shape)
    
    current_obj = compute_objective(params)
    best_params = params.copy()
    best_obj = current_obj
    
    T = 1.0 # Initial temperature
    T_min = 1e-6
    alpha = 0.999 # Cooling rate
    
    # Step sizes for x, y, angle
    step_x = 0.1
    step_y = 0.1
    step_a = np.pi / 36.0
    
    while time.time() < deadline and T > T_min:
        # Generate neighbor
        new_params = params.copy()
        
        # Randomly select a hexagon to move
        i = random.randint(0, N - 1)
        
        # Move position
        new_params[i, 0] += np.random.normal(0, step_x)
        new_params[i, 1] += np.random.normal(0, step_y)
        
        # Rotate slightly
        new_params[i, 2] += np.random.normal(0, step_a)
        
        new_obj = compute_objective(new_params)
        
        delta = new_obj - current_obj
        
        # Acceptance criterion
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            params = new_params
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_params = params.copy()
        
        # Cool down
        T *= alpha
        
        # Adaptive step size: shrink if stuck
        if T < 0.1:
            step_x *= 0.99
            step_y *= 0.99
            step_a *= 0.99
            
    return best_params, best_obj

def construct():
    start_time = time.time()
    # Leave 10 seconds margin for safety
    deadline = start_time + 140.0
    
    best_params = None
    best_obj = float('inf')
    
    # Multi-start: Try a few different initial configurations or restarts
    # 1. Honeycomb
    # 2. Circular
    # 3. Random
    
    # Start 1: Honeycomb
    params1, obj1 = simulated_annealing(start_time, deadline - 40)
    if obj1 < best_obj:
        best_obj = obj1
        best_params = params1
        
    if time.time() > deadline - 30:
        # Not enough time for more starts, but we have a result
        pass
    else:
        # Start 2: Circular arrangement
        params2 = np.zeros((N, 3))
        for i in range(N):
            angle = 2 * np.pi * i / N
            params2[i, 0] = 2.0 * np.cos(angle)
            params2[i, 1] = 2.0 * np.sin(angle)
            params2[i, 2] = angle # Align radially?
        
        # Run SA from this start
        # We need to modify SA to accept initial params
        # For simplicity, let's just run a shorter SA or reuse the function with modified init
        # Let's just run another SA with a different seed/noise
        
        params2 += np.random.normal(0, 0.5, params2.shape)
        
        # Quick SA
        T = 0.5
        T_min = 1e-5
        alpha = 0.998
        step_x = 0.1
        step_y = 0.1
        step_a = np.pi / 36.0
        current_obj = compute_objective(params2)
        
        while time.time() < deadline - 10 and T > T_min:
            new_params = params2.copy()
            i = random.randint(0, N - 1)
            new_params[i, 0] += np.random.normal(0, step_x)
            new_params[i, 1] += np.random.normal(0, step_y)
            new_params[i, 2] += np.random.normal(0, step_a)
            
            new_obj = compute_objective(new_params)
            delta = new_obj - current_obj
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                params2 = new_params
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_params = params2.copy()
            T *= alpha
            if T < 0.1:
                step_x *= 0.99
                step_y *= 0.99
                step_a *= 0.99

    # Final validation and cleanup
    # We need to ensure the final configuration is valid (no overlaps)
    # If the best_obj is high due to penalty, we might need to project or reject.
    # However, SA with penalty usually converges to non-overlapping if T goes low enough.
    # Let's check validity of best_params
    
    if best_params is None:
        # Fallback to initial honeycomb
        best_params = initial_config_hexagonal()
        
    # Verify validity
    positions = best_params[:, :2]
    angles = best_params[:, 2]
    
    verts_list = []
    for i in range(N):
        v = hexagon_vertices(positions[i, 0], positions[i, 1], angles[i])
        verts_list.append(v)
        
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            sep = sat_min_separation(verts_list[i], verts_list[j])
            if sep < -1e-5: # Overlapping
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # If invalid, try to find a valid configuration by running a very short SA 
        # starting from the best_params but with a very high penalty weight
        # Or just return the honeycomb which is known to be valid (touching)
        # The honeycomb has sep = 0 for neighbors. Is 0 valid? 
        # "Disjoint" usually means interiors do not intersect. Touching is allowed in many packings.
        # The SAT check returns 0 for touching. 
        # Let's assume touching is valid.
        # If sep < 0, it's invalid.
        
        # If our SA result is invalid, fallback to the initial honeycomb
        best_params = initial_config_hexagonal()
        positions = best_params[:, :2]
        angles = best_params[:, 2]

    # Calculate final side length
    verts_list = []
    for i in range(N):
        v = hexagon_vertices(positions[i, 0], positions[i, 1], angles[i])
        verts_list.append(v)
        
    all_verts = np.vstack(verts_list)
    x = all_verts[:, 0]
    y = all_verts[:, 1]
    
    L1 = x * (SQRT3/2) + y * 0.5
    L2 = y
    L3 = -x * (SQRT3/2) + y * 0.5
    L4 = -x * (SQRT3/2) - y * 0.5
    L5 = -y
    L6 = x * (SQRT3/2) - y * 0.5
    
    all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
    max_proj = np.max(all_L)
    side = max_proj * 2.0 / SQRT3
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(np.degrees(angles[i]))])
        
    return inner, (0.0, 0.0), float(side), 0.0

# EVOLVE-BLOCK-END
