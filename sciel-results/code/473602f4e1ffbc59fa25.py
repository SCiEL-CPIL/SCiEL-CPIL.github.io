import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# Pre-compute unit hexagon vertices (angle 0)
# Vertices at angles 0, 60, 120, 180, 240, 300
# Wait, standard orientation usually has flat sides at top/bottom or pointy.
# The problem says "regular hexagons". The orientation is a variable.
# Let's define base vertices for a hexagon centered at (0,0) with angle 0.
# Angle 0: Vertex at (1, 0).
BASE_VERTS = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))]
    for a in range(0, 360, 60)
])

# Normal axes for a hexagon with angle 0
# Normals are perpendicular to sides.
# Side between v0 and v1 is at angle 30. Normal is at 30+90=120?
# Let's just compute normals from vertices.
# Normal to side (v_i, v_{i+1}) is perpendicular to (v_{i+1}-v_i).
# For regular hexagon, normals are at angles 30, 90, 150, 210, 270, 330.
BASE_NORMALS = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))]
    for a in range(30, 360, 60)
])

def get_rot_matrix(angle_deg):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s], [s, c]])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Function to get vertices and normals for a hexagon given center and angle
    def get_hex_data(cx, cy, angle_deg):
        R = get_rot_matrix(angle_deg)
        verts = BASE_VERTS @ R.T + np.array([cx, cy])
        normals = BASE_NORMALS @ R.T
        return verts, normals

    # Evaluate the objective function
    # params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, theta_outer]
    # a_i in degrees, theta_outer in degrees
    def objective(params):
        if time.time() > deadline:
            return 1e9
            
        centers = params[:N*3].reshape(N, 3)
        theta_outer = params[N*3]
        
        # Calculate vertices and normals for all inner hexagons
        verts_list = []
        normals_list = []
        for i in range(N):
            cx, cy, ai = centers[i]
            v, n = get_hex_data(cx, cy, ai)
            verts_list.append(v)
            normals_list.append(n)
        
        # 1. Calculate Outer Hexagon Side Length
        # L = (2/sqrt(3)) * max_k max_i max_v |v . n_k(theta_outer)|
        # n_k(theta_outer) are the 6 normals of the outer hexagon
        R_outer = get_rot_matrix(theta_outer)
        outer_normals = BASE_NORMALS @ R_outer.T
        
        max_proj = 0.0
        for k in range(6):
            n_k = outer_normals[k]
            # Project all vertices of all hexagons onto n_k
            # verts_list[i] is (6,2). We want |dot(v, n_k)|
            # dot(v, n_k) = v @ n_k
            # We want max over all v
            for i in range(N):
                projs = np.abs(verts_list[i] @ n_k)
                m = np.max(projs)
                if m > max_proj:
                    max_proj = m
        
        side_length = (2.0 / SQRT3) * max_proj
        
        # 2. Calculate Overlap Penalty
        # For each pair (i, j), check SAT
        # We use a smooth approximation of the minimum overlap.
        # If min_overlap > 0, they are separated.
        # We penalize max(0, -min_overlap)^2
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Axes: normals of i and j
                axes = np.vstack([normals_list[i], normals_list[j]])
                
                # Projections of i and j onto all axes
                # proj_i: (12, 6) -> min/max per axis
                # verts_list[i] is (6, 2)
                # dots_i = verts_list[i] @ axes.T -> (6, 12)
                dots_i = verts_list[i] @ axes.T
                min_i = np.min(dots_i, axis=0)
                max_i = np.max(dots_i, axis=0)
                
                dots_j = verts_list[j] @ axes.T
                min_j = np.min(dots_j, axis=0)
                max_j = np.max(dots_j, axis=0)
                
                # Overlap length on each axis
                # overlap = min(max_i, max_j) - max(min_i, min_j)
                overlap = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
                
                # If overlap > 0 for all axes, they are disjoint.
                # We want min(overlap) to be >= 0.
                # If min(overlap) < 0, they overlap.
                # Use soft-min to make it smooth
                # soft_min(x) = -logsumexp(-x) / beta
                # For large beta, it approaches min.
                # We want to penalize if min(overlap) is negative.
                # Let d = min(overlap).
                # Penalty = max(0, -d)^2.
                
                # Approximate min(overlap)
                beta = 10.0
                # To avoid numerical issues with large negative values in exp
                # shift overlap
                max_ov = np.max(overlap)
                # If max_ov is very negative, they are deeply overlapping.
                # logsumexp(-overlap) = log(sum(exp(-overlap)))
                # We want -logsumexp(-overlap)
                # = - (log(sum(exp(-overlap - max_ov))) + max_ov)
                # = -max_ov - log(sum(exp(-overlap - max_ov)))
                
                exp_terms = np.exp(-(overlap - max_ov))
                log_sum = np.log(np.sum(exp_terms))
                soft_min_overlap = -max_ov - log_sum
                
                # If soft_min_overlap < 0, penalty
                if soft_min_overlap < 0:
                    penalty += (-soft_min_overlap) ** 2
                else:
                    # Small gradient to push them apart? No, 0 is fine.
                    pass

        return side_length + 10.0 * penalty

    # Initial configurations
    def get_initial_configs():
        configs = []
        # 1. Compact 3-5-3
        coords1 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
        configs.append(coords1)
        
        # 2. Flower
        coords2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
        configs.append(coords2)
        
        # 3. Spiral
        coords3 = [(0,0), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1), (2,0), (2,1)]
        configs.append(coords3)
        
        # 4. Random
        random.seed(42)
        coords4 = []
        placed = []
        for _ in range(50):
            if len(coords4) >= N: break
            q = random.randint(-3, 3)
            r = random.randint(-3, 3)
            if (q, r) in placed: continue
            x = SQRT3 * (q + r/2.0)
            y = 1.5 * r
            valid = True
            for (x_old, y_old) in placed:
                if math.sqrt((x-x_old)**2 + (y-y_old)**2) < SQRT3 - 0.1:
                    valid = False
                    break
            if valid:
                placed.append((x, y))
                coords4.append((x, y))
        if len(coords4) == N:
            configs.append(coords4)
            
        return configs

    best_score = float('inf')
    best_params = None
    
    configs = get_initial_configs()
    
    for config in configs:
        if time.time() > deadline - 10.0:
            break
            
        # Initialize params
        # Centers from config, angles 0, outer angle 0
        init_params = []
        for x, y in config:
            init_params.extend([x, y, 0.0])
        init_params.append(0.0) # outer angle
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # If no good result, use the first config as fallback
    if best_params is None:
        best_params = np.array([0.0, 0.0, 0.0] * N + [0.0])
        # Actually, let's just use the first config's coordinates
        config = configs[0]
        best_params = []
        for x, y in config:
            best_params.extend([x, y, 0.0])
        best_params.append(0.0)
        best_params = np.array(best_params)

    # Extract final result
    centers = best_params[:N*3].reshape(N, 3)
    theta_outer = best_params[N*3]
    
    # Center the configuration at (0,0)
    cx_avg = np.mean(centers[:, 0])
    cy_avg = np.mean(centers[:, 1])
    centers[:, 0] -= cx_avg
    centers[:, 1] -= cy_avg
    
    # Build return values
    inner = []
    for i in range(N):
        inner.append([centers[i, 0], centers[i, 1], centers[i, 2]])
    
    # Calculate final side length accurately
    # Re-use objective logic for final side
    verts_list = []
    for i in range(N):
        cx, cy, ai = centers[i]
        v, n = get_hex_data(cx, cy, ai)
        verts_list.append(v)
        
    R_outer = get_rot_matrix(theta_outer)
    outer_normals = BASE_NORMALS @ R_outer.T
    
    max_proj = 0.0
    for k in range(6):
        n_k = outer_normals[k]
        for i in range(N):
            projs = np.abs(verts_list[i] @ n_k)
            m = np.max(projs)
            if m > max_proj:
                max_proj = m
                
    final_side = (2.0 / SQRT3) * max_proj
    
    return (inner, (0.0, 0.0), final_side, theta_outer)

# EVOLVE-BLOCK-END
