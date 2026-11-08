import numpy as np
import time
from scipy.optimize import minimize

N = 11

def axial_to_cartesian(q, r):
    x = np.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def hex_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = np.deg2rad(angle_deg)
    # Vertices of a unit hexagon centered at (0,0) with angle 0 (pointy top? no, standard is flat top? 
    # Let's use standard math: vertex at angle 0 is (1,0).
    # This corresponds to a "pointy-right" hexagon.
    # The seed program used angle 0.
    # Let's ensure consistency.
    # If angle is 0, vertices are at 0, 60, 120...
    angles = np.deg2rad(np.arange(0, 360, 60) + angle_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def calculate_required_S(positions, angles):
    # positions: (N, 2)
    # angles: (N,)
    # Outer hexagon is centered at (0,0) with angle 0.
    # Normals to the sides of a hexagon with vertices at 0, 60... are at 30, 90...
    normals = np.deg2rad(np.arange(30, 360, 60))
    max_proj = 0.0
    for i in range(len(positions)):
        verts = hex_vertices(positions[i, 0], positions[i, 1], angles[i])
        # Project vertices onto normals
        projs = verts @ np.column_stack((np.cos(normals), np.sin(normals)))
        # The maximum projection distance from origin
        # The apothem a must be >= max_proj
        # S = 2 / sqrt(3) * a
        max_proj = max(max_proj, np.max(projs))
    return (2.0 / np.sqrt(3)) * max_proj

def objective(params):
    # params: 3N values: x1, y1, a1, x2, y2, a2, ...
    N_local = len(params) // 3
    positions = np.zeros((N_local, 2))
    angles = np.zeros(N_local)
    for i in range(N_local):
        positions[i, 0] = params[3*i]
        positions[i, 1] = params[3*i+1]
        angles[i] = params[3*i+2]
    
    # Calculate S
    S = calculate_required_S(positions, angles)
    
    # Calculate overlap penalty
    penalty = 0.0
    min_dist = np.sqrt(3)
    for i in range(N_local):
        for j in range(i+1, N_local):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    return S + 1000.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit, 20s margin
    
    # Initial layout: 7 hexagons in center ring, 4 in outer ring (cross shape)
    # Axial coords
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    
    initial_params = []
    for q, r in axial_coords:
        x, y = axial_to_cartesian(q, r)
        initial_params.extend([x, y, 0.0])
        
    initial_params = np.array(initial_params)
    
    best_params = initial_params
    best_score = objective(initial_params)
    
    # Try Nelder-Mead
    # We need to check time in the callback or just run with a maxiter limit
    # Nelder-Mead doesn't have a strict time limit, so we monitor time externally if possible, 
    # or just set a reasonable maxiter.
    # Given N=11, 33 variables. Nelder-Mead might be slow.
    # Let's try a few restarts or just one good run.
    
    # To respect the deadline, we can run minimize in a loop with a time check?
    # Scipy minimize doesn't easily support time-based stopping.
    # We can use `maxiter` to approximate.
    # Or we can just run it and hope it finishes.
    # With 33 variables, it might take a while.
    # Let's reduce variables? 
    # The angles are likely 0. We can fix angles to 0 initially and see if it helps.
    # If angles are 0, we have 22 variables.
    
    # Let's try fixing angles to 0 first.
    initial_params_fixed_angle = []
    for q, r in axial_coords:
        x, y = axial_to_cartesian(q, r)
        initial_params_fixed_angle.extend([x, y, 0.0])
        
    # Actually, let's just run the full optimization but with a small maxiter.
    # Or use L-BFGS-B? No, non-smooth (max function). Nelder-Mead is better for this.
    
    # To ensure we don't timeout, we can run multiple short optimizations.
    # But for now, let's just run one with a reasonable maxiter.
    # 33 variables, Nelder-Mead complexity is high.
    # Let's try to optimize only positions (fix angles to 0).
    # This is a good heuristic.
    
    params_pos_only = initial_params[:22] # Just x,y
    # We need a wrapper for pos-only objective
    
    def objective_pos_only(params):
        positions = params.reshape(-1, 2)
        angles = np.zeros(len(positions))
        return objective(np.concatenate([positions.flatten(), angles]))
        
    res = minimize(objective_pos_only, params_pos_only, method='Nelder-Mead', 
                   options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if time.time() > deadline:
        # Return best so far
        pass
    else:
        if res.fun < best_score:
            # Reconstruct full params
            new_positions = res.x.reshape(-1, 2)
            new_angles = np.zeros(N)
            new_params = np.concatenate([new_positions.flatten(), new_angles])
            best_params = new_params
            best_score = res.fun
            
    # One more try with angles free? Might be too slow.
    # Let's stick to the position-optimized solution.
    
    # Extract final positions and angles
    positions = best_params[0::3].reshape(-1, 2)
    angles = best_params[1::3]
    
    # Calculate final S
    final_S = calculate_required_S(positions, angles)
    
    # Construct return format
    # inner: list of [x, y, angle]
    # center: (0,0)
    # side: final_S
    # angle: 0.0
    
    inner = []
    for i in range(N):
        inner.append([positions[i, 0], positions[i, 1], angles[i]])
        
    return inner, (0.0, 0.0), final_S, 0.0

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END

# We need to put the implementation inside the block or reference it.
# The prompt says "Keep the function names and signatures of the seed program".
# The seed program has `construct`.
# I will put the helper functions and the main logic inside the EVOLVE-BLOCK or before it?
# "Rewrite the program from scratch... Output the COMPLETE program".
# The EVOLVE-BLOCK is just a marker. I can define helpers outside.

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit, 20s margin
    
    # Initial layout: 7 hexagons in center ring, 4 in outer ring (cross shape)
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    
    initial_params = []
    for q, r in axial_coords:
        x, y = axial_to_cartesian(q, r)
        initial_params.extend([x, y, 0.0])
        
    initial_params = np.array(initial_params)
    
    best_params = initial_params
    best_score = objective(initial_params)
    
    # Optimize positions only (angles fixed to 0)
    params_pos_only = initial_params[0::3].copy() # x
    # Wait, initial_params is x,y,a, x,y,a...
    # So indices 0, 3, 6... are x. 1, 4, 7... are y.
    # Let's just pass the full vector but fix angles in the objective?
    # No, simpler to extract x,y.
    
    x_coords = initial_params[0::3]
    y_coords = initial_params[1::3]
    pos_flat = np.concatenate([x_coords, y_coords]) # No, this mixes them.
    # Let's keep it as a flat vector of 2N coords.
    pos_vec = np.zeros(2*N)
    for i in range(N):
        pos_vec[2*i] = initial_params[3*i]
        pos_vec[2*i+1] = initial_params[3*i+1]
        
    def objective_pos_only(params):
        positions = params.reshape(-1, 2)
        angles = np.zeros(N)
        # Reconstruct full params for the general objective
        full_params = np.zeros(3*N)
        for i in range(N):
            full_params[3*i] = positions[i, 0]
            full_params[3*i+1] = positions[i, 1]
            full_params[3*i+2] = 0.0
        return objective(full_params)
        
    try:
        res = minimize(objective_pos_only, pos_vec, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if time.time() < deadline:
            if res.fun < best_score:
                new_positions = res.x.reshape(-1, 2)
                new_angles = np.zeros(N)
                new_params = np.zeros(3*N)
                for i in range(N):
                    new_params[3*i] = new_positions[i, 0]
                    new_params[3*i+1] = new_positions[i, 1]
                    new_params[3*i+2] = 0.0
                best_params = new_params
                best_score = res.fun
    except Exception:
        pass
        
    # Extract final positions and angles
    positions = best_params[0::3].reshape(-1, 2)
    angles = best_params[1::3]
    
    # Calculate final S
    final_S = calculate_required_S(positions, angles)
    
    # Construct return format
    inner = []
    for i in range(N):
        inner.append([positions[i, 0], positions[i, 1], angles[i]])
        
    return inner, (0.0, 0.0), final_S, 0.0
