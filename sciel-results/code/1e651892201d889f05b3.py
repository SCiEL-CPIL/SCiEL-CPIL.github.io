import numpy as np
import time
import itertools

N = 11

def get_hex_vertices(cx, cy, rot):
    """
    Get vertices of a regular hexagon with side length 1.
    Center (cx, cy), rotation rot (radians).
    """
    # Vertices of a hexagon with side 1 centered at origin, flat-top
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    angles = np.arange(6) * np.pi / 3 + rot
    # Radius = Side length for regular hexagon
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def objective(params):
    """
    params: array of shape (N*3, )
    First N*3 elements: for each hex i, (x, y, rot)
    """
    # Unpack
    configs = params.reshape(N, 3)
    
    # Collect all vertices
    # This is a bit slow in Python loop, but N=11 is small.
    all_vertices = []
    for i in range(N):
        x, y, rot = configs[i]
        verts = get_hex_vertices(x, y, rot)
        all_vertices.append(verts)
    
    # Concatenate
    all_vertices = np.vstack(all_vertices)
    
    # Outer hexagon is centered at (0,0), rotation 0.
    # Normals for flat-top hexagon (vertices at 0, 60, ...):
    # Edges are perpendicular to normals at 30, 90, 150, ...
    # Actually, if vertices are at 0, 60, ..., the edges are at 30, 90, ...
    # Let's check: Vertex at (1,0). Edge between (1,0) and (0.5, 0.866).
    # Midpoint (0.75, 0.433). Normal vector (1, 0.577) -> angle 30 deg.
    # So normals are at 30 + k*60 degrees.
    normal_angles = np.arange(6) * np.pi / 3 + np.pi / 6
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # Project all vertices onto normals
    # projections: (6, M)
    projections = normals @ all_vertices.T
    
    # Max absolute projection
    max_proj = np.max(np.abs(projections))
    
    # Side length S = max_proj / apothem_factor
    # Apothem = S * sqrt(3) / 2
    # So S = max_proj * 2 / sqrt(3)
    S = max_proj * 2 / np.sqrt(3)
    
    return S

def initial_config():
    """
    Generate a compact initial configuration.
    """
    # Axial coordinates for 11 hexes
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 partial: (2,0), (2,-1), (1,1), (-1,2)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, 1), (-1, 2)
    ]
    
    configs = []
    for q, r in axial_coords:
        # Convert axial to cartesian
        # x = 1.5 * q * side
        # y = sqrt(3)/2 * q * side + sqrt(3) * r * side
        # side = 1
        x = 1.5 * q
        y = np.sqrt(3)/2 * q + np.sqrt(3) * r
        rot = 0.0
        configs.append([x, y, rot])
        
    return np.array(configs).flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # Initial config
    x0 = initial_config()
    
    # We will try a few random restarts with small perturbations
    best_score = objective(x0)
    best_x = x0.copy()
    
    # Use scipy minimize
    from scipy.optimize import minimize
    
    # Try Nelder-Mead
    # We need to be careful with time.
    # N=11, 33 variables.
    
    restarts = 5
    for i in range(restarts):
        if time.time() > deadline:
            break
            
        # Add noise for restarts > 0
        if i == 0:
            x_start = x0
        else:
            x_start = x0 + np.random.normal(0, 0.5, size=x0.shape)
            
        # Optimize
        try:
            res = minimize(objective, x_start, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    # Construct output
    # inner: list of [x, y, angle_degrees]
    inner = []
    for i in range(N):
        x, y, rot = best_x[i*3], best_x[i*3+1], best_x[i*3+2]
        angle_deg = np.degrees(rot)
        inner.append([x, y, angle_deg])
        
    # Outer hexagon
    center = [0.0, 0.0]
    side = best_score
    angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# The construct function is defined above, but the task requires the function name and signature.
# The current program structure has construct() inside the block or outside?
# The prompt says: "Keep the function names and signatures of the seed program."
# Seed program:
# def construct():
#     ...
#     return ...
# EVOLVE-BLOCK-END

# I will place the helper functions and the construct function inside the block or appropriately.
# The seed program has N=11 at top level.
# I will define the helpers and construct.

# To ensure compatibility with the "EVOLVE-BLOCK" structure if strictly enforced, 
# I'll put the main logic in construct.

# Re-defining to be safe with the block structure if the evaluator wraps it.
# But the prompt says "Output the COMPLETE program".

# Let's refine the objective to be faster.
# Pre-compute normals.
_NORMALS = np.column_stack((np.cos(np.arange(6) * np.pi / 3 + np.pi / 6), 
                            np.sin(np.arange(6) * np.pi / 3 + np.pi / 6)))

def _objective_fast(params):
    configs = params.reshape(N, 3)
    # Vectorized vertex calculation?
    # x = cx + cos(rot + k*pi/3)
    # y = cy + sin(rot + k*pi/3)
    # This is hard to fully vectorize for N hexes * 6 verts without memory overhead, but N=11 is tiny.
    # Loop is fine.
    
    max_abs_proj = 0.0
    for i in range(N):
        cx, cy, rot = configs[i]
        # Vertices
        angles = rot + np.arange(6) * np.pi / 3
        vx = cx + np.cos(angles)
        vy = cy + np.sin(angles)
        
        # Projections
        # proj_k = nx_k * vx + ny_k * vy
        # We want max over k of |proj_k|
        # proj = _NORMALS @ [vx, vy] -> shape (6,)
        # But _NORMALS is (6,2).
        # So _NORMALS @ np.column_stack((vx, vy)).T -> (6, 6)
        # This is 6x6 matrix mult.
        # Or just dot products.
        
        # Let's do it manually for speed
        # _NORMALS[k] is (nx, ny)
        for k in range(6):
            nx, ny = _NORMALS[k]
            projs = nx * vx + ny * vy
            max_abs_proj = max(max_abs_proj, np.max(np.abs(projs)))
            
    return max_abs_proj * 2 / np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    x0 = initial_config()
    
    best_score = _objective_fast(x0)
    best_x = x0.copy()
    
    from scipy.optimize import minimize
    
    # Try a few restarts
    for i in range(10):
        if time.time() > deadline - 5.0:
            break
            
        if i == 0:
            x_start = x0
        else:
            x_start = x0 + np.random.normal(0, 0.2, size=x0.shape)
            
        try:
            res = minimize(_objective_fast, x_start, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            pass

    inner = []
    for i in range(N):
        x, y, rot = best_x[i*3], best_x[i*3+1], best_x[i*3+2]
        angle_deg = np.degrees(rot)
        inner.append([x, y, angle_deg])
        
    center = [0.0, 0.0]
    side = best_score
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
