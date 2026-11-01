import numpy as np
from scipy.optimize import minimize

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    verts = np.array([[cx + side * np.cos(a + k * np.pi / 3),
                       cy + side * np.sin(a + k * np.pi / 3)] for k in range(6)])
    return verts

def get_normals(verts):
    normals = []
    for i in range(len(verts)):
        edge = verts[(i + 1) % len(verts)] - verts[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normals.append(normal / norm)
    return normals

def smooth_min(values, beta=50.0):
    # Smooth minimum of a list of values
    # -1/beta * log(sum(exp(-beta * v)))
    # To avoid overflow, shift by min
    vals = np.array(values)
    min_val = np.min(vals)
    return min_val - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (vals - min_val))))

def smooth_relu(x, beta=50.0):
    # 0.5 * (x + sqrt(x^2 + 1/beta^2))
    return 0.5 * (x + np.sqrt(x * x + 1.0 / (beta * beta)))

def smooth_penetration_depth(v1, v2, beta=50.0):
    normals = get_normals(v1) + get_normals(v2)
    overlaps = []
    for normal in normals:
        p1 = v1 @ normal
        p2 = v2 @ normal
        overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
        overlaps.append(overlap)
    min_overlap = smooth_min(overlaps, beta)
    return smooth_relu(min_overlap, beta)

def compute_outer_side(inner_verts_list, cx, cy):
    # Outer hexagon centered at (cx, cy), rotation 0
    # Normals are at 30, 90, 150, 210, 270, 330 degrees
    angles = np.radians(30 + np.arange(6) * 60)
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    max_proj = -np.inf
    for verts in inner_verts_list:
        # Translate vertices relative to outer center
        rel_verts = verts - np.array([cx, cy])
        # Project onto all normals
        projs = rel_verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
        
    # s * sqrt(3)/2 = max_proj
    return max_proj * 2.0 / np.sqrt(3)

def objective(params):
    # params: [x1,y1,a1, ..., x12,y12,a12, cx, cy]
    inner_params = params[:3 * N]
    cx, cy = params[3 * N:3 * N + 2]
    
    inner_verts_list = []
    for i in range(N):
        x, y, a = inner_params[3*i:3*i+3]
        verts = hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
        
    s = compute_outer_side(inner_verts_list, cx, cy)
    
    total_penalty = 0.0
    # Pairwise overlap with smooth penalty
    for i in range(N):
        for j in range(i+1, N):
            pd = smooth_penetration_depth(inner_verts_list[i], inner_verts_list[j])
            total_penalty += pd
            
    return s + 10.0 * total_penalty

def initial_guess():
    spacing = np.sqrt(3)
    positions = []
    # Center
    positions.append((0.0, 0.0))
    # Ring 1: 6 hexes
    for k in range(6):
        theta = k * np.pi / 3
        positions.append((spacing * np.cos(theta), spacing * np.sin(theta)))
    # Ring 2: 5 hexes, offset by 30 degrees
    for k in range(5):
        theta = k * 2 * np.pi / 5 + np.pi / 6
        positions.append((2 * spacing * np.cos(theta), 2 * spacing * np.sin(theta)))
        
    params = []
    for x, y in positions:
        params.extend([x, y, 0.0])
    # Outer center at 0,0
    params.extend([0.0, 0.0])
    return np.array(params)

def construct():
    x0 = initial_guess()
    # Use L-BFGS-B with smooth objective
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    params = result.x
    inner = []
    for i in range(N):
        x, y, a = params[3*i:3*i+3]
        inner.append([float(x), float(y), float(a)])
    cx, cy = params[3*N:3*N+2]
    # Calculate final s for return
    inner_verts_list = [hex_vertices(x, y, 1.0, a) for x, y, a in inner]
    s = compute_outer_side(inner_verts_list, cx, cy)
    return inner, (float(cx), float(cy)), float(s), 0.0
