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

def penetration_depth(v1, v2):
    normals = get_normals(v1) + get_normals(v2)
    min_overlap = np.inf
    for normal in normals:
        p1 = v1 @ normal
        p2 = v2 @ normal
        overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
        if overlap < min_overlap:
            min_overlap = overlap
    return min_overlap

def compute_outer_side(inner_verts_list):
    # Outer hexagon centered at (0, 0), rotation 0
    # Normals are at 30, 90, 150, 210, 270, 330 degrees
    angles = np.radians(30 + np.arange(6) * 60)
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    max_proj = 0.0
    for verts in inner_verts_list:
        # Project onto all normals
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
        
    # s * sqrt(3)/2 = max_proj
    return max_proj * 2.0 / np.sqrt(3)

def objective(params):
    # params: [x1,y1,a1, ..., x12,y12,a12]
    inner_verts_list = []
    for i in range(N):
        x, y, a = params[3*i:3*i+3]
        verts = hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
        
    s = compute_outer_side(inner_verts_list)
    
    total_penalty = 0.0
    # Pairwise overlap
    for i in range(N):
        for j in range(i+1, N):
            pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
            total_penalty += max(0.0, pd)
            
    return s + 10.0 * total_penalty

def initial_guess():
    # Create a triangular lattice cluster
    # 1 in center, 6 around, 5 more
    spacing = np.sqrt(3)
    positions = []
    # Center
    positions.append((0.0, 0.0))
    # Ring 1
    for k in range(6):
        theta = k * np.pi / 3
        positions.append((spacing * np.cos(theta), spacing * np.sin(theta)))
    # Ring 2 (only 5 needed, pick 5 of the 6 gaps)
    # The gaps are at distance 2*spacing from center, angles offset by 30 deg
    for k in range(5):
        theta = k * 2 * np.pi / 5 + np.pi / 6
        positions.append((2 * spacing * np.cos(theta), 2 * spacing * np.sin(theta)))
        
    params = []
    for x, y in positions:
        params.extend([x, y, 0.0])
    return np.array(params)

def construct():
    best_params = None
    best_score = np.inf
    
    # Multistart: try a few random perturbations
    base_guess = initial_guess()
    
    for attempt in range(5):
        if attempt == 0:
            x0 = base_guess
        else:
            # Add random noise
            noise = np.random.normal(0, 0.5, size=base_guess.shape)
            x0 = base_guess + noise
            
        # Add random rotation to initial guess
        if attempt > 0:
            rot_angle = np.random.uniform(0, 2*np.pi)
            # Rotate positions
            for i in range(N):
                x, y, a = x0[3*i:3*i+3]
                x_new = x * np.cos(rot_angle) - y * np.sin(rot_angle)
                y_new = x * np.sin(rot_angle) + y * np.cos(rot_angle)
                a_new = a + np.degrees(rot_angle)
                x0[3*i] = x_new
                x0[3*i+1] = y_new
                x0[3*i+2] = a_new % 360

        result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
            
    params = best_params
    inner = []
    for i in range(N):
        x, y, a = params[3*i:3*i+3]
        inner.append([float(x), float(y), float(a)])
        
    # Calculate final s for return
    inner_verts_list = [hex_vertices(x, y, 1.0, a) for x, y, a in inner]
    s = compute_outer_side(inner_verts_list)
    
    return inner, (0.0, 0.0), float(s), 0.0
