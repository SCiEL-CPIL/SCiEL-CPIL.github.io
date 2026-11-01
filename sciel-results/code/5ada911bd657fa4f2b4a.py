import numpy as np
from scipy.optimize import minimize, differential_evolution

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

def compute_outer_side_and_check(inner_verts_list, cx, cy, outer_angle_deg):
    a_outer = np.radians(outer_angle_deg)
    # Outer hexagon normals
    # The vertices of a regular hexagon with angle a are at a + k*pi/3.
    # The normals to the sides are at a + pi/6 + k*pi/3.
    angles = a_outer + np.pi / 6 + np.arange(6) * np.pi / 3
    normals = np.array([[np.cos(ang), np.sin(ang)] for ang in angles])
    
    center = np.array([cx, cy])
    max_proj = -np.inf
    for verts in inner_verts_list:
        rel_verts = verts - center
        projs = rel_verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
        
    # s * sqrt(3)/2 = max_proj
    return max_proj * 2.0 / np.sqrt(3)

def objective(params):
    # params: [x1,y1,a1, ..., x12,y12,a12, cx, cy, outer_angle]
    inner_params = params[:3 * N]
    cx, cy, outer_angle = params[3 * N:3 * N + 3]
    
    inner_verts_list = []
    for i in range(N):
        x, y, a = inner_params[3*i:3*i+3]
        verts = hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
        
    s = compute_outer_side_and_check(inner_verts_list, cx, cy, outer_angle)
    
    total_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
            total_penalty += max(0.0, pd)
            
    return s + 100.0 * total_penalty

def initial_guess():
    # Create a hexagonal flower pattern
    positions = []
    positions.append((0.0, 0.0))
    spacing = np.sqrt(3)
    for k in range(6):
        theta = k * np.pi / 3
        positions.append((spacing * np.cos(theta), spacing * np.sin(theta)))
    for k in range(5):
        theta = k * 2 * np.pi / 5 + np.pi / 6
        positions.append((2 * spacing * np.cos(theta), 2 * spacing * np.sin(theta)))
        
    params = []
    for x, y in positions:
        params.extend([x, y, 0.0])
    params.extend([0.0, 0.0, 0.0])
    return np.array(params)

def construct():
    x0 = initial_guess()
    
    # Define bounds for differential evolution
    # Inner hexagons: x, y in [-5, 5], angle in [0, 360]
    # Outer: x, y in [-5, 5], angle in [0, 360]
    bounds = [(-5, 5), (-5, 5), (0, 360)] * N + [(-5, 5), (-5, 5), (0, 360)]
    
    # Use Differential Evolution for global search
    # maxiter=50 to keep time low, then refine
    result_de = differential_evolution(objective, bounds, seed=42, maxiter=50, tol=1e-6, polish=False)
    
    # Refine with L-BFGS-B
    result = minimize(objective, result_de.x, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
    
    params = result.x
    inner = []
    for i in range(N):
        x, y, a = params[3*i:3*i+3]
        inner.append([float(x), float(y), float(a)])
    cx, cy, outer_angle = params[3*N:3*N+3]
    
    # Calculate final s
    inner_verts_list = [hex_vertices(x, y, 1.0, a) for x, y, a in inner]
    s = compute_outer_side_and_check(inner_verts_list, cx, cy, outer_angle)
    
    return inner, (float(cx), float(cy)), float(s), float(outer_angle)
