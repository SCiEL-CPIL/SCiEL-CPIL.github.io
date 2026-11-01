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

def compute_outer_side(inner_verts_list, cx, cy):
    angles = np.radians(30 + np.arange(6) * 60)
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    max_proj = -np.inf
    for verts in inner_verts_list:
        rel_verts = verts - np.array([cx, cy])
        projs = rel_verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
        
    return max_proj * 2.0 / np.sqrt(3)

def objective(params):
    inner_params = params[:3 * N]
    cx, cy = params[3 * N:3 * N + 2]
    
    inner_verts_list = []
    for i in range(N):
        x, y, a = inner_params[3*i:3*i+3]
        verts = hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
        
    s = compute_outer_side(inner_verts_list, cx, cy)
    
    total_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
            total_penalty += max(0.0, pd)
            
    return s + 100.0 * total_penalty

def _get_initial_guesses():
    guesses = []
    spacing = np.sqrt(3)
    
    # Base configuration: 1 center, 6 ring, 5 outer
    base_positions = []
    base_positions.append((0.0, 0.0))
    for k in range(6):
        theta = k * np.pi / 3
        base_positions.append((spacing * np.cos(theta), spacing * np.sin(theta)))
    for k in range(5):
        theta = k * 2 * np.pi / 5 + np.pi / 6
        base_positions.append((2 * spacing * np.cos(theta), 2 * spacing * np.sin(theta)))
        
    # Guess 1: Original
    params = []
    for x, y in base_positions:
        params.extend([x, y, 0.0])
    params.extend([0.0, 0.0])
    guesses.append(np.array(params))
    
    # Guess 2: Rotated
    params = []
    for x, y in base_positions:
        params.extend([x, y, 30.0])
    params.extend([0.0, 0.0])
    guesses.append(np.array(params))
    
    # Guess 3: Jittered
    np.random.seed(42)
    params = []
    for x, y in base_positions:
        x_j = x + np.random.uniform(-0.1, 0.1)
        y_j = y + np.random.uniform(-0.1, 0.1)
        a_j = np.random.uniform(0, 60)
        params.extend([x_j, y_j, a_j])
    params.extend([0.0, 0.0])
    guesses.append(np.array(params))
    
    # Guess 4: Different ring distribution
    # Maybe put the 5 outer ones at specific angles to minimize max radius
    # Angles: 0, 72, 144, 216, 288?
    # The 6 inner are at 0, 60, 120, 180, 240, 300.
    # The gaps are at 30, 90, 150, 210, 270, 330.
    # Place 5 in 5 of the 6 gaps?
    gap_angles = [30, 90, 150, 210, 270] # Skip 330
    pos4 = []
    pos4.append((0.0, 0.0))
    for k in range(6):
        theta = k * np.pi / 3
        pos4.append((spacing * np.cos(theta), spacing * np.sin(theta)))
    for ang in gap_angles:
        theta = np.radians(ang)
        # Distance to gap center is slightly larger than spacing
        # Distance from center to gap midpoint is 1.5 * spacing?
        # Hexagon side 1. Distance from center to vertex is 1.
        # Distance from center to edge midpoint is sqrt(3)/2.
        # Inner ring centers are at distance sqrt(3).
        # Gap centers are at distance 1.5 * sqrt(3)?
        # Let's just use 2 * spacing for now, maybe 1.8 * spacing?
        pos4.append((1.8 * spacing * np.cos(theta), 1.8 * spacing * np.sin(theta)))
        
    params = []
    for x, y in pos4:
        params.extend([x, y, 0.0])
    params.extend([0.0, 0.0])
    guesses.append(np.array(params))
        
    return guesses

def construct():
    guesses = _get_initial_guesses()
    best_s = np.inf
    best_params = None
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 180.0)] * N + [(-5.0, 5.0), (-5.0, 5.0)]
    
    for x0 in guesses:
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            s_val = result.fun - 100.0 * 0.0 # This is not right, result.fun includes penalty
            # We need to recalculate s without penalty to compare valid solutions
            # Or just use result.fun if penalty is 0.
            # Let's check penalty
            inner_params = result.x[:3 * N]
            cx, cy = result.x[3 * N:3 * N + 2]
            inner_verts_list = []
            for i in range(N):
                x, y, a = inner_params[3*i:3*i+3]
                verts = hex_vertices(x, y, 1.0, a)
                inner_verts_list.append(verts)
            total_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
                    total_pen += max(0.0, pd)
            if total_pen < 1e-6:
                s_clean = compute_outer_side(inner_verts_list, cx, cy)
                if s_clean < best_s:
                    best_s = s_clean
                    best_params = result.x
            else:
                # If still overlapping, we might still want to keep it if it's the best "attempt"
                # But we prefer valid.
                pass
        except Exception:
            continue
            
    if best_params is None:
        # Fallback to first guess result
        result = minimize(objective, guesses[0], method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        best_params = result.x
        
    params = best_params
    inner = []
    for i in range(N):
        x, y, a = params[3*i:3*i+3]
        inner.append([float(x), float(y), float(a)])
    cx, cy = params[3*N:3*N+2]
    inner_verts_list = [hex_vertices(x, y, 1.0, a) for x, y, a in inner]
    s = compute_outer_side(inner_verts_list, cx, cy)
    return inner, (float(cx), float(cy)), float(s), 0.0
