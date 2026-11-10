# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side d and angle (deg)."""
    a = np.radians(angle_deg)
    thetas = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(thetas),
                            cy + d * np.sin(thetas)])

def _compute_min_s(centers, angles):
    """
    Compute the minimum side length s of a regular hexagon (centered at origin, angle 0)
    that contains all 12 unit hexagons with given centers and angles.
    
    For a regular hexagon with side s centered at origin with angle 0:
    - The 6 outward edge normals point at angles 30, 90, 150, 210, 270, 330 degrees
    - A point (px, py) is inside if dot((px,py), n_k) <= s*sqrt(3)/2 for all k
    - So s >= max_k [max over all vertices of dot(v, n_k)] / (sqrt(3)/2)
    """
    sqrt3_2 = np.sqrt(3) / 2
    normal_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_proj = 0.0
    for i in range(12):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        # Project all vertices onto each normal
        projections = verts @ normals.T  # (6, 6)
        for k in range(6):
            proj_max = projections[:, k].max()
            if proj_max > max_proj:
                max_proj = proj_max
    
    return max_proj / sqrt3_2

def _separation_gap(verts1, verts2):
    """
    Separating axis test: returns the minimum gap between two hexagons.
    Positive gap means separated, negative means overlapping.
    """
    # Get edge normals for both hexagons
    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / np.where(norms == 0, 1.0, norms)
    
    n1 = get_normals(verts1)
    n2 = get_normals(verts2)
    all_normals = np.vstack([n1, n2])
    
    min_gap = np.inf
    for normal in all_normals:
        proj1 = verts1 @ normal
        proj2 = verts2 @ normal
        gap1 = proj2.min() - proj1.max()
        gap2 = proj1.min() - proj2.max()
        min_gap = min(min_gap, gap1, gap2)
    return min_gap

def _check_validity(centers, angles, s):
    """Check if the configuration is valid."""
    # Check s >= sqrt(12)
    if s < np.sqrt(12) - 1e-8:
        return False
    
    # Check overlaps
    for i in range(12):
        verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for j in range(i + 1, 12):
            verts_j = _hex_vertices(centers[j, 0], centers[j, 1], 1.0, angles[j])
            gap = _separation_gap(verts_i, verts_j)
            if gap < -1e-6:
                return False
    
    # Check boundary: all inner vertices must be inside outer hexagon
    outer_verts = _hex_vertices(0, 0, s, 0)
    for i in range(12):
        verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for v in verts_i:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False
    
    return True

def _penalty_objective(x, deadline):
    """
    Objective function for optimization: minimize s + penalties for violations.
    x: 36-vector = 12*(cx, cy, angle_deg)
    """
    if time.time() > deadline:
        return 1e10
    
    centers = x[:24].reshape(12, 2)
    angles = x[24:36]
    
    # Compute s needed
    s = _compute_min_s(centers, angles)
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(12):
        verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for j in range(i + 1, 12):
            verts_j = _hex_vertices(centers[j, 0], centers[j, 1], 1.0, angles[j])
            gap = _separation_gap(verts_i, verts_j)
            if gap < 0:
                penalty += (-gap) ** 2
    
    # Penalty for boundary violations
    outer_verts = _hex_vertices(0, 0, s, 0)
    for i in range(12):
        verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for v in verts_i:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < 0:
                    penalty += (-cross) ** 2
    
    # Ensure s is at least sqrt(12)
    s_eff = max(s, np.sqrt(12))
    
    return s_eff + penalty * 100.0

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit with safety margin
    
    best_score = 0.0
    best_x = None
    best_s = np.sqrt(12)
    
    # Initial guess: hexagonal grid arrangement
    d = 2 * np.sqrt(3)  # distance between centers for touching hexagons
    initial_centers = []
    # Flower pattern: center + 6 around + 5 more
    # Center
    initial_centers.append([0, 0])
    # 6 around at distance 2*sqrt(3)
    for k in range(6):
        angle = np.radians(60 * k)
        initial_centers.append([d * np.cos(angle), d * np.sin(angle)])
    # 5 more in a ring
    d2 = 2 * d
    for k in range(5):
        angle = np.radians(60 * k + 30)
        initial_centers.append([d2 * np.cos(angle), d2 * np.sin(angle)])
    
    initial_centers = np.array(initial_centers)
    initial_angles = np.zeros(12)
    
    # Center the configuration
    initial_centers -= initial_centers.mean(axis=0)
    
    x0 = np.concatenate([initial_centers.flatten(), initial_angles])
    
    # Try multiple optimization strategies
    np.random.seed(42)
    
    # Strategy 1: Nelder-Mead
    for attempt in range(30):
        if time.time() > deadline - 30:
            break
        
        if attempt == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.randn(36) * 1.0
            x_start[24:36] = np.random.uniform(0, 60, 12)
        
        try:
            from scipy.optimize import minimize
            result = minimize(
                _penalty_objective,
                x_start,
                args=(deadline,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if result.fun < 1e9:
                x_opt = result.x
                centers = x_opt[:24].reshape(12, 2)
                angles = x_opt[24:36]
                
                s = _compute_min_s(centers, angles)
                
                if _check_validity(centers, angles, s):
                    score = 1.0 / s
                    if score > best_score:
                        best_score = score
                        best_x = x_opt.copy()
                        best_s = s
                        
        except Exception:
            continue
    
    # Strategy 2: L-BFGS-B
    for attempt in range(20):
        if time.time() > deadline - 30:
            break
        
        if attempt == 0:
            x_start = x0.copy()
        elif best_x is not None:
            x_start = best_x + np.random.randn(36) * 0.5
        else:
            x_start = x0 + np.random.randn(36) * 1.0
            x_start[24:36] = np.random.uniform(0, 60, 12)
        
        try:
            from scipy.optimize import minimize
            result = minimize(
                _penalty_objective,
                x_start,
                args=(deadline,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000}
            )
            
            if result.fun < 1e9:
                x_opt = result.x
                centers = x_opt[:24].reshape(12, 2)
                angles = x_opt[24:36]
                
                s = _compute_min_s(centers, angles)
                
                if _check_validity(centers, angles, s):
                    score = 1.0 / s
                    if score > best_score:
                        best_score = score
                        best_x = x_opt.copy()
                        best_s = s
                        
        except Exception:
            continue
    
    # If we found a good solution, use it
    if best_x is not None:
        centers = best_x[:24].reshape(12, 2)
        angles = best_x[24:36]
        s = _compute_min_s(centers, angles)
        
        inner_hex_data = np.column_stack([centers, angles])
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        
        return inner_hex_data, outer_hex_data, float(s)
    
    # Fallback: use a known valid configuration
    # Create a simple valid packing
    d = 2 * np.sqrt(3)
    centers = np.array([
        [0, 0],
        [d, 0],
        [-d, 0],
        [d/2, d*np.sqrt(3)/2],
        [-d/2, d*np.sqrt(3)/2],
        [d/2, -d*np.sqrt(3)/2],
        [-d/2, -d*np.sqrt(3)/2],
        [2*d, 0],
        [-2*d, 0],
        [d, d*np.sqrt(3)],
        [-d, d*np.sqrt(3)],
        [0, -d*np.sqrt(3)]
    ])
    angles = np.zeros(12)
    
    # Center
    centers -= centers.mean(axis=0)
    
    s = _compute_min_s(centers, angles)
    
    inner_hex_data = np.column_stack([centers, angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, float(s)

# EVOLVE-BLOCK-END
