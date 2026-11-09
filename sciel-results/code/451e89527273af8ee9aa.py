"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def hex_vertices(cx, cy, side, angle_deg):
        """Get 6 vertices of a regular hexagon."""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            vertices.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
        return np.array(vertices)
    
    def required_outer_radius(point, outer_center, outer_angle):
        """Compute required circumradius of outer hexagon to contain this point."""
        dx = point[0] - outer_center[0]
        dy = point[1] - outer_center[1]
        angle_rad = np.radians(outer_angle)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        max_proj = 0.0
        for i in range(6):
            theta = np.radians(30 + i * 60)
            nx = np.cos(theta)
            ny = np.sin(theta)
            proj = lx * nx + ly * ny
            if proj > max_proj:
                max_proj = proj
        
        return max_proj * 2.0 / np.sqrt(3)
    
    def compute_outer_params(inner_hexes):
        """Compute optimal outer hexagon params for given inner hexagons."""
        # Collect all vertices
        all_verts = []
        for (cx, cy, angle) in inner_hexes:
            verts = hex_vertices(cx, cy, 1.0, angle)
            all_verts.extend(verts.tolist())
        all_verts = np.array(all_verts)
        
        # Optimal center is centroid of all vertices
        center = all_verts.mean(axis=0)
        
        # Try different orientations to find minimum required radius
        best_radius = np.inf
        best_angle = 0.0
        for angle in np.linspace(0, 180, 36):
            max_r = 0.0
            for v in all_verts:
                r = required_outer_radius(v, center, angle)
                if r > max_r:
                    max_r = r
            if max_r < best_radius:
                best_radius = max_r
                best_angle = angle
        
        return center, best_radius, best_angle
    
    def check_disjoint(hex1, hex2, eps=1e-6):
        """Check if two unit hexagons are disjoint."""
        dx = hex1[0] - hex2[0]
        dy = hex1[1] - hex2[1]
        dist = np.sqrt(dx*dx + dy*dy)
        return dist >= 2.0 - eps
    
    def objective(params):
        """Objective: outer hexagon side length."""
        if time.time() > deadline:
            return 1e10
        
        inner_hexes = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_hexes.append((x, y, a))
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint(inner_hexes[i], inner_hexes[j]):
                    return 1e10
        
        # Compute optimal outer hexagon
        center, radius, angle = compute_outer_params(inner_hexes)
        return radius
    
    # Initial configuration: honeycomb grid
    hex_spacing_x = np.sqrt(3)
    hex_spacing_y = 1.5
    
    inner_hexes = []
    rows = 3
    cols = 4
    for row in range(rows):
        for col in range(cols):
            x = (col - (cols-1)/2.0) * hex_spacing_x
            if row % 2 == 1:
                x += hex_spacing_x / 2.0
            y = (row - (rows-1)/2.0) * hex_spacing_y
            inner_hexes.append((x, y, 0.0))
    
    # Center the configuration
    xs = [h[0] for h in inner_hexes]
    ys = [h[1] for h in inner_hexes]
    cx_offset = (max(xs) + min(xs)) / 2.0
    cy_offset = (max(ys) + min(ys)) / 2.0
    inner_hexes = [(h[0] - cx_offset, h[1] - cy_offset, h[2]) for h in inner_hexes]
    
    # Build initial params
    params = np.zeros(3*N)
    for i, (x, y, a) in enumerate(inner_hexes):
        params[3*i] = x
        params[3*i+1] = y
        params[3*i+2] = a
    
    best_params = params.copy()
    best_score = objective(params)
    
    # Local optimization
    try:
        result = minimize(objective, params, method='Nelder-Mead', 
                         options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        inner = [[best_params[3*i], best_params[3*i+1], np.degrees(best_params[3*i+2])] for i in range(N)]
        center, radius, angle = compute_outer_params([(p[3*i], p[3*i+1], p[3*i+2]) for i in range(N)])
        return inner, (center[0], center[1]), radius, np.degrees(angle)
    
    # More restarts
    for restart in range(8):
        if time.time() > deadline - 10:
            break
        np.random.seed(42 + restart)
        perturbed = best_params.copy()
        perturbed += np.random.normal(0, 0.15, 3*N)
        
        try:
            result = minimize(objective, perturbed, method='Nelder-Mead',
                             options={'maxiter': 400, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Final extraction
    inner = [[best_params[3*i], best_params[3*i+1], np.degrees(best_params[3*i+2])] for i in range(N)]
    center, radius, angle = compute_outer_params([(p[3*i], p[3*i+1], p[3*i+2]) for i in range(N)])
    
    return inner, (center[0], center[1]), radius, np.degrees(angle)
# EVOLVE-BLOCK-END
