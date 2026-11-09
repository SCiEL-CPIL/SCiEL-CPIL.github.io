import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = 620  # Leave 10s margin for 630s limit
    
    def hex_vertices(cx, cy, s, a_deg):
        a = np.radians(a_deg)
        angles = a + np.radians(np.array([0, 60, 120, 180, 240, 300]))
        return np.column_stack([cx + s*np.cos(angles), cy + s*np.sin(angles)])
    
    def hex_edge_normals(a_deg):
        a = np.radians(a_deg)
        base = np.radians(np.array([30, 90, 150, 210, 270, 330]))
        angles = base + a
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def check_separation(vi, vj, ni, nj):
        """Check if two hexagons are separated (no intersection)"""
        all_normals = np.vstack([ni, nj])
        for n in all_normals:
            proj_i = np.dot(vi, n)
            proj_j = np.dot(vj, n)
            min_i, max_i = np.min(proj_i), np.max(proj_i)
            min_j, max_j = np.min(proj_j), np.max(proj_j)
            gap = min(min_j - max_i, min_i - max_j)
            if gap > 1e-6:
                return True
        return False
    
    def compute_min_side(coords, angles, outer_angle):
        """Compute minimum side length of enclosing hexagon with given outer angle"""
        all_vertices = []
        for i in range(11):
            v = hex_vertices(coords[i,0], coords[i,1], 1.0, angles[i])
            all_vertices.append(v)
        all_vertices = np.vstack(all_vertices)
        
        outer_normals = hex_edge_normals(outer_angle)
        
        max_proj = 0
        for n in outer_normals:
            proj = np.dot(all_vertices, n)
            max_p = np.max(proj)
            s_required = max_p / (np.sqrt(3)/2)
            if s_required > max_proj:
                max_proj = s_required
        
        return max_proj
    
    def is_valid(coords, angles, outer_angle, side):
        """Check if packing is valid"""
        if side < np.sqrt(11) - 1e-9:
            return False
        
        # Check edge lengths
        for i in range(11):
            v = hex_vertices(coords[i,0], coords[i,1], 1.0, angles[i])
            diffs = np.diff(v, axis=0, append=v[:1])
            lengths = np.linalg.norm(diffs, axis=1)
            if not np.all(np.abs(lengths - 1.0) < 1e-8):
                return False
        
        # Check non-overlap
        for i in range(11):
            vi = hex_vertices(coords[i,0], coords[i,1], 1.0, angles[i])
            ni = hex_edge_normals(angles[i])
            for j in range(i+1, 11):
                vj = hex_vertices(coords[j,0], coords[j,1], 1.0, angles[j])
                nj = hex_edge_normals(angles[j])
                if not check_separation(vi, vj, ni, nj):
                    return False
        
        # Check containment
        outer_v = hex_vertices(0, 0, side, outer_angle)
        for i in range(6):
            p1 = outer_v[i]
            p2 = outer_v[(i+1)%6]
            for i_h in range(11):
                vi = hex_vertices(coords[i_h,0], coords[i_h,1], 1.0, angles[i_h])
                for vp in vi:
                    c = np.cross(p2 - p1, vp - p1)
                    if c < -1e-6:
                        return False
        return True
    
    def objective(params):
        """Objective: minimize side length. params = [coords(22), angles(11), outer_angle(1)]"""
        if time.time() > deadline - 5:
            return 1e10
        
        coords = params[:22].reshape(11, 2)
        angles = params[22:33]
        outer_angle = params[33]
        
        side = compute_min_side(coords, angles, outer_angle)
        
        if side < np.sqrt(11) - 1e-9:
            return 1e10
        
        if not is_valid(coords, angles, outer_angle, side):
            return 1e10 + side
        
        return side
    
    # Initial configuration: hexagonal flower pattern
    # Center + 6 ring + 4 corners
    coords0 = np.zeros((11, 2))
    angles0 = np.zeros(11)
    
    # Hex 0: center
    coords0[0] = [0, 0]
    
    # Hex 1-6: ring at distance sqrt(3)
    for i in range(1, 7):
        angle = np.radians(60 * (i-1))
        r = np.sqrt(3)
        coords0[i] = [r * np.cos(angle), r * np.sin(angle)]
    
    # Hex 7-10: 4 corners
    for i in range(7, 11):
        angle = np.radians(30 + 120 * ((i-7) % 3))
        r = 2.5
        coords0[i] = [r * np.cos(angle), r * np.sin(angle)]
    
    outer_angle0 = 0.0
    
    # Build initial params
    params0 = np.concatenate([coords0.flatten(), angles0, [outer_angle0]])
    
    best_side = None
    best_params = None
    
    # Try multiple initial configurations
    configs = []
    
    # Config 1: Flower pattern (as above)
    configs.append(params0.copy())
    
    # Config 2: Slightly different ring positions
    coords1 = coords0.copy()
    for i in range(1, 7):
        angle = np.radians(60 * (i-1) + 30)
        r = np.sqrt(3)
        coords1[i] = [r * np.cos(angle), r * np.sin(angle)]
    for i in range(7, 11):
        angle = np.radians(60 + 120 * ((i-7) % 3))
        r = 2.5
        coords1[i] = [r * np.cos(angle), r * np.sin(angle)]
    configs.append(np.concatenate([coords1.flatten(), angles0, [0.0]]))
    
    # Config 3: Linear-ish arrangement
    coords2 = np.zeros((11, 2))
    for i in range(11):
        coords2[i] = [(i - 5) * 1.5, 0]
    configs.append(np.concatenate([coords2.flatten(), angles0, [0.0]]))
    
    # Config 4: Hexagonal lattice subset
    coords3 = np.zeros((11, 2))
    lattice_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * 1.5
            y = j * np.sqrt(3) + (i % 2) * np.sqrt(3) / 2
            lattice_points.append([x, y])
    for i in range(min(11, len(lattice_points))):
        coords3[i] = lattice_points[i]
    configs.append(np.concatenate([coords3.flatten(), angles0, [0.0]]))
    
    # Optimize each configuration
    for cfg_idx, cfg in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        # Check if initial is valid
        coords_c = cfg[:22].reshape(11, 2)
        angles_c = cfg[22:33]
        outer_c = cfg[33]
        side_c = compute_min_side(coords_c, angles_c, outer_c)
        
        if is_valid(coords_c, angles_c, outer_c, side_c):
            if best_side is None or side_c < best_side:
                best_side = side_c
                best_params = cfg.copy()
        
        # Try optimization
        if time.time() < deadline - 20:
            try:
                result = minimize(objective, cfg, method='Nelder-Mead', 
                                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
                
                if result.fun < 1e9:
                    res_coords = result.x[:22].reshape(11, 2)
                    res_angles = result.x[22:33]
                    res_outer = result.x[33]
                    res_side = compute_min_side(res_coords, res_angles, res_outer)
                    
                    if is_valid(res_coords, res_angles, res_outer, res_side):
                        if best_side is None or res_side < best_side:
                            best_side = res_side
                            best_params = result.x.copy()
            except Exception:
                pass
    
    # If no valid result, use fallback
    if best_params is None:
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Format result
    coords = best_params[:22].reshape(11, 2)
    angles = best_params[22:33]
    outer_angle = best_params[33]
    side = compute_min_side(coords, angles, outer_angle)
    
    inner = np.zeros((11, 3))
    for i in range(11):
        inner[i, 0] = coords[i, 0]
        inner[i, 1] = coords[i, 1]
        inner[i, 2] = angles[i]
    
    outer = np.array([0, 0, outer_angle])
    
    return inner, outer, side
