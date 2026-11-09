import numpy as np
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = 620  # Leave 10s margin for 630s limit

    # Constants
    n_inner = 11
    epsilon = 1e-6
    apothem_ratio = np.sqrt(3) / 2.0

    def get_hex_vertices(cx, cy, angle_deg):
        a_rad = np.radians(angle_deg)
        angles = a_rad + np.radians([0, 60, 120, 180, 240, 300])
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

    def get_hex_normals(angle_deg):
        a_rad = np.radians(angle_deg)
        base_angles = np.radians([30, 90, 150, 210, 270, 330])
        angles = base_angles + a_rad
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def compute_side(coords, angles):
        # coords: (11, 2), angles: (11,)
        # Compute minimal side length of enclosing hexagon centered at origin with angle 0
        all_vertices = []
        for i in range(11):
            v = get_hex_vertices(coords[i, 0], coords[i, 1], angles[i])
            all_vertices.append(v)
        all_vertices = np.vstack(all_vertices)
        
        # Normals of enclosing hexagon (angle 0)
        outer_normals = np.array([
            [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
            for k in range(6)
        ])
        
        max_proj = 0
        for n in outer_normals:
            proj = np.dot(all_vertices, n)
            max_p = np.max(proj)
            s_required = max_p / apothem_ratio
            if s_required > max_proj:
                max_proj = s_required
        
        return max_proj

    def check_validity(inner, outer, side):
        if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
            return False
        if side < np.sqrt(11) - 1e-9:
            return False
            
        ox, oy, oa = outer
        inner_shifted = inner.copy()
        inner_shifted[:, 0] -= ox
        inner_shifted[:, 1] -= oy
        
        # Check edge lengths
        for i in range(11):
            v = get_hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1], inner_shifted[i, 2])
            diffs = np.diff(v, axis=0, append=v[:1])
            lengths = np.linalg.norm(diffs, axis=1)
            if not np.all(np.abs(lengths - 1.0) < 1e-8):
                return False

        # Check non-overlap
        for i in range(11):
            vi = get_hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1], inner_shifted[i, 2])
            for j in range(i+1, 11):
                vj = get_hex_vertices(inner_shifted[j, 0], inner_shifted[j, 1], inner_shifted[j, 2])
                n_i = get_hex_normals(inner_shifted[i, 2])
                n_j = get_hex_normals(inner_shifted[j, 2])
                all_normals = np.vstack([n_i, n_j])
                
                separated = False
                for n in all_normals:
                    proj_i = np.dot(vi, n)
                    proj_j = np.dot(vj, n)
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    gap = min(min_j - max_i, min_i - max_j)
                    if gap > epsilon:
                        separated = True
                        break
                if not separated:
                    return False
        
        # Check containment
        oa_rad = np.radians(oa)
        outer_v = np.column_stack([
            ox + side * np.cos(oa_rad + np.radians(60*k)),
            oy + side * np.sin(oa_rad + np.radians(60*k))
        ] for k in range(6))
        
        for i in range(6):
            p1 = outer_v[i]
            p2 = outer_v[(i+1)%6]
            for i_h in range(11):
                vi = get_hex_vertices(inner_shifted[i_h, 0], inner_shifted[i_h, 1], inner_shifted[i_h, 2])
                for vp in vi:
                    c = np.cross(p2 - p1, vp - p1)
                    if c < -epsilon:
                        return False
        return True

    def is_valid_packing(coords, angles):
        # Quick check for non-overlap and containment (assuming centered at origin, outer angle 0)
        side = compute_side(coords, angles)
        if side < np.sqrt(11) - 1e-9:
            return False
        
        # Check non-overlap
        for i in range(11):
            vi = get_hex_vertices(coords[i, 0], coords[i, 1], angles[i])
            for j in range(i+1, 11):
                vj = get_hex_vertices(coords[j, 0], coords[j, 1], angles[j])
                n_i = get_hex_normals(angles[i])
                n_j = get_hex_normals(angles[j])
                all_normals = np.vstack([n_i, n_j])
                
                separated = False
                for n in all_normals:
                    proj_i = np.dot(vi, n)
                    proj_j = np.dot(vj, n)
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    gap = min(min_j - max_i, min_i - max_j)
                    if gap > epsilon:
                        separated = True
                        break
                if not separated:
                    return False
        
        # Check containment (approximate: just check if all vertices are within side * apothem in all normal directions)
        all_vertices = []
        for i in range(11):
            v = get_hex_vertices(coords[i, 0], coords[i, 1], angles[i])
            all_vertices.append(v)
        all_vertices = np.vstack(all_vertices)
        
        outer_normals = np.array([
            [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
            for k in range(6)
        ])
        
        for n in outer_normals:
            proj = np.dot(all_vertices, n)
            if np.max(proj) > side * apothem_ratio + 1e-6:
                return False
        
        return True

    def local_refinement(coords, angles, max_iters=1000):
        # Simple gradient-free local search
        best_coords = coords.copy()
        best_angles = angles.copy()
        best_side = compute_side(best_coords, best_angles)
        
        if not is_valid_packing(best_coords, best_angles):
            return None, None, best_side
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            # Try small perturbations
            for dim in range(33):
                for step in [0.01, 0.05, 0.1]:
                    new_coords = best_coords.copy()
                    new_angles = best_angles.copy()
                    
                    if dim < 22:
                        new_coords[dim // 3, dim % 3] += step
                    else:
                        new_angles[dim - 22] += step
                    
                    if is_valid_packing(new_coords, new_angles):
                        new_side = compute_side(new_coords, new_angles)
                        if new_side < best_side - 1e-8:
                            best_coords = new_coords
                            best_angles = new_angles
                            best_side = new_side
                            improved = True
            
            if not improved:
                break
        
        return best_coords, best_angles, best_side

    # Initial configurations to try
    configs = []
    
    # Config 1: Center + 6 ring + 4 corners (classic flower)
    coords1 = np.zeros((11, 2))
    angles1 = np.zeros(11)
    # Hex 0 at center
    coords1[0] = [0, 0]
    # Hex 1-6 in ring at distance sqrt(3)
    for i in range(1, 7):
        angle = np.radians(60 * (i-1))
        r = np.sqrt(3)
        coords1[i] = [r * np.cos(angle), r * np.sin(angle)]
    # Hex 7-10 at corners
    for i in range(7, 11):
        angle = np.radians(30 + 120 * ((i-7) % 3))
        r = 2.5
        coords1[i] = [r * np.cos(angle), r * np.sin(angle)]
    configs.append((coords1, angles1))
    
    # Config 2: Different arrangement - 2 rows of 3, 2 rows of 3, 1 center
    coords2 = np.zeros((11, 2))
    angles2 = np.zeros(11)
    # Row 1 (top): 3 hexagons
    for i in range(3):
        coords2[i] = [i * 1.5 - 1.5, 2.6]
    # Row 2: 3 hexagons
    for i in range(3):
        coords2[3+i] = [i * 1.5 - 1.5, 0.8]
    # Row 3: 3 hexagons
    for i in range(3):
        coords2[6+i] = [i * 1.5 - 1.5, -1.0]
    # Center hex
    coords2[9] = [0, 0]
    # One more
    coords2[10] = [0, -2.6]
    configs.append((coords2, angles2))
    
    # Config 3: Linear arrangement
    coords3 = np.zeros((11, 2))
    angles3 = np.zeros(11)
    for i in range(11):
        coords3[i] = [(i - 5) * 1.5, 0]
    configs.append((coords3, angles3))
    
    # Config 4: Hexagonal lattice subset
    coords4 = np.zeros((11, 2))
    angles4 = np.zeros(11)
    # Generate hex lattice points
    lattice_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * 1.5
            y = j * np.sqrt(3) + (i % 2) * np.sqrt(3) / 2
            lattice_points.append([x, y])
    # Take first 11
    for i in range(min(11, len(lattice_points))):
        coords4[i] = lattice_points[i]
    configs.append((coords4, angles4))
    
    best_result = None
    best_score = -np.inf
    
    # Try each config with local refinement
    for idx, (coords, angles) in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        # Check if initial config is valid
        if is_valid_packing(coords, angles):
            side = compute_side(coords, angles)
            if 1.0 / side > best_score:
                best_score = 1.0 / side
                best_result = (coords.copy(), np.array([0, 0, 0]), side)
        
        # Try local refinement
        if time.time() < deadline - 20:
            ref_coords, ref_angles, ref_side = local_refinement(coords, angles, max_iters=500)
            if ref_coords is not None and is_valid_packing(ref_coords, ref_angles):
                if 1.0 / ref_side > best_score:
                    best_score = 1.0 / ref_side
                    best_result = (ref_coords.copy(), np.array([0, 0, 0]), ref_side)
    
    # If no valid result found, use fallback
    if best_result is None:
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
    coords, outer, side = best_result
    inner = np.zeros((11, 3))
    for i in range(11):
        inner[i, 0] = coords[i, 0]
        inner[i, 1] = coords[i, 1]
        inner[i, 2] = 0  # angles are 0 in our configs
    
    return inner, outer, side
