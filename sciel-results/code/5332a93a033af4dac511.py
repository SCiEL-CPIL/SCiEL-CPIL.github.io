import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices (flat-top: vertices at 0, 60, ..., 300)
    unit_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    sqrt3 = math.sqrt(3)

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        theta_rad = math.radians(outer_angle)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3),
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        return max_proj * 2.0 / sqrt3

    def soft_max(values, temperature=0.01):
        """Log-sum-exp soft max for smooth surrogate of max."""
        shifted = values / temperature
        shifted -= np.max(shifted)
        return temperature * (np.max(values) + temperature * math.log(np.sum(np.exp(shifted - np.max(shifted)))))

    def calc_min_outer_side_smooth(positions, angles, outer_angle=0.0, temp=0.01):
        """Smooth version using log-sum-exp for the max projection."""
        theta_rad = math.radians(outer_angle)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3),
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        
        # Collect all projections
        all_projs = []
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T  # (6, 6)
            all_projs.append(projs)
        
        all_projs = np.concatenate(all_projs, axis=0)  # (72, 6)
        # Soft max over all 72 values
        soft_max_val = soft_max(all_projs.ravel(), temp)
        return soft_max_val * 2.0 / sqrt3

    # Honeycomb lattice
    def lattice_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        return (x, y)

    def is_connected(coords):
        if not coords: return True
        coord_set = set(coords)
        start = next(iter(coord_set))
        visited = {start}
        stack = [start]
        while stack:
            q, r = stack.pop()
            neighbors = [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]
            for nq, nr in neighbors:
                if (nq, nr) in coord_set and (nq, nr) not in visited:
                    visited.add((nq, nr))
                    stack.append((nq, nr))
        return len(visited) == len(coords)

    def normalize_shape(coords):
        xs = [lattice_to_cartesian(q, r)[0] for q, r in coords]
        ys = [lattice_to_cartesian(q, r)[1] for q, r in coords]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(xs)
        return [(lattice_to_cartesian(q, r)[0] - cx, lattice_to_cartesian(q, r)[1] - cy) for q, r in coords]

    def generate_random_polyhex(size):
        import random
        coords = {(0, 0)}
        frontier = [(0, 0)]
        while len(coords) < size:
            if not frontier:
                return None
            random.shuffle(frontier)
            q, r = frontier[0]
            neighbors = [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]
            random.shuffle(neighbors)
            added = False
            for nq, nr in neighbors:
                if (nq, nr) not in coords:
                    coords.add((nq, nr))
                    frontier.append((nq, nr))
                    added = True
                    break
            if not added:
                return None
        if is_connected(list(coords)):
            return list(coords)
        return None

    # Generate candidate shapes
    candidate_shapes = []
    
    # 2x3 rectangle
    shape_2x3 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)]
    candidate_shapes.append(normalize_shape(shape_2x3))
    
    # 3x2 rectangle  
    shape_3x2 = [(0,0), (1,0), (0,1), (1,1), (0,2), (1,2)]
    candidate_shapes.append(normalize_shape(shape_3x2))
    
    # Compact 4x3-ish block
    shape_4x3 = [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (3,2)]
    candidate_shapes.append(normalize_shape(shape_4x3))
    
    # Random polyhexes
    import random
    random.seed(42)
    for _ in range(80):
        if time.time() > deadline - 40:
            break
        shape = generate_random_polyhex(N)
        if shape:
            candidate_shapes.append(normalize_shape(shape))

    # Deduplicate
    unique_shapes = []
    seen = set()
    for shape in candidate_shapes:
        key = tuple(sorted((round(x, 4), round(y, 4)) for x, y in shape))
        if key not in seen:
            seen.add(key)
            unique_shapes.append(shape)

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0

    for shape in unique_shapes:
        if time.time() > deadline - 15:
            break
        
        pos = np.array(shape)
        
        # Grid search over cluster rotation (alpha) and outer rotation (beta)
        # Both are periodic with period 60 degrees
        best_shape_side = float('inf')
        best_alpha = 0.0
        best_beta = 0.0
        
        # Coarse grid: 24 steps each (2.5 degree resolution)
        alphas = np.linspace(0, 60, 24, endpoint=False)
        betas = np.linspace(0, 60, 24, endpoint=False)
        
        for alpha in alphas:
            alpha_rad = math.radians(alpha)
            cos_a = math.cos(alpha_rad)
            sin_a = math.sin(alpha_rad)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_pos = pos @ rot.T
            angles_arr = np.full(N, alpha)
            
            for beta in betas:
                side = calc_min_outer_side(rotated_pos, angles_arr, beta)
                if side < best_shape_side:
                    best_shape_side = side
                    best_alpha = alpha
                    best_beta = beta
        
        # Fine grid around best: 10 steps each (0.25 degree resolution)
        alpha_fine = np.linspace(max(0, best_alpha - 2.5), min(60, best_alpha + 2.5), 10)
        beta_fine = np.linspace(max(0, best_beta - 2.5), min(60, best_beta + 2.5), 10)
        
        for alpha in alpha_fine:
            alpha_rad = math.radians(alpha)
            cos_a = math.cos(alpha_rad)
            sin_a = math.sin(alpha_rad)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_pos = pos @ rot.T
            angles_arr = np.full(N, alpha)
            
            for beta in beta_fine:
                side = calc_min_outer_side(rotated_pos, angles_arr, beta)
                if side < best_shape_side:
                    best_shape_side = side
                    best_alpha = alpha
                    best_beta = beta
        
        # Smooth refinement using scipy minimize on the soft-max objective
        def objective_smooth(params):
            alpha, beta = params
            alpha_rad = math.radians(alpha)
            cos_a = math.cos(alpha_rad)
            sin_a = math.sin(alpha_rad)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_pos = pos @ rot.T
            angles_arr = np.full(N, alpha)
            return calc_min_outer_side_smooth(rotated_pos, angles_arr, beta, temp=0.05)
        
        # Try a few starting points near the grid optimum
        starts = [(best_alpha, best_beta),
                  (best_alpha - 1, best_beta),
                  (best_alpha + 1, best_beta),
                  (best_alpha, best_beta - 1),
                  (best_alpha, best_beta + 1)]
        
        for start in starts:
            try:
                res = minimize(objective_smooth, start, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
                if res.fun < best_shape_side:
                    # Verify with exact objective
                    alpha, beta = res.x
                    alpha_rad = math.radians(alpha)
                    cos_a = math.cos(alpha_rad)
                    sin_a = math.sin(alpha_rad)
                    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    rotated_pos = pos @ rot.T
                    angles_arr = np.full(N, alpha)
                    exact_side = calc_min_outer_side(rotated_pos, angles_arr, beta)
                    if exact_side < best_shape_side:
                        best_shape_side = exact_side
                        best_alpha = alpha
                        best_beta = beta
            except Exception:
                pass
        
        if best_shape_side < best_side:
            best_side = best_shape_side
            alpha_rad = math.radians(best_alpha)
            cos_a = math.cos(alpha_rad)
            sin_a = math.sin(alpha_rad)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            final_pos = pos @ rot.T
            final_angles = np.full(N, best_alpha)
            best_positions = final_pos
            best_angles = final_angles
            best_outer_angle = best_beta

    # Fallback
    if best_positions is None:
        pos = np.array([(i * 1.5, 0) for i in range(N)])
        best_positions = pos
        best_angles = np.zeros(N)
        best_side = calc_min_outer_side(pos, best_angles, 0.0)
        best_outer_angle = 0.0

    # Verify disjointness and repair if needed
    def sat_gap(poly1, poly2):
        min_gap = float('inf')
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        return min_gap

    if best_positions is not None:
        overlap_found = False
        for i in range(N):
            for j in range(i+1, N):
                v1 = get_hex_verts(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i])
                v2 = get_hex_verts(best_positions[j][0], best_positions[j][1], 1.0, best_angles[j])
                if sat_gap(v1, v2) < -1e-5:
                    overlap_found = True
                    break
            if overlap_found:
                break
        
        if overlap_found:
            # Scale positions to separate
            for scale in [1.01, 1.02, 1.05, 1.1]:
                scaled_pos = best_positions * scale
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        v1 = get_hex_verts(scaled_pos[i][0], scaled_pos[i][1], 1.0, best_angles[i])
                        v2 = get_hex_verts(scaled_pos[j][0], scaled_pos[j][1], 1.0, best_angles[j])
                        if sat_gap(v1, v2) < -1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_positions = scaled_pos
                    best_side = calc_min_outer_side(best_positions, best_angles, best_outer_angle)
                    break

    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    return inner, (0.0, 0.0), best_side, best_outer_angle
