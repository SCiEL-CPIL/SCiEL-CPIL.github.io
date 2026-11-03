import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    # Precompute unit hexagon vertices (pointy-top orientation: vertices at 30, 90, 150, 210, 270, 330 degrees)
    # This orientation has flat sides on top and bottom if we consider the "horizontal" direction,
    # but let's stick to a standard definition.
    # Let's use vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    # Vertex k is at (cos(k*pi/3), sin(k*pi/3)).
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
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals to the sides of the outer hexagon
        # If outer hexagon is rotated by theta, its side normals are at theta + 30 + k*60
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3), 
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        # Side length s relates to inradius r by r = s * sqrt(3) / 2
        # So s = 2 * r / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3)

    def hexagons_overlap(verts1, verts2, tol=1e-9):
        """Check if two convex polygons overlap using separation axis theorem."""
        axes = []
        for i in range(len(verts1)):
            j = (i + 1) % len(verts1)
            edge = verts1[j] - verts1[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > tol:
                axes.append(normal / np.linalg.norm(normal))
        for i in range(len(verts2)):
            j = (i + 1) % len(verts2)
            edge = verts2[j] - verts2[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > tol:
                axes.append(normal / np.linalg.norm(normal))
        
        for axis in axes:
            proj1 = verts1 @ axis
            proj2 = verts2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # If there's a gap, no overlap
            if max1 + tol < min2 or max2 + tol < min1:
                return False
        return True

    def check_validity(positions, angles, outer_side, outer_center, outer_angle):
        """
        Check if the configuration is valid:
        1. All inner hexagons are inside the outer hexagon.
        2. No inner hexagons overlap.
        """
        # Check inner-inner overlap
        for i in range(N):
            for j in range(i+1, N):
                v1 = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
                v2 = get_hex_verts(positions[j][0], positions[j][1], 1.0, angles[j])
                if hexagons_overlap(v1, v2):
                    return False
        
        # Check inner-outer containment
        # Outer hexagon is centered at outer_center, side length outer_side, rotation outer_angle
        # Inradius of outer hexagon: R = outer_side * sqrt(3) / 2
        R = outer_side * sqrt3 / 2
        theta_rad = math.radians(outer_angle)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3), 
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        
        for i in range(N):
            v = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
            # Translate vertices relative to outer center
            v_rel = v - np.array(outer_center)
            # Project onto normals
            projs = v_rel @ normals.T
            # All projections must be <= R
            if np.max(projs) > R + 1e-9:
                return False
        return True

    # Strategy:
    # 1. Generate candidate polyhex shapes (connected sets of 12 hexagons on honeycomb lattice).
    # 2. For each shape, optimize global rotation (alpha) and outer rotation (beta).
    # 3. Use Nelder-Mead for optimization.
    # 4. Verify validity and repair if necessary.
    
    # Honeycomb lattice basis
    # v1 = (1.5, sqrt(3)/2)
    # v2 = (0, sqrt(3))
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
            neighbors = [
                (q+1, r), (q-1, r),
                (q, r+1), (q-1, r-1),
                (q+1, r-1), (q-1, r+1)
            ]
            for nq, nr in neighbors:
                if (nq, nr) in coord_set and (nq, nr) not in visited:
                    visited.add((nq, nr))
                    stack.append((nq, nr))
        return len(visited) == len(coords)

    def normalize_shape(coords):
        xs = [lattice_to_cartesian(q, r)[0] for q, r in coords]
        ys = [lattice_to_cartesian(q, r)[1] for q, r in coords]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        return [(lattice_to_cartesian(q, r)[0] - cx, lattice_to_cartesian(q, r)[1] - cy) for q, r in coords]

    candidate_shapes = []

    # Shape 1: 2x3 Rectangle
    shape_2x3 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)]
    if is_connected(shape_2x3):
        candidate_shapes.append(normalize_shape(shape_2x3))
        
    # Shape 2: 3x2 Rectangle
    shape_3x2 = [(0,0), (1,0), (0,1), (1,1), (0,2), (1,2)]
    if is_connected(shape_3x2):
        candidate_shapes.append(normalize_shape(shape_3x2))

    # Shape 3: 4x3 Rectangle (12 cells)
    shape_4x3 = [(i, j) for i in range(4) for j in range(3)]
    if is_connected(shape_4x3):
        candidate_shapes.append(normalize_shape(shape_4x3))

    # Shape 4: 6x2 Rectangle
    shape_6x2 = [(i, j) for i in range(6) for j in range(2)]
    if is_connected(shape_6x2):
        candidate_shapes.append(normalize_shape(shape_6x2))

    # Shape 5: 12x1 Line
    shape_12x1 = [(i, 0) for i in range(12)]
    if is_connected(shape_12x1):
        candidate_shapes.append(normalize_shape(shape_12x1))

    # Shape 6: "Star" or compact blob
    # Let's generate some random polyhexes
    def generate_random_polyhex(size):
        coords = {(0,0)}
        frontier = [(0,0)]
        while len(coords) < size:
            if not frontier:
                return None
            import random
            random.shuffle(frontier)
            q, r = frontier[0]
            neighbors = [
                (q+1, r), (q-1, r),
                (q, r+1), (q-1, r-1),
                (q+1, r-1), (q-1, r+1)
            ]
            import random
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

    for _ in range(50):
        if time.time() > deadline - 30: break
        shape = generate_random_polyhex(N)
        if shape:
            candidate_shapes.append(normalize_shape(shape))
            
    # Remove duplicates
    unique_shapes = []
    seen = set()
    for shape in candidate_shapes:
        key = tuple(sorted((round(x, 3), round(y, 3)) for x, y in shape))
        if key not in seen:
            seen.add(key)
            unique_shapes.append(shape)

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0
    best_outer_center = (0.0, 0.0)

    # For each shape, optimize alpha (cluster rotation) and beta (outer rotation)
    for shape in unique_shapes:
        if time.time() > deadline - 10: break
        
        pos = np.array(shape)
        
        def objective(params):
            alpha, beta = params
            alpha_rad = math.radians(alpha)
            cos_a = math.cos(alpha_rad)
            sin_a = math.sin(alpha_rad)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_pos = pos @ rot.T
            return calc_min_outer_side(rotated_pos, np.full(N, alpha), beta)
            
        # Try multiple starting points
        starts = [(0, 0), (15, 0), (30, 0), (45, 0), (0, 15), (15, 15), (30, 15), (45, 15), (0, 30), (15, 30), (30, 30), (45, 30)]
        
        best_shape_side = float('inf')
        best_alpha = 0
        best_beta = 0
        
        for start in starts:
            try:
                res = minimize(objective, start, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
                if res.fun < best_shape_side:
                    best_shape_side = res.fun
                    best_alpha, best_beta = res.x
            except Exception:
                pass
        
        if best_shape_side < best_side:
            best_side = best_shape_side
            best_alpha, best_beta = best_alpha, best_beta
            
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

    # Verify and repair
    if best_positions is not None:
        overlap_found = False
        for i in range(N):
            for j in range(i+1, N):
                v1 = get_hex_verts(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i])
                v2 = get_hex_verts(best_positions[j][0], best_positions[j][1], 1.0, best_angles[j])
                if hexagons_overlap(v1, v2):
                    overlap_found = True
                    break
            if overlap_found:
                break
        
        if overlap_found:
            # Scale up positions
            scale = 1.0 + 0.01
            best_positions = best_positions * scale
            best_side = calc_min_outer_side(best_positions, best_angles, best_outer_angle)

    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    return inner, (0.0, 0.0), best_side, best_outer_angle
