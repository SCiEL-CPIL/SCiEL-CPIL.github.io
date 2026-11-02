import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Helper to get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def hex_vertices(cx, cy, angle_deg=0.0):
        angle = np.deg2rad(angle_deg)
        base_vertices = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        rotated = base_vertices @ rot_matrix.T
        return rotated + np.array([cx, cy])

    def min_dist_polygons(v1, v2):
        """
        v1, v2: arrays of shape (6, 2)
        Returns minimum distance between the two convex polygons.
        Returns 0 if they overlap or intersect.
        """
        # Check for overlap using Separating Axis Theorem (SAT) for convex polygons
        # If no separating axis is found, they overlap.
        def project_poly(poly, axis):
            projections = np.dot(poly, axis)
            return np.min(projections), np.max(projections)
        
        axes = []
        # Normals of edges of v1
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        # Normals of edges of v2
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
                
        for axis in axes:
            min1, max1 = project_poly(v1, axis)
            min2, max2 = project_poly(v2, axis)
            if max1 < min2 or max2 < min1:
                # Separating axis found, no overlap
                # Distance is the gap
                gap = max(min2 - max1, min1 - max2)
                return gap if gap > 0 else 0.0
                
        # If no separating axis, they overlap
        return 0.0

    def required_outer_side(inner_positions, inner_angles, outer_center, outer_angle_deg=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at outer_center
        with rotation outer_angle_deg that contains all inner hexagons.
        """
        oc = np.array(outer_center)
        oa = np.deg2rad(outer_angle_deg)
        
        normals = []
        for k in range(6):
            angle = oa + np.deg2rad(k * 60)
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_proj = 0.0
        for i in range(len(inner_positions)):
            cx, cy = inner_positions[i]
            ang = inner_angles[i]
            verts = hex_vertices(cx, cy, ang)
            for v in verts:
                rel_v = v - oc
                for n in normals:
                    proj = np.dot(rel_v, n)
                    if proj > max_proj:
                        max_proj = proj
        
        if max_proj <= 0:
            return 0.0
        
        apothem_required = max_proj
        side_length = apothem_required / (np.sqrt(3) / 2)
        return side_length

    def check_validity(inner_positions, inner_angles, tol=1e-6):
        """
        Check if inner hexagons are disjoint.
        """
        verts_list = []
        for i in range(len(inner_positions)):
            cx, cy = inner_positions[i]
            ang = inner_angles[i]
            verts_list.append(hex_vertices(cx, cy, ang))
        
        for i in range(len(verts_list)):
            for j in range(i + 1, len(verts_list)):
                d = min_dist_polygons(verts_list[i], verts_list[j])
                if d < tol:
                    return False
        return True

    def objective(vars, outer_angle=0.0):
        """
        Objective function for optimization.
        vars: [x1, y1, a1, ..., x12, y12, a12]
        Returns outer side length + penalty for overlaps.
        """
        pos = [(vars[i*3], vars[i*3+1]) for i in range(N)]
        ang = [vars[i*3+2] for i in range(N)]
        
        # Compute outer side length
        s = required_outer_side(pos, ang, (0,0), outer_angle)
        
        # Compute penalty for overlaps
        penalty = 0.0
        verts_list = [hex_vertices(pos[i][0], pos[i][1], ang[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                d = min_dist_polygons(verts_list[i], verts_list[j])
                if d < 0.01: # Small tolerance for near-overlaps
                    # Penalty scales with how much they overlap (approximated by 0.01 - d)
                    # Actually, if d < 0, they are deeply overlapping. 
                    # min_dist_polygons returns 0 for overlap.
                    # We need a smooth penalty.
                    # Let's use a fixed large penalty if they overlap.
                    # Or better, use the distance to a "buffer" zone.
                    # If d < 0.05, add penalty.
                    penalty += (0.05 - d) ** 2 * 1000
        
        return s + penalty

    def refine_config(config, outer_angle=0.0, max_iter=100):
        """
        Refine a given configuration using L-BFGS-B.
        config: (pos, ang)
        """
        x0 = []
        for i in range(N):
            x0.append(config[0][i][0])
            x0.append(config[0][i][1])
            x0.append(config[1][i])
            
        # Initial score
        init_score = objective(x0, outer_angle)
        
        try:
            res = minimize(objective, x0, args=(outer_angle,), method='L-BFGS-B', options={'maxiter': max_iter, 'disp': False})
            if res.fun < init_score:
                pos_opt = [(res.x[i*3], res.x[i*3+1]) for i in range(N)]
                ang_opt = [res.x[i*3+2] for i in range(N)]
                # Check validity strictly
                if check_validity(pos_opt, ang_opt):
                    s_opt = required_outer_side(pos_opt, ang_opt, (0,0), outer_angle)
                    return (pos_opt, ang_opt, s_opt)
        except Exception:
            pass
        return None

    # Strategy: Try various structured initial configurations and refine them.
    # We will also try different outer hexagon orientations.
    
    best_config = None
    best_score = float('inf')
    
    # Generate initial configurations
    
    # 1. 3x4 Grid
    pos_3x4 = []
    angles_3x4 = []
    for r in range(3):
        for c in range(4):
            x = c * 1.5
            y = r * np.sqrt(3)
            if r % 2 == 1:
                x += 0.75
            pos_3x4.append((x, y))
            angles_3x4.append(0.0)
    cx_grid = np.mean([p[0] for p in pos_3x4])
    cy_grid = np.mean([p[1] for p in pos_3x4])
    pos_3x4 = [(x - cx_grid, y - cy_grid) for x, y in pos_3x4]
    
    # 2. 4x3 Grid
    pos_4x3 = []
    angles_4x3 = []
    for r in range(4):
        for c in range(3):
            x = c * 1.5
            y = r * np.sqrt(3)
            if r % 2 == 1:
                x += 0.75
            pos_4x3.append((x, y))
            angles_4x3.append(0.0)
    cx_grid = np.mean([p[0] for p in pos_4x3])
    cy_grid = np.mean([p[1] for p in pos_4x3])
    pos_4x3 = [(x - cx_grid, y - cy_grid) for x, y in pos_4x3]
    
    # 3. 2x6 Grid
    pos_2x6 = []
    angles_2x6 = []
    for r in range(2):
        for c in range(6):
            x = c * 1.5
            y = r * np.sqrt(3)
            if r % 2 == 1:
                x += 0.75
            pos_2x6.append((x, y))
            angles_2x6.append(0.0)
    cx_grid = np.mean([p[0] for p in pos_2x6])
    cy_grid = np.mean([p[1] for p in pos_2x6])
    pos_2x6 = [(x - cx_grid, y - cy_grid) for x, y in pos_2x6]
    
    # 4. Hexagonal Cluster (1 center, 6 ring, 5 more)
    # Let's generate a hex lattice and pick 12 points
    lattice_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = 1.5 * q
            y = np.sqrt(3) * (r + q/2.0)
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    closest_12 = lattice_points[:12]
    cx_c = np.mean([p[0] for p in closest_12])
    cy_c = np.mean([p[1] for p in closest_12])
    pos_c = [(x - cx_c, y - cy_c) for x, y in closest_12]
    angles_c = [0.0]*12
    
    initial_configs = [
        (pos_3x4, angles_3x4),
        (pos_4x3, angles_4x3),
        (pos_2x6, angles_2x6),
        (pos_c, angles_c)
    ]
    
    # Try refining each config with different outer angles
    outer_angles_to_try = [0.0, 30.0, 15.0, 45.0]
    
    for config in initial_configs:
        if time.time() > deadline - 10:
            break
        for oa in outer_angles_to_try:
            if time.time() > deadline - 5:
                break
            # Quick check
            if not check_validity(config[0], config[1]):
                continue
            s_init = required_outer_side(config[0], config[1], (0,0), oa)
            
            # Refine
            refined = refine_config(config, oa, max_iter=50)
            if refined:
                pos_r, ang_r, s_r = refined
                if s_r < best_score:
                    best_score = s_r
                    best_config = (pos_r, ang_r, (0.0, 0.0), oa)
            else:
                # If refinement failed, use initial if better
                if s_init < best_score:
                    best_score = s_init
                    best_config = (config[0], config[1], (0.0, 0.0), oa)

    if best_config is None:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    # Format output
    inner = []
    for i in range(N):
        inner.append([best_config[0][i][0], best_config[0][i][1], np.rad2deg(best_config[1][i])])
    
    return inner, best_config[2], best_score, best_config[3]
# EVOLVE-BLOCK-END
