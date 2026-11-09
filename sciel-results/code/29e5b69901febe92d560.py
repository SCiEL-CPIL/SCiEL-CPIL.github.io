import numpy as np
import time
import math
import itertools

N = 11
EPS = 1e-10
SQRT3 = math.sqrt(3)

def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def hex_edge_normals(angle_deg):
    """Return the 6 outward edge normals of a regular hexagon rotated by angle_deg."""
    a = math.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + math.pi / 6 + i * math.pi / 3
        normals.append([math.cos(theta), math.sin(theta)])
    return np.array(normals)

def point_in_hex(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) side s rotated angle_deg."""
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-a), math.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    for i in range(6):
        theta = math.pi / 6 + i * math.pi / 3
        if lx * math.cos(theta) + ly * math.sin(theta) > s * SQRT3 / 2 + 1e-12:
            return False
    return True

def hexagons_disjoint_sat(c1, a1, c2, a2, side=1.0):
    """Check if two hexagons are disjoint using Separating Axis Theorem."""
    x1, y1 = c1
    x2, y2 = c2
    
    v1 = hex_vertices(x1, y1, side, a1)
    v2 = hex_vertices(x2, y2, side, a2)
    
    n1 = hex_edge_normals(a1)
    n2 = hex_edge_normals(a2)
    
    for i in range(6):
        proj1 = v1[:, 0] * n1[i, 0] + v1[:, 1] * n1[i, 1]
        proj2 = v2[:, 0] * n1[i, 0] + v2[:, 1] * n1[i, 1]
        if proj1.max() < proj2.min() - EPS or proj2.max() < proj1.min() - EPS:
            return True
    
    for i in range(6):
        proj1 = v1[:, 0] * n2[i, 0] + v1[:, 1] * n2[i, 1]
        proj2 = v2[:, 0] * n2[i, 0] + v2[:, 1] * n2[i, 1]
        if proj1.max() < proj2.min() - EPS or proj2.max() < proj1.min() - EPS:
            return True
    
    return False

def min_enclosing_hex(centers, angles, side=1.0):
    """Compute minimal outer hexagon to contain all inner hexagons.
    Returns (side_length, center_x, center_y, outer_angle_deg)
    """
    all_verts = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = hex_vertices(cx, cy, side, angles[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_side = float('inf')
    best_center = (0, 0)
    best_angle = 0.0
    
    for outer_angle_deg in np.linspace(0, 60, 121):
        a = math.radians(outer_angle_deg)
        normals = []
        for i in range(6):
            theta = a + math.pi / 6 + i * math.pi / 3
            normals.append([math.cos(theta), math.sin(theta)])
        normals = np.array(normals)
        
        # For a regular hexagon centered at (cx,cy) with side s,
        # a point p is inside iff for all 6 normals n_i:
        #   (p - c) · n_i <= s * sqrt(3)/2
        # So: c · n_i >= max_i (p_i · n_i) - s*sqrt(3)/2
        # And: c · n_i <= min_i (p_i · n_i) + s*sqrt(3)/2
        
        # The optimal center minimizes s subject to these constraints.
        # This is a linear program in (cx, cy, s).
        # We solve it by trying the centroid and then refining.
        
        # Simple approach: use the average of max and min projections to find center
        # For each pair of opposite normals, the center projection must be
        # between (max_proj - s*sqrt3/2) and (min_proj + s*sqrt3/2)
        
        # Let's use a more systematic approach:
        # The center should be such that the max over i of [(max_proj_i - c.n_i) + (c.n_i - min_proj_i)] / sqrt3/2 is minimized
        # = max over i of (max_proj_i - min_proj_i) / sqrt3/2 ... no that's not right
        
        # Actually, for a given orientation, the minimal s is:
        # s = max_i [(max_proj_i - min_proj_i)] / sqrt(3)
        # and the center is at the midpoint: c.n_i = (max_proj_i + min_proj_i) / 2
        
        # But we need to find a center c that satisfies ALL 6 constraints simultaneously.
        # c.n_i = (max_proj_i + min_proj_i)/2 for all i may not be consistent.
        
        # Let's use linear programming via scipy
        from scipy.optimize import linprog
        
        # Variables: cx, cy, s
        # Constraints: for each i:
        #   cx * n_i[0] + cy * n_i[1] - s * sqrt3/2 >= max_proj_i  =>  -n_i[0]*cx - n_i[1]*cy + sqrt3/2*s <= -max_proj_i
        #   -cx * n_i[0] - cy * n_i[1] - s * sqrt3/2 >= -min_proj_i  =>  n_i[0]*cx + n_i[1]*cy + sqrt3/2*s <= min_proj_i
        
        A_ub = []
        b_ub = []
        for i in range(6):
            proj = all_verts[:, 0] * normals[i, 0] + all_verts[:, 1] * normals[i, 1]
            pmax = proj.max()
            pmin = proj.min()
            # -n_i[0]*cx - n_i[1]*cy + sqrt3/2*s <= -pmax
            A_ub.append([-normals[i, 0], -normals[i, 1], SQRT3 / 2])
            b_ub.append(-pmax)
            # n_i[0]*cx + n_i[1]*cy + sqrt3/2*s <= pmin
            A_ub.append([normals[i, 0], normals[i, 1], SQRT3 / 2])
            b_ub.append(pmin)
        
        c_obj = [0, 0, 1]  # minimize s
        
        try:
            res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(None, None), (None, None), (0, None)])
            if res.success:
                side_val = res.x[2]
                cx_val = res.x[0]
                cy_val = res.x[1]
                if side_val < best_side:
                    best_side = side_val
                    best_center = (cx_val, cy_val)
                    best_angle = outer_angle_deg
        except:
            pass
    
    return best_side, best_center[0], best_center[1], best_angle

def check_validity(centers, angles, outer_side, outer_center, outer_angle):
    """Check if the configuration is valid."""
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if not hexagons_disjoint_sat(centers[i], angles[i], centers[j], angles[j], 1.0):
                return False
    
    for i in range(len(centers)):
        verts = hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        for v in verts:
            if not point_in_hex(v[0], v[1], outer_center[0], outer_center[1], outer_side, outer_angle):
                return False
    
    return True

def get_valid_fallback():
    """Return a known valid fallback configuration."""
    sqrt3 = math.sqrt(3)
    
    # Hexagonal lattice arrangement for 11 hexagons
    axial_positions = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    
    centers = []
    for q, r in axial_positions:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append((x, y))
    
    angles = [0.0] * N
    
    side, cx, cy, angle = min_enclosing_hex(centers, angles)
    side *= 1.001
    
    return side, centers, angles, (cx, cy), angle

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_side = float('inf')
    
    # First, try the fallback to have a valid solution
    fb_side, fb_centers, fb_angles, fb_center, fb_angle = get_valid_fallback()
    best_side = fb_side
    best_result = (fb_centers, fb_angles, fb_center, fb_angle)
    
    # Try various lattice-based arrangements with different orientations
    sqrt3 = math.sqrt(3)
    
    # Generate different hexagonal lattice configurations
    lattice_configs = []
    
    # Standard hexagonal lattice
    for radius in [2, 3]:
        positions = []
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                if abs(q) + abs(r) + abs(q + r) <= 2 * radius:
                    positions.append((q, r))
        
        # Take the first N positions (closest to center)
        positions.sort(key=lambda p: abs(p[0]) + abs(p[1]) + abs(p[0] + p[1]))
        positions = positions[:N]
        
        centers = []
        for q, r in positions:
            x = 1.5 * q
            y = sqrt3 * (r + 0.5 * q)
            centers.append((x, y))
        lattice_configs.append(centers)
    
    # Try each lattice config with different outer orientations
    for centers in lattice_configs:
        if time.time() > deadline:
            break
        
        for outer_angle in [0, 30]:
            angles = [float(outer_angle)] * N
            side, cx, cy, angle = min_enclosing_hex(centers, angles)
            side *= 1.0005
            
            if check_validity(centers, angles, side, (cx, cy), angle):
                if side < best_side:
                    best_side = side
                    best_result = (centers, angles, (cx, cy), angle)
    
    # Try optimized lattice with relaxed positions
    # Use force-directed relaxation to optimize positions
    np.random.seed(42)
    
    for _ in range(5):
        if time.time() > deadline:
            break
        
        # Start from a random perturbation of the lattice
        base_positions = [
            (0, 0),
            (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
            (2, 0), (2, -1), (1, -2), (0, -2),
        ]
        
        centers = []
        for q, r in base_positions:
            x = 1.5 * q + np.random.uniform(-0.1, 0.1)
            y = sqrt3 * (r + 0.5 * q) + np.random.uniform(-0.1, 0.1)
            centers.append([x, y])
        
        angles = [0.0] * N
        
        # Force-directed relaxation
        for it in range(200):
            if time.time() > deadline:
                break
            
            forces = [[0, 0] for _ in range(N)]
            
            for i in range(N):
                for j in range(i + 1, N):
                    if not hexagons_disjoint_sat(tuple(centers[i]), angles[i], tuple(centers[j]), angles[j], 1.0):
                        dx = centers[j][0] - centers[i][0]
                        dy = centers[j][1] - centers[i][1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist < EPS:
                            dx, dy = 1, 0
                            dist = 1
                        else:
                            dx /= dist
                            dy /= dist
                        
                        f = 0.5 * (1.0 / max(dist, 0.1))
                        forces[i][0] -= dx * f
                        forces[i][1] -= dy * f
                        forces[j][0] += dx * f
                        forces[j][1] += dy * f
            
            for i in range(N):
                centers[i][0] += forces[i][0] * 0.9
                centers[i][1] += forces[i][1] * 0.9
        
        # Optimize outer hexagon
        centers = [tuple(c) for c in centers]
        side, cx, cy, angle = min_enclosing_hex(centers, angles)
        side *= 1.0005
        
        if check_validity(centers, angles, side, (cx, cy), angle):
            if side < best_side:
                best_side = side
                best_result = (centers, angles, (cx, cy), angle)
    
    # Try different inner hexagon orientations
    for inner_angle in [0, 30]:
        if time.time() > deadline:
            break
        
        base_positions = [
            (0, 0),
            (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
            (2, 0), (2, -1), (1, -2), (0, -2),
        ]
        
        centers = []
        for q, r in base_positions:
            x = 1.5 * q
            y = sqrt3 * (r + 0.5 * q)
            centers.append((x, y))
        
        angles = [float(inner_angle)] * N
        
        side, cx, cy, angle = min_enclosing_hex(centers, angles)
        side *= 1.0005
        
        if check_validity(centers, angles, side, (cx, cy), angle):
            if side < best_side:
                best_side = side
                best_result = (centers, angles, (cx, cy), angle)
    
    # If no valid result found, use fallback
    if best_result is None:
        side, centers, angles, center, angle = get_valid_fallback()
        inner = [[float(cx), float(cy), float(ang)] for (cx, cy), ang in zip(centers, angles)]
        return inner, (float(center[0]), float(center[1])), float(side), float(angle)
    
    centers, angles, center, angle = best_result
    inner = [[float(cx), float(cy), float(ang)] for (cx, cy), ang in zip(centers, angles)]
    
    return inner, (float(center[0]), float(center[1])), float(best_side), float(angle)
