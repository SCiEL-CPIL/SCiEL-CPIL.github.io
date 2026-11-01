"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def hexagon_vertices(center, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    cx, cy = center
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def voronoi_min_distance(centers, outer_center, outer_side, outer_angle):
    """
    Compute the minimum distance from outer hexagon boundary to any inner hexagon.
    Uses a simplified approach: for each inner hexagon, find its furthest vertex
    from outer center, and check if it's inside the outer hexagon.
    Returns (min_clearance, all_clearances) where min_clearance < 0 means overlap.
    """
    ocx, ocy = outer_center
    oa = np.radians(outer_angle)
    
    # Outer hexagon vertices
    outer_verts = hexagon_vertices(outer_center, outer_side, outer_angle)
    
    min_clearance = float('inf')
    all_clearances = []
    
    for i, (px, py, pa) in enumerate(centers):
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        
        # For each inner vertex, check distance to outer hexagon boundary
        for v in inner_verts:
            # Check if vertex is inside outer hexagon
            # Compute signed distance to each edge of outer hexagon
            min_dist_to_edge = float('inf')
            for j in range(6):
                v1 = outer_verts[j]
                v2 = outer_verts[(j + 1) % 6]
                # Edge vector
                edge = v2 - v1
                edge_len = np.linalg.norm(edge)
                if edge_len < 1e-12:
                    continue
                edge_norm = edge / edge_len
                # Normal pointing inward (toward center)
                n = np.array([-edge_norm[1], edge_norm[0]])
                # Check which side
                to_center = np.array(outer_center) - v1
                if np.dot(n, to_center) < 0:
                    n = -n
                # Signed distance from vertex to this edge
                d = np.dot(v - v1, n)
                min_dist_to_edge = min(min_dist_to_edge, d)
            
            all_clearances.append(min_dist_to_edge)
            min_clearance = min(min_clearance, min_dist_to_edge)
    
    return min_clearance, all_clearances

def check_validity(centers, outer_center, outer_side, outer_angle):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    # Check containment
    min_clear, _ = voronoi_min_distance(centers, outer_center, outer_side, outer_angle)
    if min_clear < -1e-9:
        return False
    
    # Check disjointness using SAT
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if not sat_disjoint(centers[i], centers[j]):
                return False
    return True

def sat_disjoint(c1, c2):
    """Separating Axis Theorem check for two hexagons."""
    v1 = hexagon_vertices((c1[0], c1[1]), 1.0, c1[2])
    v2 = hexagon_vertices((c2[0], c2[1]), 1.0, c2[2])
    
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            n = np.array([-edge[1], edge[0]]) / edge_len
            axes.append(n)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return True  # Separated
    return False  # Overlapping

def objective(params):
    """
    Objective: minimize outer_side subject to constraints.
    params: [px1, py1, pa1, px2, py2, pa2, ..., pxN, pyN, paN, ocx, ocy, oa]
    """
    n = N
    inner_params = params[:3 * n]
    outer_params = params[3 * n:]
    
    centers = []
    for i in range(n):
        centers.append([inner_params[3 * i], inner_params[3 * i + 1], inner_params[3 * i + 2]])
    
    outer_center = (outer_params[0], outer_params[1])
    outer_angle = outer_params[2]
    
    # Compute required outer side length
    min_clear, _ = voronoi_min_distance(centers, outer_center, 1.0, outer_angle)
    # We need outer_side such that min_clear >= 0
    # So outer_side = 1.0 - min_clear (approximately, need refinement)
    # Better: binary search or direct computation
    
    # Direct computation: find smallest S such that all inner vertices are inside
    # For a regular hexagon centered at origin with side S, a point (x,y) is inside
    # if max over 6 directions of |dot(point, dir)| <= S * cos(30°) = S * sqrt(3)/2
    
    max_extent = 0.0
    for i, (px, py, pa) in enumerate(centers):
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        for v in inner_verts:
            rel = v - np.array(outer_center)
            # Rotate to align with outer hexagon
            oa = np.radians(outer_angle)
            cos_a = np.cos(-oa)
            sin_a = np.sin(-oa)
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            # For regular hexagon with side S, inscribed radius = S * sqrt(3)/2
            # A point (x,y) is inside if for all 6 edge normals, dot <= S*sqrt(3)/2
            # Edge normals at angles 0, 60, 120, 180, 240, 300 degrees
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rx * nx + ry * ny)
                max_extent = max(max_extent, d)
    
    required_side = max_extent / (np.sqrt(3) / 2)
    
    # Penalty for overlaps
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if not sat_disjoint(centers[i], centers[j]):
                # Compute overlap amount
                v1 = hexagon_vertices((centers[i][0], centers[i][1]), 1.0, centers[i][2])
                v2 = hexagon_vertices((centers[j][0], centers[j][1]), 1.0, centers[j][2])
                # Approximate overlap as negative of min separation
                axes = []
                for v in [v1, v2]:
                    for k in range(6):
                        edge = v[(k + 1) % 6] - v[k]
                        edge_len = np.linalg.norm(edge)
                        if edge_len < 1e-12:
                            continue
                        nrm = np.array([-edge[1], edge[0]]) / edge_len
                        axes.append(nrm)
                
                min_separation = float('inf')
                for axis in axes:
                    proj1 = np.dot(v1, axis)
                    proj2 = np.dot(v2, axis)
                    sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
                    min_separation = min(min_separation, sep)
                
                if min_separation < 0:
                    overlap_penalty += (-min_separation) ** 2
    
    return required_side + 100.0 * overlap_penalty

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    # Multiple diverse initial configurations
    configs = []
    
    # Config 1: Hexagonal lattice
    centers1 = []
    dx = 2.0 * np.cos(np.radians(30))
    dy = 2.0
    positions = [
        (0, 0), (dx, dy/2), (-dx, dy/2), (2*dx, 0), (-2*dx, 0),
        (dx, -dy/2), (-dx, -dy/2), (3*dx, dy/2), (-3*dx, dy/2),
        (2*dx, dy), (-2*dx, dy), (0, dy)
    ]
    for (x, y) in positions:
        centers1.append([x, y, 0.0])
    configs.append(centers1)
    
    # Config 2: Staggered rows
    centers2 = []
    row_y = [0, 1.5, 3.0]
    for r, y in enumerate(row_y):
        offset = 0 if r % 2 == 0 else np.sqrt(3)
        for c in range(4):
            x = c * 2.0 + offset - 3.0
            centers2.append([x, y, 0.0])
    configs.append(centers2)
    
    # Config 3: Random
    np.random.seed(42)
    centers3 = []
    for i in range(N):
        x = np.random.uniform(-3, 3)
        y = np.random.uniform(-3, 3)
        a = np.random.uniform(0, 60)
        centers3.append([x, y, a])
    configs.append(centers3)
    
    # Config 4: Circular arrangement
    centers4 = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a = np.degrees(theta)
        centers4.append([x, y, a])
    configs.append(centers4)
    
    for config in configs:
        if time.time() - start_time > DEADLINE:
            break
        
        # Initial parameter vector
        params = []
        for (x, y, a) in config:
            params.extend([x, y, a])
        params.extend([0.0, 0.0, 0.0])  # outer center and angle
        
        # Optimize
        try:
            result = minimize(objective, params, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if result.success or result.fun < best_score:
                opt_params = result.x
                n = N
                inner_opt = []
                for i in range(n):
                    inner_opt.append([opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]])
                ocx = opt_params[3*n]
                ocy = opt_params[3*n+1]
                oa = opt_params[3*n+2]
                
                # Compute actual required side
                max_extent = 0.0
                for (px, py, pa) in inner_opt:
                    inner_verts = hexagon_vertices((px, py), 1.0, pa)
                    for v in inner_verts:
                        rel = v - np.array([ocx, ocy])
                        oa_rad = np.radians(oa)
                        cos_a = np.cos(-oa_rad)
                        sin_a = np.sin(-oa_rad)
                        rx = rel[0] * cos_a - rel[1] * sin_a
                        ry = rel[0] * sin_a + rel[1] * cos_a
                        for k in range(6):
                            theta = np.radians(60 * k)
                            nx = np.cos(theta)
                            ny = np.sin(theta)
                            d = abs(rx * nx + ry * ny)
                            max_extent = max(max_extent, d)
                
                required_side = max_extent / (np.sqrt(3) / 2)
                
                # Validate
                if check_validity(inner_opt, (ocx, ocy), required_side, oa):
                    if required_side < best_score:
                        best_score = required_side
                        best_result = (inner_opt, (ocx, ocy), required_side, oa)
        except Exception:
            pass
    
    # Fallback: return best found or a valid default
    if best_result is not None:
        return best_result
    
    # Default valid packing (larger outer hexagon)
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.0 * np.cos(theta)
        y = 3.0 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    # Compute required side for default
    max_extent = 0.0
    for (px, py, pa) in inner_default:
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        for v in inner_verts:
            rel = v - np.array([0.0, 0.0])
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rel[0] * nx + rel[1] * ny)
                max_extent = max(max_extent, d)
    
    required_side = max_extent / (np.sqrt(3) / 2)
    
    return inner_default, (0.0, 0.0), required_side, 0.0
# EVOLVE-BLOCK-END
