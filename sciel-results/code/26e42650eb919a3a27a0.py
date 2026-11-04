import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(cx, cy, side, angle_deg=0):
    """Get vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def polygons_overlap(v1, v2, eps=1e-10):
    """Check if two convex polygons overlap using SAT."""
    for poly in [v1, v2]:
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 + eps or max2 < min1 + eps:
                return False
    return True

def max_overlap_depth(v1, v2):
    """Estimate overlap depth for penalty."""
    # Use center distance as rough proxy
    c1 = v1.mean(axis=0)
    c2 = v2.mean(axis=0)
    return np.linalg.norm(c1 - c2)

def compute_score(positions, outer_center, outer_side, outer_angle):
    """Compute the score (outer side length) if valid, else inf."""
    # Check all inner hexagons don't overlap
    inner_verts = []
    for x, y, ang in positions:
        inner_verts.append(hex_vertices(x, y, 1.0, ang))
    
    for i in range(len(inner_verts)):
        for j in range(i + 1, len(inner_verts)):
            if polygons_overlap(inner_verts[i], inner_verts[j]):
                return np.inf
    
    # Check all inner hexagons are inside outer hexagon
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # For each inner hex vertex, check it's inside outer hex
    for iv in inner_verts:
        for pt in iv:
            # Point in convex polygon test
            inside = True
            for k in range(6):
                v1 = outer_verts[k]
                v2 = outer_verts[(k + 1) % 6]
                cross = np.cross(v2 - v1, pt - v1)
                if cross < -1e-10:
                    inside = False
                    break
            if not inside:
                return np.inf
    
    return outer_side

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Initial guess: hexagonal lattice arrangement
    # For 11 hexagons, use a compact arrangement
    # Flat-top hexagons: spacing dx=1.5, dy=√3, offset=√3/2
    sqrt3 = np.sqrt(3)
    
    # Positions for 11 hexagons in a hexagonal cluster
    # Center + 6 ring + 4 more
    positions = []
    # Center
    positions.append([0.0, 0.0, 0.0])
    # First ring (6 hexagons)
    for i in range(6):
        angle = i * np.pi / 3
        x = 2.0 * np.cos(angle)
        y = 2.0 * np.sin(angle)
        positions.append([x, y, 0.0])
    # Second ring partial (4 more to make 11)
    # Place them at alternating positions in second ring
    for i in range(0, 6, 2):
        if len(positions) >= 11:
            break
        angle = i * np.pi / 3
        x = 4.0 * np.cos(angle)
        y = 4.0 * np.sin(angle)
        positions.append([x, y, 0.0])
    # If we have more than 11, trim
    positions = positions[:11]
    
    # Estimate initial outer hexagon
    # Find bounding box of positions
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    r = max(np.sqrt((x - cx)**2 + (y - cy)**2) for x, y, _ in positions)
    # For a hexagon, side = r / cos(pi/6) approximately for circumscribed
    outer_side_init = r / (np.sqrt(3) / 2) + 1.0  # extra margin
    outer_center = [cx, cy]
    outer_angle = 0.0
    
    # Objective function for optimization
    def objective(params):
        if time.time() > deadline:
            return 1e10
        # Unpack parameters
        # First N*3 params: inner hex positions and angles
        # Next 3 params: outer center x, y, side
        # Last param: outer angle
        inner_params = params[:N * 3].reshape(N, 3)
        ocx = params[N * 3]
        ocy = params[N * 3 + 1]
        oside = params[N * 3 + 2]
        oangle = params[N * 3 + 3]
        
        if oside < 1.0:
            return 1e10
        
        # Compute overlap penalty
        inner_verts = []
        for x, y, ang in inner_params:
            inner_verts.append(hex_vertices(x, y, 1.0, ang))
        
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if polygons_overlap(inner_verts[i], inner_verts[j]):
                    # Estimate penetration depth
                    c1 = inner_verts[i].mean(axis=0)
                    c2 = inner_verts[j].mean(axis=0)
                    dist = np.linalg.norm(c1 - c2)
                    penalty += max(0, 2.0 - dist) ** 2
        
        # Check containment penalty
        outer_verts = hex_vertices(ocx, ocy, oside, oangle)
        for iv in inner_verts:
            for pt in iv:
                inside = True
                for k in range(6):
                    v1 = outer_verts[k]
                    v2 = outer_verts[(k + 1) % 6]
                    cross = np.cross(v2 - v1, pt - v1)
                    if cross < 0:
                        inside = False
                        break
                if not inside:
                    # Distance to boundary approximation
                    penalty += 10.0
        
        return oside + penalty * 10.0
    
    # Initial parameters
    init_params = []
    for x, y, ang in positions:
        init_params.extend([x, y, ang])
    init_params.extend([outer_center[0], outer_center[1], outer_side_init, outer_angle])
    
    # Try Nelder-Mead optimization
    best_result = None
    best_score = np.inf
    
    # First, try to get a valid solution with the initial arrangement
    init_score = compute_score(positions, outer_center, outer_side_init, outer_angle)
    if init_score < np.inf:
        best_score = init_score
        best_result = (positions, outer_center, outer_side_init, outer_angle)
    
    # Run optimization
    try:
        result = minimize(objective, init_params, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if time.time() < deadline:
            params = result.x
            inner_params = params[:N * 3].reshape(N, 3)
            ocx = params[N * 3]
            ocy = params[N * 3 + 1]
            oside = params[N * 3 + 2]
            oangle = params[N * 3 + 3]
            
            pos_list = [[float(p[0]), float(p[1]), float(p[2])] for p in inner_params]
            score = compute_score(pos_list, [ocx, ocy], oside, oangle)
            
            if score < np.inf and score < best_score:
                best_score = score
                best_result = (pos_list, [ocx, ocy], oside, oangle)
                
    except Exception:
        pass
    
    # If no valid result found, use a known valid configuration
    if best_result is None:
        # Use a simple valid configuration: line of hexagons
        # This is not optimal but valid
        positions_valid = [[2.0 * i, 0.0, 0.0] for i in range(N)]
        xs = [p[0] for p in positions_valid]
        cx = (min(xs) + max(xs)) / 2
        r = max(abs(x - cx) for x in xs) + 1.0
        outer_side_valid = r / (np.sqrt(3) / 2) + 0.5
        best_result = (positions_valid, [cx, 0.0], outer_side_valid, 0.0)
        best_score = outer_side_valid
    
    # Verify the best result
    final_score = compute_score(best_result[0], best_result[1], best_result[2], best_result[3])
    if final_score >= np.inf:
        # Fallback to guaranteed valid
        positions_valid = [[2.0 * i, 0.0, 0.0] for i in range(N)]
        xs = [p[0] for p in positions_valid]
        cx = (min(xs) + max(xs)) / 2
        r = max(abs(x - cx) for x in xs) + 1.0
        outer_side_valid = r / (np.sqrt(3) / 2) + 0.5
        return positions_valid, [cx, 0.0], outer_side_valid, 0.0
    
    return best_result[0], best_result[1], best_result[2], best_result[3]
