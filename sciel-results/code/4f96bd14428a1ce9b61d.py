"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

# Pre-compute hexagon vertices for a unit regular hexagon (side length 1)
# A regular hexagon with side length s has vertices at distance s from center
# at angles 0, 60, 120, 180, 240, 300 degrees
_HEX_VERTS_UNIT = np.array([
    [np.cos(np.radians(a)), np.sin(np.radians(a))]
    for a in range(0, 360, 60)
])

def _rotate_points(pts, angle_deg):
    """Rotate points by angle_deg degrees around origin."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T

def _hexagon_vertices(cx, cy, side, angle_deg):
    """Get vertices of a regular hexagon with given center, side length, and rotation."""
    verts = _HEX_VERTS_UNIT * side
    verts = _rotate_points(verts, angle_deg)
    return verts + np.array([cx, cy])

def _point_in_hexagon(point, cx, cy, side, angle_deg):
    """Check if a point is inside or on the boundary of a hexagon."""
    # Transform point to hexagon's local coordinate system
    local = point - np.array([cx, cy])
    theta = np.radians(angle_deg)
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    local = R @ local
    
    # In local coords, hexagon has vertices at (side*cos(k*60), side*sin(k*60))
    # A point is inside if it satisfies all 6 half-plane inequalities
    # The edges are between consecutive vertices
    # For a regular hexagon with circumradius = side (since for regular hexagon, 
    # the circumradius equals the side length), the apothem = side * sqrt(3)/2
    # The 6 edges are at distance apothem from center, at angles 30, 90, 150, 210, 270, 330
    
    apothem = side * np.sqrt(3) / 2
    for k in range(6):
        edge_angle = np.radians(30 + 60 * k)
        normal = np.array([np.cos(edge_angle), np.sin(edge_angle)])
        dist = local @ normal
        if dist > apothem + 1e-9:
            return False
    return True

def _hexagon_distance(p1, c1, a1, p2, c2, a2):
    """Compute minimum distance between two hexagons (approximate via vertex-to-edge).
    Returns negative if overlapping, positive if separated."""
    # For simplicity, use a conservative check: 
    # Two hexagons overlap if any vertex of one is inside the other,
    # or if any edge of one intersects any edge of the other.
    # For now, use a simpler proxy: check if centers are too close.
    # Better: check all vertices of hex1 against hex2 and vice versa.
    
    v1 = _hexagon_vertices(p1[0], p1[1], 1.0, a1)
    v2 = _hexagon_vertices(p2[0], p2[1], 1.0, a2)
    
    # Check if any vertex of v1 is inside hex2
    for pt in v1:
        if _point_in_hexagon(pt, p2[0], p2[1], 1.0, a2):
            return -1.0  # Overlapping
    
    # Check if any vertex of v2 is inside hex1
    for pt in v2:
        if _point_in_hexagon(pt, p1[0], p1[1], 1.0, a1):
            return -1.0  # Overlapping
    
    # Check edge intersections (simplified: just check if edges are close)
    # For a more robust check, we'd need proper segment intersection
    # For now, if no vertices are inside, we assume they might still overlap
    # via edge crossing. Let's do a quick check by sampling edge midpoints.
    
    # Actually, let's compute the actual minimum distance more carefully
    # by checking all vertex-edge pairs
    
    min_dist = np.inf
    
    # Vertices of hex1 to edges of hex2
    for pt in v1:
        for i in range(6):
            p_a = v2[i]
            p_b = v2[(i + 1) % 6]
            # Distance from point pt to segment p_a-p_b
            ab = p_b - p_a
            t = np.dot(pt - p_a, ab) / (np.dot(ab, ab) + 1e-12)
            t = np.clip(t, 0, 1)
            closest = p_a + t * ab
            dist = np.linalg.norm(pt - closest)
            min_dist = min(min_dist, dist)
    
    # Vertices of hex2 to edges of hex1
    for pt in v2:
        for i in range(6):
            p_a = v1[i]
            p_b = v1[(i + 1) % 6]
            ab = p_b - p_a
            t = np.dot(pt - p_a, ab) / (np.dot(ab, ab) + 1e-12)
            t = np.clip(t, 0, 1)
            closest = p_a + t * ab
            dist = np.linalg.norm(pt - closest)
            min_dist = min(min_dist, dist)
    
    if min_dist < 1e-9:
        return -1.0  # Touching or overlapping
    return min_dist

def _fits_in_outer(inner_positions, inner_angles, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons fit inside the outer hexagon and don't overlap."""
    # Check each inner hexagon fits in outer
    for i, (pos, ang) in enumerate(zip(inner_positions, inner_angles)):
        v_inner = _hexagon_vertices(pos[0], pos[1], 1.0, ang)
        for pt in v_inner:
            if not _point_in_hexagon(pt, outer_center[0], outer_center[1], outer_side, outer_angle):
                return False
    
    # Check no two inner hexagons overlap
    n = len(inner_positions)
    for i in range(n):
        for j in range(i + 1, n):
            d = _hexagon_distance(inner_positions[i], inner_angles[i], 
                                   inner_positions[j], inner_angles[j])
            if d < -1e-9:
                return False
    
    return True

def _compute_outer_side(inner_positions, inner_angles):
    """Compute the minimum side length of a regular hexagon (with center at origin, 
    variable rotation) that contains all inner hexagons."""
    # For a given outer rotation, we need to find the minimum side length
    # such that all vertices of all inner hexagons are inside the outer hexagon.
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for pos, ang in zip(inner_positions, inner_angles):
        v = _hexagon_vertices(pos[0], pos[1], 1.0, ang)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    # For a hexagon with center at origin, side length S, rotation theta:
    # A point (x,y) is inside if for all k=0..5:
    #   (x,y) dot (cos(30+60k+theta), sin(30+60k+theta)) <= S * sqrt(3)/2
    # So S >= 2/sqrt(3) * max over all points and all k of [point dot normal_k]
    
    best_side = np.inf
    # Try several outer rotations
    for outer_angle in np.linspace(0, 60, 13):  # 0 to 60 degrees, 13 points
        theta = np.radians(outer_angle)
        max_proj = 0
        for k in range(6):
            edge_angle = np.radians(30 + 60 * k) + theta
            normal = np.array([np.cos(edge_angle), np.sin(edge_angle)])
            projs = all_verts @ normal
            max_proj = max(max_proj, np.max(projs))
        side = 2.0 / np.sqrt(3) * max_proj
        best_side = min(best_side, side)
    
    return best_side

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20 second safety margin
    
    N = 11  # Number of inner hexagons
    
    # Initial configuration: place hexagons in a line
    # A row of N unit hexagons side by side has width N * 2 (since each hex has width 2)
    # and height sqrt(3)
    # For N=11, width = 22, height = sqrt(3)
    # The outer hexagon needs to contain this.
    
    # Let's try a few configurations:
    # 1. Linear arrangement
    # 2. Hexagonal packing
    
    best_result = None
    best_score = np.inf
    
    # Configuration 1: Linear arrangement along x-axis
    # Hexagons at x = 2*k for k in range(-(N//2), N-N//2)
    positions = []
    for k in range(-(N // 2), N - N // 2):
        positions.append([2.0 * k, 0.0])
    angles = [0.0] * N
    
    outer_side = _compute_outer_side(np.array(positions), angles)
    score = outer_side
    
    if best_result is None or score < best_score:
        best_score = score
        best_result = (positions, angles, score)
    
    # Try rotating the whole configuration
    for rot in [0, 30, 60, 90, 120, 150]:
        theta = np.radians(rot)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated_positions = [R @ np.array(p) for p in positions]
        rotated_angles = [(a + rot) % 360 for a in angles]
        
        outer_side = _compute_outer_side(np.array(rotated_positions), rotated_angles)
        score = outer_side
        
        if best_result is None or score < best_score:
            best_score = score
            best_result = (rotated_positions, rotated_angles, score)
    
    # Configuration 2: Hexagonal packing (rows offset)
    # Row 0: y=0, x = 2*k
    # Row 1: y=sqrt(3), x = 2*k + 1
    # etc.
    
    # For 11 hexagons, try a 3-4-3 or 2-4-3-2 arrangement
    configs = []
    
    # 3-4-3 arrangement
    pos343 = []
    for k in range(-1, 2):  # 3 hexagons in row 0
        pos343.append([2.0 * k, 0.0])
    for k in range(-2, 2):  # 4 hexagons in row 1
        pos343.append([2.0 * k + 1.0, np.sqrt(3)])
    for k in range(-1, 2):  # 3 hexagons in row 2
        pos343.append([2.0 * k, 2.0 * np.sqrt(3)])
    configs.append(pos343)
    
    # 2-4-3-2 arrangement
    pos2432 = []
    for k in range(-1, 1):  # 2
        pos2432.append([2.0 * k, 0.0])
    for k in range(-2, 2):  # 4
        pos2432.append([2.0 * k + 1.0, np.sqrt(3)])
    for k in range(-1, 2):  # 3
        pos2432.append([2.0 * k, 2.0 * np.sqrt(3)])
    for k in range(-1, 1):  # 2
        pos2432.append([2.0 * k + 1.0, 3.0 * np.sqrt(3)])
    configs.append(pos2432)
    
    # 1-3-4-3 arrangement
    pos1343 = []
    pos1343.append([0.0, 0.0])
    for k in range(-1, 2):  # 3
        pos1343.append([2.0 * k + 1.0, np.sqrt(3)])
    for k in range(-2, 2):  # 4
        pos1343.append([2.0 * k, 2.0 * np.sqrt(3)])
    for k in range(-1, 2):  # 3
        pos1343.append([2.0 * k + 1.0, 3.0 * np.sqrt(3)])
    configs.append(pos1343)
    
    for config in configs:
        if len(config) != N:
            continue
        angles = [0.0] * N
        outer_side = _compute_outer_side(np.array(config), angles)
        score = outer_side
        
        if best_result is None or score < best_score:
            best_score = score
            best_result = (config, angles, score)
        
        # Try rotating
        for rot in [0, 30, 60, 90, 120, 150]:
            theta = np.radians(rot)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            rotated_positions = [R @ np.array(p) for p in config]
            rotated_angles = [(a + rot) % 360 for a in angles]
            
            outer_side = _compute_outer_side(np.array(rotated_positions), rotated_angles)
            score = outer_side
            
            if best_result is None or score < best_score:
                best_score = score
                best_result = (rotated_positions, rotated_angles, score)
    
    # Check deadline
    if time.time() > deadline:
        # Return best result found
        if best_result is not None:
            positions, angles, score = best_result
            inner = [[p[0], p[1], a] for p, a in zip(positions, angles)]
            return inner, (0.0, 0.0), score, 0.0
        else:
            # Fallback
            inner = [[2.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
    
    # Try to optimize the best configuration using local search
    # This is a simplified optimization - just try small perturbations
    if best_result is not None:
        positions, angles, score = best_result
        positions = [np.array(p) for p in positions]
        angles = list(angles)
        
        # Try small rotations of individual hexagons
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Try rotating one hexagon by a small amount
            idx = trial % N
            for delta_ang in [-15, -10, -5, 5, 10, 15]:
                new_angles = angles[:]
                new_angles[idx] = (angles[idx] + delta_ang) % 360
                
                outer_side = _compute_outer_side(np.array(positions), new_angles)
                if outer_side < score - 1e-9:
                    angles = new_angles
                    score = outer_side
                    best_result = (positions, angles, score)
    
    # Final result
    if best_result is not None:
        positions, angles, score = best_result
        inner = [[float(p[0]), float(p[1]), float(a)] for p, a in zip(positions, angles)]
        return inner, (0.0, 0.0), float(score), 0.0
    else:
        # Fallback
        inner = [[2.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
