import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform point to hexagon's local frame
    a = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    # Rotate by -a
    rx = dx * np.cos(-a) - dy * np.sin(-a)
    ry = dx * np.sin(-a) + dy * np.cos(-a)
    # For a regular hexagon with side s, the boundary in local coords:
    # |rx| <= s and |ry| <= s*sqrt(3)/2 and |rx| + |ry|/sqrt(3) <= s*sqrt(3)/2 ... 
    # Actually for a regular hexagon with "radius" (center to vertex) = side:
    # The hexagon has vertices at distance 'side' from center.
    # In local frame, the hexagon is defined by:
    # max(|rx|, |ry|*2/sqrt(3), |rx|/2 + |ry|/sqrt(3)) <= side
    # Let me use the standard: for hex with circumradius R=side:
    # |x| <= R, |y| <= R*sqrt(3)/2, |x| + 2*|y|/sqrt(3) <= R*sqrt(3)/2... 
    # Simpler: use the fact that a regular hexagon with circumradius R has:
    # the six edges at distance R*cos(30) = R*sqrt(3)/2 from center
    # Edge normals at angles 0, 60, 120, 180, 240, 300
    # For each edge: dot(point, normal) <= R*sqrt(3)/2
    R = side
    d = R * np.sqrt(3) / 2  # apothem
    for i in range(6):
        theta = i * np.pi / 3
        nx, ny = np.cos(theta), np.sin(theta)
        if rx * nx + ry * ny > d + 1e-10:
            return False
    return True

def _min_dist_hex_hex(v1, v2):
    """Minimum distance between two hexagons (0 if overlapping)."""
    # Check if any vertex of one is inside the other
    for px, py in v1:
        cx2, cy2, s2, a2 = v2[0], v2[1], v2[2], v2[3]
        if _point_in_hex(px, py, cx2, cy2, s2, a2):
            return 0.0
    for px, py in v2:
        cx1, cy1, s1, a1 = v1[0], v1[1], v1[2], v1[3]
        if _point_in_hex(px, py, cx1, cy1, s1, a1):
            return 0.0
    # If no vertex inside, compute min distance between edges (approximate by vertex-vertex)
    min_d = np.inf
    for p1 in v1:
        for p2 in v2:
            d = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            min_d = min(min_d, d)
    return min_d

def _max_dist_to_center(points, cx, cy):
    """Max distance from center to any point."""
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    return np.sqrt(dx**2 + dy**2).max()

def _required_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    cx, cy = outer_center
    oa = np.radians(outer_angle)
    # For each inner hexagon, find max distance from outer center in each of 6 edge-normal directions
    # Outer hexagon with center (cx,cy), side S, rotation oa contains point p if:
    # For each of 6 directions theta_i = oa + i*pi/3:
    #   dot(p - center, (cos(theta_i), sin(theta_i))) <= S * sqrt(3)/2
    # So required S = max over all points and all 6 directions of dot(p-center, dir) / (sqrt(3)/2)
    
    all_points = []
    for i in range(len(inner_positions)):
        pos = inner_positions[i]
        ang = inner_angles[i]
        verts = _hex_vertices(pos[0], pos[1], 1.0, ang)
        all_points.append(verts)
    all_points = np.vstack(all_points)
    
    max_proj = 0.0
    for i in range(6):
        theta = oa + i * np.pi / 3
        nx, ny = np.cos(theta), np.sin(theta)
        projections = (all_points[:, 0] - cx) * nx + (all_points[:, 1] - cy) * ny
        max_proj = max(max_proj, projections.max())
    
    return max_proj / (np.sqrt(3) / 2)

def _overlap_penalty(inner_positions, inner_angles, eps=0.01):
    """Penalty for overlaps between inner hexagons."""
    penalty = 0.0
    n = len(inner_positions)
    for i in range(n):
        for j in range(i+1, n):
            vi = _hex_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i])
            vj = _hex_vertices(inner_positions[j][0], inner_positions[j][1], 1.0, inner_angles[j])
            # Check vertex-in-hex
            for px, py in vi:
                if _point_in_hex(px, py, inner_positions[j][0], inner_positions[j][1], 1.0, inner_angles[j]):
                    penalty += 100.0
            for px, py in vj:
                if _point_in_hex(px, py, inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i]):
                    penalty += 100.0
    return penalty

def _objective(params, outer_center, outer_angle):
    """Objective: outer hexagon side length + overlap penalty."""
    n = N
    inner_positions = params[:2*n].reshape(n, 2)
    inner_angles = params[2*n:]
    
    side = _required_outer_side(inner_positions, inner_angles, outer_center, outer_angle)
    penalty = _overlap_penalty(inner_positions, inner_angles)
    return side + penalty

def construct():
    start_time = time.time()
    best_result = None
    best_score = float('inf')
    
    # Initial configuration: hexagonal cluster
    # Center at origin, 6 in first ring, 4 more in second ring
    # Use a compact arrangement
    
    # Try a few initial configurations
    initial_configs = []
    
    # Config 1: Center + 6 ring + 4 extra
    config1 = []
    config1.append([0, 0, 0])  # center
    for i in range(6):
        a = i * 60
        config1.append([2.0 * np.cos(np.radians(a)), 2.0 * np.sin(np.radians(a)), 0])
    # Add 4 more in a ring at distance ~3.5
    for i in range(4):
        a = 30 + i * 90
        config1.append([3.5 * np.cos(np.radians(a)), 3.5 * np.sin(np.radians(a)), 0])
    initial_configs.append(config1)
    
    # Config 2: Linear-ish arrangement
    config2 = []
    for i in range(N):
        config2.append([float(i), 0.0, 0.0])
    initial_configs.append(config2)
    
    # Config 3: Two rows
    config3 = []
    for i in range(6):
        config3.append([float(i), 0.0, 0.0])
    for i in range(5):
        config3.append([float(i) + 0.5, np.sqrt(3)/2, 0])
    initial_configs.append(config3)
    
    # Config 4: Hexagonal lattice - center, 6 at dist 2, 6 at dist 4 (but we need 11)
    config4 = []
    config4.append([0, 0, 0])
    for i in range(6):
        a = i * 60
        config4.append([2.0 * np.cos(np.radians(a)), 2.0 * np.sin(np.radians(a)), 0])
    for i in range(4):
        a = 60 + i * 60
        config4.append([4.0 * np.cos(np.radians(a)), 4.0 * np.sin(np.radians(a)), 0])
    initial_configs.append(config4)
    
    for config in initial_configs:
        if time.time() - start_time > DEADLINE:
            break
        
        # Convert to parameter vector
        params0 = []
        for x, y, a in config:
            params0.extend([x, y, a])
        params0 = np.array(params0, dtype=float)
        
        outer_center = [0.0, 0.0]
        outer_angle = 0.0
        
        # Optimize
        try:
            result = minimize(
                _objective,
                params0,
                args=(outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
            )
            
            if time.time() - start_time > DEADLINE:
                break
            
            params_opt = result.x
            inner_positions = params_opt[:2*N].reshape(N, 2)
            inner_angles = params_opt[2*N:]
            
            # Check for overlaps
            penalty = _overlap_penalty(inner_positions, inner_angles)
            if penalty < 1e-6:
                side = _required_outer_side(inner_positions, inner_angles, outer_center, outer_angle)
                if side < best_score:
                    best_score = side
                    inner_list = [[float(inner_positions[i][0]), float(inner_positions[i][1]), float(inner_angles[i])] for i in range(N)]
                    best_result = (inner_list, (0.0, 0.0), float(side), 0.0)
        except Exception:
            pass
    
    # If no valid result, return a safe fallback
    if best_result is None:
        # Fallback: linear arrangement with large outer hex
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
