import numpy as np
import time
from scipy.optimize import minimize
import itertools

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 110.0  # Leave safety margin for 120s limit

def _elapsed():
    return time.time() - _START_TIME

def _is_deadline():
    return _elapsed() > TIME_LIMIT

def _hex_vertices(cx, cy, side, angle_deg):
    """
    Return 6 vertices of a regular hexagon with center (cx, cy), side length 'side',
    rotated by angle_deg degrees.
    Vertices are in counterclockwise order starting from the rightmost point at 0 degrees.
    """
    angles = np.deg2rad(angle_deg) + np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def _hexagon_distance(h1_center, h1_angle, h2_center, h2_angle, side=1.0):
    """
    Compute the minimum distance between two unit regular hexagons.
    Returns a positive value if they are disjoint, negative if overlapping.
    We use the support function / separating axis approach for convex polygons.
    """
    v1 = _hex_vertices(h1_center[0], h1_center[1], side, h1_angle)
    v2 = _hex_vertices(h2_center[0], h2_center[1], side, h2_angle)
    
    # Get all edge normals from both hexagons
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1) % 6]
            edge = p2 - p1
            # Normal pointing outward (for CCW polygon, left normal is outward if we use cross product appropriately)
            # For a convex polygon in CCW order, the outward normal of edge (p1->p2) is (dy, -dx) normalized? 
            # Actually, let's just compute the distance between the two sets of points using a robust method.
    
    # Simpler: Use scipy's distance between convex hulls or just compute min distance between all point pairs? 
    # No, that's not correct for polygons. We need the minimum distance between two convex polygons.
    # We can use the separating axis theorem to check for overlap, and if no overlap, find min distance.
    
    # Let's use a simpler approximation for speed: compute the distance between centers and subtract the "radius" 
    # but that's not accurate. Let's implement a proper convex polygon distance.
    
    # For efficiency in optimization, we might use a different metric.
    # Let's compute the minimum distance between the two sets of edges.
    
    def _point_to_segment_distance(p, a, b):
        ab = b - a
        ap = p - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq == 0:
            return np.linalg.norm(p - a)
        t = np.dot(ap, ab) / ab_len_sq
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)
    
    min_dist = np.inf
    for i in range(6):
        p1a = v1[i]
        p1b = v1[(i+1)%6]
        for j in range(6):
            p2a = v2[j]
            p2b = v2[(j+1)%6]
            # Distance from p2a to segment p1a-p1b
            d = _point_to_segment_distance(p2a, p1a, p1b)
            min_dist = min(min_dist, d)
            # Distance from p2b to segment p1a-p1b
            d = _point_to_segment_distance(p2b, p1a, p1b)
            min_dist = min(min_dist, d)
            # Distance from p1a to segment p2a-p2b
            d = _point_to_segment_distance(p1a, p2a, p2b)
            min_dist = min(min_dist, d)
            # Distance from p1b to segment p2a-p2b
            d = _point_to_segment_distance(p1b, p2a, p2b)
            min_dist = min(min_dist, d)
            # Also check edge-edge intersection? For now, point-to-segment should catch most.
    
    # If they overlap, min_dist will be 0 or we need to detect overlap.
    # The above computes min distance between boundaries. If polygons overlap, this is 0.
    # We want a "gap" that is positive when disjoint.
    # But if they overlap, we want a negative value to penalize.
    # Let's check if they are disjoint by checking if min_dist > 0.
    # However, if one hexagon is inside another, min_dist could be > 0 but they overlap.
    # Since all hexagons are same size, one cannot be inside another without intersection.
    # So min_dist == 0 implies touching or overlapping.
    # To get a signed distance, we can use the separating axis theorem.
    
    # For now, let's assume min_dist is the gap. If they overlap, we need to detect it.
    # Let's add a check: if any vertex of v1 is inside v2 or vice versa, they overlap.
    
    def _point_in_hexagon(p, v):
        # Check if point p is inside convex polygon v (CCW)
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1)%6]
            cross = np.cross(p2 - p1, p - p1)
            if cross < -1e-9:  # Assuming CCW, cross should be >= 0
                return False
        return True
    
    overlap = False
    for p in v1:
        if _point_in_hexagon(p, v2):
            overlap = True
            break
    if not overlap:
        for p in v2:
            if _point_in_hexagon(p, v1):
                overlap = True
                break
    
    if overlap:
        # Estimate penetration depth roughly as negative of min_dist if min_dist is 0
        # Or just return a small negative number
        return -min_dist - 0.01
    else:
        return min_dist

def _construct_initial(N):
    """
    Create an initial configuration of N unit hexagons in a hexagonal lattice.
    """
    inner = []
    # Hexagonal lattice spacing: distance between centers of adjacent hexagons is sqrt(3) for side 1
    # if they share an edge. Let's verify: 
    # A regular hexagon with side s has distance between opposite sides = sqrt(3)*s.
    # So if we pack them edge-to-edge, center-to-center distance is sqrt(3).
    dx = np.sqrt(3)
    dy = 1.5
    # Generate hexagonal lattice points
    points = []
    # Center at (0,0)
    # Rows
    max_r = int(np.ceil(N ** 0.5)) + 2
    for row in range(-max_r, max_r + 1):
        for col in range(-max_r, max_r + 1):
            x = dx * (col + 0.5 * (row & 1))
            y = dy * row
            points.append((x, y))
    
    # Sort by distance from origin to pick the closest N
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    points = points[:N]
    
    inner = [[x, y, 0.0] for x, y in points]
    return inner

def _objective(params, N):
    """
    Objective function for optimization.
    params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, cx, cy, side]
    We want to minimize 'side' while ensuring:
    1. Hexagons don't overlap (distance > 0)
    2. Hexagons are inside the outer hexagon
    
    For simplicity, we fix the outer hexagon at (0,0) with angle 0.
    So params: [x1, y1, a1, ..., xN, yN, aN, side]
    """
    if _is_deadline():
        return 1e10
    
    pos = params[:3*N]
    side = params[3*N]
    
    if side < 0.5:
        return 1e10
    
    inner = [[pos[i], pos[i+1], pos[i+2]] for i in range(0, 3*N, 3)]
    
    # Check overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _hexagon_distance((inner[i][0], inner[i][1]), inner[i][2], 
                                  (inner[j][0], inner[j][1]), inner[j][2])
            if d < 0.01:
                penalty += (0.01 - d) ** 2 * 100
    
    # Check containment
    # Outer hexagon: center (0,0), side 'side', angle 0
    outer_v = _hex_vertices(0, 0, side, 0)
    for i in range(N):
        v_inner = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for p in v_inner:
            if not _point_in_hexagon(p, outer_v):
                # Calculate distance to boundary
                # Approximate by finding closest point on outer boundary
                min_d = np.inf
                for k in range(6):
                    p1 = outer_v[k]
                    p2 = outer_v[(k+1)%6]
                    d = _point_to_segment_distance(p, p1, p2)
                    min_d = min(min_d, d)
                penalty += min_d ** 2 * 100
    
    return side + penalty

def _point_to_segment_distance(p, a, b):
    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq == 0:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / ab_len_sq
    t = np.clip(t, 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def _point_in_hexagon(p, v):
    for i in range(6):
        p1 = v[i]
        p2 = v[(i+1)%6]
        cross = np.cross(p2 - p1, p - p1)
        if cross < -1e-9:
            return False
    return True

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    N = 11
    # Initial configuration
    inner = _construct_initial(N)
    
    # Initial side length: estimate based on bounding box
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    max_x = max(xs)
    min_x = min(xs)
    max_y = max(ys)
    min_y = min(ys)
    # Hexagon side length to contain this bounding box
    # For a hexagon with side S, the width is 2*S and height is sqrt(3)*S
    # So S >= max(max_x - min_x, (max_y - min_y) / np.sqrt(3)) / 2 + 1
    initial_side = max((max_x - min_x) / 2, (max_y - min_y) / (2 * np.sqrt(3))) + 1.5
    
    # Parameters: x1, y1, a1, ..., xN, yN, aN, side
    params = []
    for p in inner:
        params.extend([p[0], p[1], p[2]])
    params.append(initial_side)
    
    # Optimize
    best_params = params
    best_side = initial_side
    
    try:
        result = minimize(_objective, params, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        if result.success or result.fun < _objective(params, N):
            best_params = result.x
            best_side = best_params[3*N]
    except Exception as e:
        pass
    
    if _is_deadline():
        # Return current best
        inner_out = [[best_params[i], best_params[i+1], best_params[i+2]] for i in range(0, 3*N, 3)]
        return inner_out, (0.0, 0.0), best_side, 0.0
    
    # Try a few more iterations with different initializations
    for restart in range(5):
        if _is_deadline():
            break
        
        # Perturb initial configuration
        inner_pert = []
        for p in inner:
            x = p[0] + np.random.uniform(-0.5, 0.5)
            y = p[1] + np.random.uniform(-0.5, 0.5)
            a = p[2] + np.random.uniform(-30, 30)
            inner_pert.append([x, y, a])
        
        params_pert = []
        for p in inner_pert:
            params_pert.extend([p[0], p[1], p[2]])
        params_pert.append(initial_side * 1.1)
        
        try:
            result = minimize(_objective, params_pert, method='Nelder-Mead', 
                             options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < _objective(best_params, N):
                best_params = result.x
                best_side = best_params[3*N]
        except Exception as e:
            pass
    
    inner_out = [[best_params[i], best_params[i+1], best_params[i+2]] for i in range(0, 3*N, 3)]
    return inner_out, (0.0, 0.0), best_side, 0.0
