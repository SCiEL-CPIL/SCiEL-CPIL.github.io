"""Packing 12 unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import time
import math
from scipy.optimize import least_squares

N = 12

# Deadline guard
_DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side length and rotation."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def _point_in_hexagon_dist(px, py, cx, cy, side, angle_deg):
    """
    Return the signed distance from point (px,py) to the boundary of a regular hexagon.
    Positive means outside, negative means inside.
    For a regular hexagon with 'flat top' orientation (angle=0), the hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    The hexagon can be described by 6 half-planes.
    """
    a = math.radians(angle_deg)
    # The 6 edge normals for a regular hexagon (pointy-top vs flat-top depends on convention)
    # For a hexagon with vertices at angles a + k*pi/3, the edges are perpendicular to directions a + (k+0.5)*pi/3
    # The distance from center to each edge (inradius) is side * sqrt(3)/2
    r_in = side * math.sqrt(3.0) / 2.0
    
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    # Rotate by -a to align hexagon
    ca = math.cos(-a)
    sa = math.sin(-a)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    
    # In local frame, hexagon vertices are at angles 0, 60, 120, 180, 240, 300
    # The 6 supporting lines (edges) have normals at angles 30, 90, 150, 210, 270, 330 degrees
    # i.e., normals at (k+0.5)*60 degrees for k=0..5
    # Distance from center to each edge line is r_in
    # For each edge, compute the signed distance; the point is inside if all are <= 0 (using inward normals)
    
    max_dist = 0.0
    for k in range(6):
        normal_angle = math.radians(30 + k * 60)
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        # Signed distance from point to the edge line (positive = outside)
        dist = lx * nx + ly * ny - r_in
        if dist > max_dist:
            max_dist = dist
    return max_dist

def _hex_hex_overlap_depth(v1, v2):
    """
    Compute the minimum distance needed to separate two hexagons.
    v1, v2 are arrays of 6 vertices each.
    Returns 0 if no overlap, positive if overlap (penetration depth).
    We use the separating axis theorem for convex polygons.
    """
    # SAT for two convex polygons
    # Get all edge normals from both polygons
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i + 1) % n]
            edge = p2 - p1
            # Normal perpendicular to edge
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                normals.append(np.array([nx, ny]) / norm)
        return normals
    
    normals = get_normals(v1) + get_normals(v2)
    
    min_overlap = float('inf')
    for n in normals:
        # Project both polygons onto this normal
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        # Check for separation
        if max1 <= min2 or max2 <= min1:
            return 0.0  # Separating axis found, no overlap
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    
    return min_overlap

def _residuals(params, outer_center, outer_side, outer_angle):
    """
    params: array of length 3*N = [x1, y1, a1, x2, y2, a2, ...]
    Returns residuals:
      - For each pair of inner hexagons: penetration depth (positive if overlapping)
      - For each inner vertex: distance outside outer hexagon (positive if outside)
    We want all residuals to be <= 0, so we use max(0, residual) as penalty.
    """
    residuals = []
    
    # Extract inner hexagon parameters
    centers = params[0::3]
    ys = params[1::3]
    angles = params[2::3]
    
    # Compute vertices for each inner hexagon
    inner_verts = []
    for i in range(N):
        verts = _hex_vertices(centers[i], ys[i], 1.0, angles[i])
        inner_verts.append(verts)
    
    # Pairwise inner-inner overlaps
    for i in range(N):
        for j in range(i + 1, N):
            depth = _hex_hex_overlap_depth(inner_verts[i], inner_verts[j])
            if depth > 0:
                residuals.append(depth)
    
    # Inner vertices vs outer hexagon
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for i in range(N):
        for v in inner_verts[i]:
            # Check if vertex is outside outer hexagon
            # Use point_in_hexagon_dist: positive means outside
            d = _point_in_hexagon_dist(v[0], v[1], outer_center[0], outer_center[1], outer_side, outer_angle)
            if d > 0:
                residuals.append(d)
    
    return np.array(residuals)

def _compute_outer_side(centers, ys, angles, outer_center, outer_angle):
    """
    Given inner hexagon positions, compute the minimum outer side length such that
    all inner hexagons fit inside the outer hexagon.
    This is the maximum over all inner vertices of the required distance to fit.
    """
    # For each inner vertex, compute the minimum outer side length needed so that
    # the vertex is inside the outer hexagon.
    # For a regular hexagon, a point is inside if its distance in each of the 6 normal directions
    # is <= r_in = side * sqrt(3)/2.
    # So side >= max over all normals and all vertices of (lx*nx + ly*ny) / (sqrt(3)/2)
    # where lx, ly are in the outer hexagon's local frame.
    
    a = math.radians(outer_angle)
    ca = math.cos(-a)
    sa = math.sin(-a)
    
    max_ratio = 0.0
    for i in range(N):
        verts = _hex_vertices(centers[i], ys[i], 1.0, angles[i])
        for v in verts:
            dx = v[0] - outer_center[0]
            dy = v[1] - outer_center[1]
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            for k in range(6):
                normal_angle = math.radians(30 + k * 60)
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                proj = lx * nx + ly * ny
                if proj > 0:
                    ratio = proj / (math.sqrt(3.0) / 2.0)
                    if ratio > max_ratio:
                        max_ratio = ratio
    
    return max_ratio

def construct():
    start_time = time.time()
    
    # Initial guess: arrange 12 hexagons in a hexagonal-ish pattern
    # 12 hexagons: center + ring of 6 + partial ring? Or 2 rows of 6?
    # Let's try a compact arrangement.
    
    # Strategy: start with a few different initial configurations and optimize each.
    
    best_side = float('inf')
    best_params = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try several initial configurations
    configs = []
    
    # Config 1: Two rows of 6, offset
    # Hexagonal lattice spacing for unit hexagons: horizontal = 2*sqrt(3), vertical = 3
    # Actually for unit side hexagons, the distance between centers in a hex grid:
    # Nearest neighbor distance = 2 (for pointy-top) or 2*sqrt(3)/sqrt(3)... 
    # Let's just use reasonable initial positions.
    
    # Config: 4x3 grid
    x0 = 0
    y0 = 0
    for row in range(3):
        for col in range(4):
            x = x0 + col * 3.5
            y = y0 + row * 3.0
            configs.append([x, y, 0.0])
    
    # Config: Hexagonal arrangement (1 + 6 + 5 partial)
    configs2 = []
    configs2.append([0, 0, 0.0])  # center
    for k in range(6):
        angle = k * 60
        configs2.append([2.0 * math.cos(math.radians(angle)), 2.0 * math.sin(math.radians(angle)), 0.0])
    for k in range(5):
        angle = 30 + k * 60
        configs2.append([4.0 * math.cos(math.radians(angle)), 4.0 * math.sin(math.radians(angle)), 0.0])
    configs.append(configs2)
    
    # Config: 3 rows: 4, 4, 4 with hex offset
    configs3 = []
    for row in range(3):
        for col in range(4):
            x = col * 3.46
            if row % 2 == 1:
                x += 1.73
            y = row * 3.0
            configs3.append([x, y, 0.0])
    configs.append(configs3)
    
    for config in configs:
        if time.time() > _DEADLINE:
            break
        if len(config) != N:
            continue
        
        # Flatten to params
        params0 = np.array(config).flatten()
        
        # Optimize outer angle and center as well? For simplicity, fix outer at origin, angle 0
        # and optimize inner positions. Then compute outer side.
        
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        
        # First, compute initial outer side
        centers0 = params0[0::3]
        ys0 = params0[1::3]
        angles0 = params0[2::3]
        side0 = _compute_outer_side(centers0, ys0, angles0, outer_center, outer_angle)
        
        # Now use least_squares to minimize a combined objective:
        # We want to minimize the outer side length, subject to no inner-inner overlaps.
        # Reformulate: minimize side, where side is determined by the vertices.
        # But side is a function of params. We can use scipy.optimize.minimize on side(params)
        # with penalty for overlaps.
        
        from scipy.optimize import minimize
        
        def objective(p):
            c = p[0::3]
            y = p[1::3]
            a = p[2::3]
            side = _compute_outer_side(c, y, a, outer_center, outer_angle)
            
            # Penalty for overlaps
            inner_verts = []
            for i in range(N):
                verts = _hex_vertices(c[i], y[i], 1.0, a[i])
                inner_verts.append(verts)
            
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    depth = _hex_hex_overlap_depth(inner_verts[i], inner_verts[j])
                    if depth > 0:
                        penalty += depth ** 2
            
            return side + 1000.0 * penalty
        
        # Try L-BFGS-B
        try:
            res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 2000})
            if time.time() > _DEADLINE:
                break
            
            # Evaluate result
            c = res.x[0::3]
            y = res.x[1::3]
            a = res.x[2::3]
            
            # Check for overlaps
            inner_verts = []
            for i in range(N):
                verts = _hex_vertices(c[i], y[i], 1.0, a[i])
                inner_verts.append(verts)
            
            has_overlap = False
            for i in range(N):
                for j in range(i + 1, N):
                    depth = _hex_hex_overlap_depth(inner_verts[i], inner_verts[j])
                    if depth > 0.001:
                        has_overlap = True
                        break
                if has_overlap:
                    break
            
            if not has_overlap:
                side = _compute_outer_side(c, y, a, outer_center, outer_angle)
                if side < best_side:
                    best_side = side
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # If no valid solution found, return a fallback
    if best_params is None:
        # Fallback: return the initial config with a large outer side
        inner = [[float(configs[0][i*3]), float(configs[0][i*3+1]), float(configs[0][i*3+2])] for i in range(N)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract best solution
    centers = best_params[0::3]
    ys = best_params[1::3]
    angles = best_params[2::3]
    
    inner = []
    for i in range(N):
        inner.append([float(centers[i]), float(ys[i]), float(angles[i])])
    
    return inner, (0.0, 0.0), float(best_side), 0.0
