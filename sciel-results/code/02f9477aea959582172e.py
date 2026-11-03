import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(vertices, outer_angle_deg):
    """
    Find minimum side length of regular hexagon (given rotation) containing all vertices.
    Returns (cx, cy, side) or None.
    The hexagon has 6 sides; each side has a normal direction.
    For a hexagon with center (cx,cy), side s, and rotation theta:
      The support function in direction n_k is s * cos(30 deg) = s * sqrt(3)/2
      where n_k are the 6 outward normals at angles theta+30+60k.
    Constraint: for each vertex v and each normal n_k:
      (v - c) · n_k <= s * sqrt(3)/2
    """
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    A = []
    b = []
    
    for vx, vy in vertices:
        for nx, ny in normals:
            # (vx - cx)*nx + (vy - cy)*ny <= s*sqrt(3)/2
            # -cx*nx - cy*ny - s*sqrt(3)/2 <= -vx*nx - vy*ny
            A.append([-nx, -ny, -math.sqrt(3)/2])
            b.append(-vx*nx - vy*ny)
            # -(vx - cx)*nx - (vy - cy)*ny <= s*sqrt(3)/2  (opposite direction)
            A.append([nx, ny, -math.sqrt(3)/2])
            b.append(vx*nx + vy*ny)
    
    # s >= 0
    c = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            cx, cy, s = res.x
            return cx, cy, s
        else:
            return None
    except:
        return None

def _check_disjoint(inner_positions, inner_angles, gap=1.0001):
    """
    Check that all pairs of unit hexagons are disjoint (with small gap).
    Two convex hexagons are disjoint iff no edge of one intersects an edge of the other
    and neither contains the other's center.
    For regular hexagons of same size, checking center-to-center distance >= 2*apothem
    is sufficient when they have the same orientation. For different orientations,
    we use the separating axis theorem.
    """
    def hex_vertices_local(angle_deg):
        a = math.radians(angle_deg)
        return [(math.cos(a + math.pi*k/3), math.sin(a + math.pi*k/3)) for k in range(6)]
    
    def segments_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)
    
    def point_in_hex(px, py, cx, cy, angle_deg):
        a = math.radians(angle_deg)
        # Check if point is inside hexagon centered at (cx,cy)
        dx, dy = px - cx, py - cy
        # Rotate point to align hexagon
        cos_a, sin_a = math.cos(-a), math.sin(-a)
        rx = dx*cos_a - dy*sin_a
        ry = dx*sin_a + dy*cos_a
        # In local coords, hexagon has vertices at angles 0,60,...,300
        # |rx| <= 1 and |ry| <= sqrt(3)/2 and |rx|/1 + |ry|/(sqrt(3)/2) <= 2/sqrt(3) ... 
        # Actually simpler: check against 6 edges
        for k in range(6):
            a1 = math.pi * k / 3
            a2 = math.pi * (k+1) / 3
            # Edge from vertex k to vertex k+1
            vx1, vy1 = math.cos(a1), math.sin(a1)
            vx2, vy2 = math.cos(a2), math.sin(a2)
            # Normal pointing inward
            ex, ey = vx2 - vx1, vy2 - vy1
            nx, ny = -ey, ex  # rotate edge 90 deg
            # Check which side
            dot = (rx - vx1)*nx + (ry - vy1)*ny
            # The center (0,0) should be on the same side
            center_dot = (0 - vx1)*nx + (0 - vy1)*ny
            if dot * center_dot < 0:
                return False
        return True
    
    n = len(inner_positions)
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1 = inner_positions[i][0], inner_positions[i][1]
            cx2, cy2 = inner_positions[j][0], inner_positions[j][1]
            a1 = inner_angles[i]
            a2 = inner_angles[j]
            
            # Quick check: center distance
            dist = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if dist > 2.0 * gap:
                continue  # definitely disjoint
            
            # Check if center of one is in the other
            if point_in_hex(cx1, cy1, cx2, cy2, a2):
                return False
            if point_in_hex(cx2, cy2, cx1, cy1, a1):
                return False
            
            # Check edge intersections
            verts1 = _get_hex_vertices(cx1, cy1, a1)
            verts2 = _get_hex_vertices(cx2, cy2, a2)
            for k in range(6):
                p1 = verts1[k]
                p2 = verts1[(k+1) % 6]
                for m in range(6):
                    p3 = verts2[m]
                    p4 = verts2[(m+1) % 6]
                    if segments_intersect(p1, p2, p3, p4):
                        return False
    return True

def _generate_hex_layout(n_hexes):
    """
    Generate hexagonal arrangement of n unit hexagons on a hex lattice.
    The center hexagon is at origin, surrounded by rings.
    """
    sqrt3 = math.sqrt(3)
    # Hex lattice: neighbors at distance sqrt(3) in 6 directions
    # For flat-top hexagons (angle 0), the lattice vectors are:
    # (sqrt(3), 0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), etc.
    
    # Generate all lattice points in a hexagonal region
    # Using axial coordinates (q, r)
    points = []
    max_ring = 4
    for q in range(-max_ring, max_ring+1):
        for r in range(-max_ring, max_ring+1):
            if abs(q + r) <= max_ring:
                # Convert axial to cartesian for flat-top hexagons
                x = sqrt3 * (q + r/2)
                y = 1.5 * r
                dist_sq = x*x + y*y
                points.append((dist_sq, x, y))
    
    points.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y in points:
        positions.append((x, y))
        if len(positions) == n_hexes:
            break
    
    return positions

def _compute_outer_side(positions, inner_angles, outer_angle_deg):
    """Compute the minimum enclosing hexagon side for given inner layout."""
    all_vertices = []
    for (px, py), ang in zip(positions, inner_angles):
        verts = _get_hex_vertices(px, py, ang)
        all_vertices.extend(verts)
    
    res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    if res is None:
        return None
    return res  # (cx, cy, side)

def construct():
    global _START_TIME, _DEADLINE
    
    # Fallback if too little time
    if time.time() > _DEADLINE - 3.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate hexagonal layout
    positions = _generate_hex_layout(N)
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_positions = None
    best_angles = None
    
    # Try different inner angle configurations
    # For hexagonal packing, all hexagons typically have the same orientation
    # Try 0 and 30 degrees
    for inner_angle in [0.0, 30.0]:
        if time.time() > _DEADLINE - 10.0:
            break
        
        inner_angles = [inner_angle] * N
        
        # Verify disjointness
        if not _check_disjoint(positions, inner_angles):
            continue
        
        # Coarse search for outer angle
        best_coarse_side = float('inf')
        best_coarse_angle = 0.0
        best_coarse_center = (0.0, 0.0)
        
        for outer_angle_deg in np.arange(0, 30.1, 0.5):
            if time.time() > _DEADLINE - 10.0:
                break
            res = _compute_outer_side(positions, inner_angles, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_coarse_side:
                    best_coarse_side = side
                    best_coarse_angle = outer_angle_deg
                    best_coarse_center = (cx, cy)
        
        # Fine search
        if best_coarse_side < float('inf'):
            for outer_angle_deg in np.arange(max(0, best_coarse_angle - 0.5), 
                                            min(30, best_coarse_angle + 0.5) + 0.01, 0.01):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _compute_outer_side(positions, inner_angles, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_angle = outer_angle_deg
                        best_coarse_center = (cx, cy)
        
        # Polish with L-BFGS-B
        if time.time() < _DEADLINE - 15.0 and best_coarse_side < float('inf'):
            all_vertices = []
            for (px, py), ang in zip(positions, inner_angles):
                verts = _get_hex_vertices(px, py, ang)
                all_vertices.extend(verts)
            
            def objective(params):
                angle_deg = params[0]
                res = _min_enclosing_hex_side_lp(all_vertices, angle_deg)
                if res is None:
                    return 1e10
                return res[2]
            
            try:
                result = minimize(objective, [best_coarse_angle], method='L-BFGS-B',
                                 options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14})
                if result.success:
                    polished_angle = result.x[0]
                    res = _min_enclosing_hex_side_lp(all_vertices, polished_angle)
                    if res:
                        cx, cy, side = res
                        if side < best_coarse_side:
                            best_coarse_side = side
                            best_coarse_angle = polished_angle
                            best_coarse_center = (cx, cy)
            except:
                pass
        
        if best_coarse_side < best_side:
            best_side = best_coarse_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_center = best_coarse_center
            best_outer_angle = best_coarse_angle
            best_positions = positions
            best_angles = inner_angles
    
    # Try to improve by optimizing positions with Nelder-Mead
    if time.time() < _DEADLINE - 30.0 and best_positions is not None:
        all_vertices = []
        for (px, py), ang in zip(best_positions, best_angles):
            verts = _get_hex_vertices(px, py, ang)
            all_vertices.extend(verts)
        
        # Use the best outer angle found
        outer_angle = best_outer_angle
        
        def layout_objective(params):
            # params: flattened list of (x, y) for each hexagon
            # First check disjointness (with gap)
            test_positions = [(params[2*i], params[2*i+1]) for i in range(N)]
            if not _check_disjoint(test_positions, best_angles, gap=1.0001):
                return 1e6
            
            test_vertices = []
            for (px, py), ang in zip(test_positions, best_angles):
                verts = _get_hex_vertices(px, py, ang)
                test_vertices.extend(verts)
            
            res = _min_enclosing_hex_side_lp(test_vertices, outer_angle)
            if res is None:
                return 1e6
            return res[2]
        
        x0 = np.array([pos[0] for pos in best_positions] + [pos[1] for pos in best_positions])
        
        try:
            result = minimize(layout_objective, x0, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_side:
                new_positions = [(result.x[2*i], result.x[2*i+1]) for i in range(N)]
                if _check_disjoint(new_positions, best_angles, gap=1.0001):
                    # Recompute outer hex
                    new_vertices = []
                    for (px, py), ang in zip(new_positions, best_angles):
                        verts = _get_hex_vertices(px, py, ang)
                        new_vertices.extend(verts)
                    res = _min_enclosing_hex_side_lp(new_vertices, outer_angle)
                    if res:
                        cx, cy, side = res
                        if side < best_side:
                            best_side = side
                            best_inner = [[pos[0], pos[1], best_angles[i]] for i, pos in enumerate(new_positions)]
                            best_center = (cx, cy)
        except:
            pass
    
    # Final polish: try to optimize outer angle with tighter tolerance
    if time.time() < _DEADLINE - 5.0 and best_inner is not None:
        all_vertices = []
        for entry in best_inner:
            px, py, ang = entry[0], entry[1], entry[2]
            verts = _get_hex_vertices(px, py, ang)
            all_vertices.extend(verts)
        
        for outer_angle_deg in np.arange(max(0, best_outer_angle - 0.1), 
                                        min(30, best_outer_angle + 0.1) + 0.001, 0.001):
            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_outer_angle = outer_angle_deg
    
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
