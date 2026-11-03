"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hex_support(cx, cy, s, angle_deg, dx, dy):
    """Support function of hexagon in direction (dx,dy)."""
    a = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
    vx = s * np.cos(angles)
    vy = s * np.sin(angles)
    proj = vx * dx + vy * dy
    return cx * dx + cy * dy + np.max(proj)

def _min_separation(p1, p2):
    """
    Minimum separation between two unit hexagons using SAT.
    Returns positive if separated, negative if overlapping (magnitude = penetration).
    """
    x1, y1, a1 = p1
    x2, y2, a2 = p2
    
    # Edge normals for hex1 and hex2
    def get_normals(angle_deg):
        a = np.radians(angle_deg)
        # For a regular hexagon, edge normals are at angles 30, 90, 150, 210, 270, 330 + rotation
        # Actually, the outward normals of edges are perpendicular to edges.
        # Vertices at 0,60,...300 deg. Edge from v[i] to v[i+1] has direction 60*i+30 deg
        # Normal is perpendicular: 60*i+30+90 = 60*i+120? No.
        # Let me compute properly.
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
        vx = np.cos(angles)
        vy = np.sin(angles)
        normals = []
        for i in range(6):
            ex = vx[(i+1)%6] - vx[i]
            ey = vy[(i+1)%6] - vy[i]
            length = np.sqrt(ex*ex + ey*ey)
            if length > 1e-12:
                # Outward normal (for CCW polygon, right normal is outward)
                nx = ey / length
                ny = -ex / length
                normals.append((nx, ny))
        return normals
    
    n1 = get_normals(a1)
    n2 = get_normals(a2)
    
    all_normals = n1 + n2
    
    max_penetration = 0.0
    for nx, ny in all_normals:
        # Project both hexagons onto this axis
        # Hex1: center (x1,y1), support in dir (nx,ny) and (-nx,-ny)
        s1_pos = _hex_support(x1, y1, 1.0, a1, nx, ny)
        s1_neg = _hex_support(x1, y1, 1.0, a1, -nx, -ny)
        min1 = -s1_neg
        max1 = s1_pos
        
        s2_pos = _hex_support(x2, y2, 1.0, a2, nx, ny)
        s2_neg = _hex_support(x2, y2, 1.0, a2, -nx, -ny)
        min2 = -s2_neg
        max2 = s2_pos
        
        # Check separation
        if max1 < min2 or max2 < min1:
            return 1.0  # Separated, no overlap
        
        # Overlap on this axis: penetration = min(max1, max2) - max(min1, min2)
        pen = min(max1, max2) - max(min1, min2)
        if pen > max_penetration:
            max_penetration = pen
    
    return -max_penetration

def _point_in_hex(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s, rotated angle_deg."""
    # Transform point to hexagon's local frame
    a = np.radians(angle_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    dx = px - cx
    dy = py - cy
    # Rotate by -a
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side s, the constraints in local frame:
    # |lx| <= s * sqrt(3)/2  (for the two vertical-ish faces)
    # Actually, let me use the support function approach.
    # The hexagon contains the point iff for all 6 outward normals, (p-center).n <= s*sqrt(3)/2
    # Wait, that's not quite right. Let me use vertex-based point-in-polygon.
    
    verts = _hex_vertices(0, 0, s, 0)  # local frame
    # Point in convex polygon: check all edges
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i+1)%6]
        # Cross product to check if point is on the correct side
        cross = (x2-x1)*(ly-y1) - (y2-y1)*(lx-x1)
        if cross < -1e-9:
            return False
    return True

def _hex_inside_hex(inner_pts, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons are inside the outer hexagon."""
    cx, cy = outer_center
    for ix, iy, ia in inner_pts:
        iv = _hex_vertices(ix, iy, 1.0, ia)
        for vx, vy in iv:
            if not _point_in_hex(vx, vy, cx, cy, outer_side, outer_angle):
                return False
    return True

def _compute_min_side(inner_pts, outer_center, outer_angle):
    """Compute minimum outer side length for given inner config and outer center/angle."""
    cx, cy = outer_center
    theta = np.radians(outer_angle)
    # Outward normals of the 6 faces
    # For a regular hexagon with vertices at angles 0,60,...,300 + theta,
    # the face normals (outward) are at angles 30, 90, 150, 210, 270, 330 + theta
    face_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + theta
    normals = np.column_stack([np.cos(face_angles), np.sin(face_angles)])
    
    min_s = 0.0
    for ix, iy, ia in inner_pts:
        iv = _hex_vertices(ix, iy, 1.0, ia)
        for vx, vy in iv:
            for k in range(6):
                dot = (vx - cx) * normals[k, 0] + (vy - cy) * normals[k, 1]
                if dot > 0:
                    s_needed = dot / (np.sqrt(3) / 2)
                    min_s = max(min_s, s_needed)
    
    return min_s

def _objective(x, deadline):
    """Objective function: outer side length, with penalty for overlaps."""
    if time.time() > deadline:
        return 1e10
    
    inner_pts = []
    for i in range(N):
        ix = x[3*i]
        iy = x[3*i+1]
        ia = np.degrees(x[3*i+2])
        inner_pts.append((ix, iy, ia))
    outer_angle = np.degrees(x[3*N])
    
    # Check non-overlap with fast SAT
    for i in range(N):
        for j in range(i+1, N):
            sep = _min_separation(inner_pts[i], inner_pts[j])
            if sep < -1e-6:
                # Overlapping, return penalty proportional to penetration
                return 1e6 + abs(sep)
    
    s = _compute_min_side(inner_pts, (0.0, 0.0), outer_angle)
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = 40.0
    best_angle = 0.0
    
    def _update_best(inner_pts, side, angle):
        nonlocal best_score, best_inner, best_center, best_side, best_angle
        if side < best_score:
            best_score = side
            best_inner = inner_pts
            best_center = (0.0, 0.0)
            best_side = side
            best_angle = angle
    
    # Initial configurations: various hexagonal lattice arrangements
    def make_hex_lattice(rows, cols, rotation_deg=0.0):
        spacing = 2 * np.sqrt(3)
        pts = []
        for r in range(rows):
            for c in range(cols):
                x = c * spacing
                y = r * spacing * 1.5
                if r % 2 == 1:
                    x += spacing / 2
                pts.append([x, y, rotation_deg])
        return pts
    
    configs = []
    # 3x4, 4x3, 2x6, 6x2, 1x12, 12x1
    configs.append(make_hex_lattice(3, 4))
    configs.append(make_hex_lattice(4, 3))
    configs.append(make_hex_lattice(2, 6))
    configs.append(make_hex_lattice(6, 2))
    configs.append(make_hex_lattice(1, 12))
    configs.append(make_hex_lattice(12, 1))
    
    # Also try a more compact "flower" arrangement: 1 center + 6 around + 5 more
    def make_flower():
        pts = [[0, 0, 0]]  # center
        r = 2 * np.sqrt(3)
        for i in range(6):
            angle = i * np.pi / 3
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            pts.append([x, y, 0])
        # Add 5 more in a second ring
        r2 = 2 * r
        for i in range(5):
            angle = i * np.pi / 3 + np.pi / 6
            x = r2 * np.cos(angle)
            y = r2 * np.sin(angle)
            pts.append([x, y, 0])
        return pts
    configs.append(make_flower())
    
    for config in configs:
        if time.time() > deadline:
            break
        
        # Center the config
        xs = [p[0] for p in config]
        ys = [p[1] for p in config]
        cx0 = (min(xs) + max(xs)) / 2
        cy0 = (min(ys) + max(ys)) / 2
        config_centered = [(p[0] - cx0, p[1] - cy0, p[2]) for p in config]
        
        # Initial outer side estimate
        s0 = _compute_min_side(config_centered, (0.0, 0.0), 0.0)
        _update_best(config_centered, s0, 0.0)
        
        # Prepare initial vector for optimization
        x0 = []
        for p in config_centered:
            x0.extend([p[0], p[1], np.radians(p[2])])
        x0.append(0.0)  # outer angle
        
        # Local optimization with Nelder-Mead
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                            args=(deadline,),
                            options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-7,
                                   'adaptive': True})
            
            if result.fun < 1e5:  # Valid solution
                inner_pts = []
                for i in range(N):
                    ix = result.x[3*i]
                    iy = result.x[3*i+1]
                    ia = np.degrees(result.x[3*i+2])
                    inner_pts.append((ix, iy, ia))
                outer_angle = np.degrees(result.x[3*N])
                s_final = _compute_min_side(inner_pts, (0.0, 0.0), outer_angle)
                
                # Verify no overlaps
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if _min_separation(inner_pts[i], inner_pts[j]) < -1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    _update_best(inner_pts, s_final, outer_angle)
        except Exception:
            pass
    
    # If time remains, try differential evolution for global search
    if time.time() < deadline - 30:
        try:
            # Bounds for each variable
            bounds = [(-10, 10), (-10, 10), (-np.pi, np.pi)] * N + [(-np.pi, np.pi)]
            
            def de_objective(x):
                return _objective(x, deadline)
            
            de_result = differential_evolution(de_objective, bounds,
                                              maxiter=50, popsize=15,
                                              tol=1e-5, seed=42,
                                              polish=True,
                                              updating='immediate',
                                              workers=1)
            
            if de_result.fun < 1e5:
                inner_pts = []
                for i in range(N):
                    ix = de_result.x[3*i]
                    iy = de_result.x[3*i+1]
                    ia = np.degrees(de_result.x[3*i+2])
                    inner_pts.append((ix, iy, ia))
                outer_angle = np.degrees(de_result.x[3*N])
                s_final = _compute_min_side(inner_pts, (0.0, 0.0), outer_angle)
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if _min_separation(inner_pts[i], inner_pts[j]) < -1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    _update_best(inner_pts, s_final, outer_angle)
        except Exception:
            pass
    
    # Return best result
    if best_inner is not None:
        return best_inner, best_center, best_side, best_angle
    
    # Fallback
    inner = [(4.0 * k, 0.0, 0.0) for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
