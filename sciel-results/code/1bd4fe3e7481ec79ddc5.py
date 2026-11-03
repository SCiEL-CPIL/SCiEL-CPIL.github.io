import math
import time
import numpy as np

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _hex_vertices(cx, cy, s=1.0, angle=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side s, rotated by angle (radians)."""
    # Pointy-top hexagon vertices at angle 0: angles 30, 90, 150, 210, 270, 330 degrees
    # Actually for flat-top: 0, 60, 120, 180, 240, 300
    # Use flat-top: vertex at angle k*60 degrees, radius = s
    verts = []
    for k in range(6):
        a = angle + k * math.pi / 3.0
        verts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return np.array(verts)

def _hexagon_overlap(v1, v2, eps=1e-9):
    """Check if two convex hexagons (arrays of 6 points) overlap using SAT."""
    for verts in (v1, v2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            # Edge normal
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Normal (perpendicular)
            nx = -dy
            ny = dx
            # Project both hexagons onto this normal
            proj1 = v1 @ np.array([nx, ny])
            proj2 = v2 @ np.array([nx, ny])
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 + eps or max2 < min1 + eps:
                return False
    return True

def _separation_distance(v1, v2):
    """Compute minimum separation between two hexagons using SAT (0 if overlapping)."""
    min_sep = float('inf')
    for verts in (v1, v2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            nx = -dy
            ny = dx
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-12:
                continue
            nx /= norm
            ny /= norm
            proj1 = v1 @ np.array([nx, ny])
            proj2 = v2 @ np.array([nx, ny])
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            sep = max(min2 - max1, min1 - max2)
            if sep < 0:
                return 0.0
            min_sep = min(min_sep, sep)
    return min_sep if min_sep < float('inf') else 0.0

def _minimal_enclosing_hexagon(points, beta):
    """
    Given a set of points and a rotation angle beta for the outer hexagon,
    compute the minimal side length S of a regular hexagon (rotated by beta)
    that encloses all points, and the center of that hexagon.
    
    The outer hexagon has 6 supporting lines. For rotation beta, the normals
    to the sides are at angles beta + pi/6 + k*pi/3 for k=0..5.
    
    We need to find center (cx, cy) and apothem D such that for all points p:
        |(p - c) . n_k| <= D for all k
    
    This is a linear program: minimize D subject to constraints.
    """
    from scipy.optimize import linprog
    
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    
    for nx, ny in normals:
        for px, py in points:
            # (px-cx)*nx + (py-cy)*ny <= D  =>  -nx*cx - ny*cy - D <= -px*nx - py*ny
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            # -(px-cx)*nx - (py-cy)*ny <= D  =>  nx*cx + ny*cy - D <= px*nx + py*ny
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.status == 0:
        D = res.x[2]
        cx = res.x[0]
        cy = res.x[1]
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D
    else:
        return float('inf'), None, None

def _compute_min_hexagon(points):
    """Find minimal enclosing regular hexagon over all rotations."""
    best_S = float('inf')
    best_beta = 0.0
    best_center = None
    
    # Coarse search
    for i in range(37):
        beta = i * (math.pi / 6) / 36
        S, center, D = _minimal_enclosing_hexagon(points, beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_center = center
    
    # Refine
    from scipy.optimize import minimize_scalar
    try:
        def obj(b):
            S, _, _ = _minimal_enclosing_hexagon(points, b)
            return S
        res = minimize_scalar(obj, bounds=(0, math.pi/6), method='bounded', options={'maxiter': 200, 'xatol': 1e-12})
        if res.fun < best_S:
            best_S = res.fun
            best_beta = res.x
            _, best_center, _ = _minimal_enclosing_hexagon(points, best_beta)
    except Exception:
        pass
    
    return best_S, best_beta, best_center

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Force-directed relaxation to pack 11 hexagons
    # Each hexagon: position (x, y) and angle
    # We'll use a repulsive potential that pushes overlapping hexagons apart
    
    def relax(positions, angles, iterations=2000, lr=0.01):
        """Force-directed relaxation to minimize overlaps."""
        pos = np.array(positions, dtype=float)
        ang = np.array(angles, dtype=float)
        
        for it in range(iterations):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            forces = np.zeros_like(pos)
            
            for i in range(N):
                vi = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                for j in range(i+1, N):
                    vj = _hex_vertices(pos[j, 0], pos[j, 1], 1.0, ang[j])
                    sep = _separation_distance(vi, vj)
                    if sep < 0.1:  # Only interact when close
                        # Compute repulsive force direction
                        # Use the direction from j to i, weighted by overlap
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-9:
                            # Random direction
                            a = np.random.uniform(0, 2*math.pi)
                            dx, dy = math.cos(a), math.sin(a)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        
                        # Force magnitude: stronger when more overlapping
                        overlap = max(0, 0.5 - sep)
                        f = overlap * 10.0
                        forces[i, 0] += f * dx
                        forces[i, 1] += f * dy
                        forces[j, 0] -= f * dx
                        forces[j, 1] -= f * dy
            
            # Update positions
            pos += lr * forces
            # Damping
            lr *= 0.999
        
        return pos, ang

    def try_configuration(initial_positions, initial_angles, iterations=3000):
        """Try a configuration and return the best result."""
        pos, ang = list(initial_positions), list(initial_angles)
        
        # Multiple rounds of relaxation with decreasing learning rate
        for round_num in range(5):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            lr = 0.05 / (round_num + 1)
            pos, ang = relax(pos, ang, iterations=iterations//(round_num+1), lr=lr)
        
        # Check for remaining overlaps
        points = []
        for i in range(N):
            vi = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
            points.append(vi)
        
        # Verify no overlaps
        has_overlap = False
        for i in range(N):
            for j in range(i+1, N):
                if _hexagon_overlap(points[i], points[j]):
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if has_overlap:
            return None
        
        # Compute minimal enclosing hexagon
        # Use centers as representative points, but we need to account for the hexagon extent
        # Actually, we need to enclose all vertices of all hexagons
        all_points = []
        for vi in points:
            all_points.extend(vi.tolist())
        all_points = np.array(all_points)
        
        S, beta, center = _compute_min_hexagon(all_points)
        
        return S, pos, ang, beta, center

    # Try multiple initial configurations
    best_result = None
    best_S = float('inf')
    
    # Configuration 1: Hexagonal grid pattern
    configs = []
    
    # Hex grid: center + 6 around + 4 more
    # Axial coordinates for a compact arrangement
    hex_grid = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    
    # Convert axial to cartesian
    def axial_to_cartesian(axial):
        q, r = axial
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    # Generate multiple starting configurations
    start_configs = []
    
    # Config 1: Standard hex grid
    pos1 = [axial_to_cartesian(a) for a in hex_grid]
    ang1 = [0.0] * N
    start_configs.append((pos1, ang1))
    
    # Config 2: Slightly perturbed
    np.random.seed(42)
    pos2 = [(x + np.random.uniform(-0.1, 0.1), y + np.random.uniform(-0.1, 0.1)) for x, y in pos1]
    ang2 = [np.random.uniform(-0.1, 0.1) for _ in range(N)]
    start_configs.append((pos2, ang2))
    
    # Config 3: Different arrangement - more linear
    hex_linear = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2)
    ]
    pos3 = [axial_to_cartesian(a) for a in hex_linear]
    ang3 = [0.0] * N
    start_configs.append((pos3, ang3))
    
    # Config 4: Diamond shape
    hex_diamond = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
        (2, 0), (0, 2)
    ]
    pos4 = [axial_to_cartesian(a) for a in hex_diamond]
    ang4 = [0.0] * N
    start_configs.append((pos4, ang4))
    
    # Config 5: Star pattern
    hex_star = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (1, -2), (-1, -2)
    ]
    pos5 = [axial_to_cartesian(a) for a in hex_star]
    ang5 = [0.0] * N
    start_configs.append((pos5, ang5))
    
    # Config 6: With some rotations
    pos6 = [axial_to_cartesian(a) for a in hex_grid]
    ang6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.pi/6, math.pi/6, math.pi/6, math.pi/6]
    start_configs.append((pos6, ang6))
    
    # Config 7: Clustered
    hex_cluster = [
        (0, 0), (1, 0), (0, 1), (-1, 1),
        (1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-1, 2)
    ]
    pos7 = [axial_to_cartesian(a) for a in hex_cluster]
    ang7 = [0.0] * N
    start_configs.append((pos7, ang7))
    
    # Config 8: Elongated
    hex_elong = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (0, -1)
    ]
    pos8 = [axial_to_cartesian(a) for a in hex_elong]
    ang8 = [0.0] * N
    start_configs.append((pos8, ang8))
    
    # Config 9: With rotations mixed
    pos9 = [axial_to_cartesian(a) for a in hex_grid]
    ang9 = [0.0]*7 + [math.pi/12, -math.pi/12, math.pi/6, -math.pi/6]
    start_configs.append((pos9, ang9))
    
    # Config 10: Random perturbation of hex grid
    np.random.seed(123)
    pos10 = [(x + np.random.uniform(-0.3, 0.3), y + np.random.uniform(-0.3, 0.3)) for x, y in pos1]
    ang10 = [np.random.uniform(-0.2, 0.2) for _ in range(N)]
    start_configs.append((pos10, ang10))
    
    for idx, (init_pos, init_ang) in enumerate(start_configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        result = try_configuration(init_pos, init_ang, iterations=2000)
        if result is not None:
            S, pos, ang, beta, center = result
            if S < best_S:
                best_S = S
                best_result = (pos, ang, beta, center)
    
    if best_result is None:
        # Fallback: return the initial hex grid with a computed enclosing hexagon
        pos = np.array([axial_to_cartesian(a) for a in hex_grid])
        ang = [0.0] * N
        all_points = []
        for i in range(N):
            vi = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
            all_points.extend(vi.tolist())
        all_points = np.array(all_points)
        S, beta, center = _compute_min_hexagon(all_points)
        best_result = (pos, ang, beta, center)
        best_S = S
    
    pos, ang, beta, center = best_result
    cx, cy = center
    angle_deg = math.degrees(beta)
    
    # Scale points very slightly to ensure no overlap due to floating point
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for i in range(N):
        sx = cx + (pos[i, 0] - cx) * SCALE
        sy = cy + (pos[i, 1] - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    # Recompute the enclosing hexagon with scaled points
    all_points = []
    for i in range(N):
        vi = _hex_vertices(pts_scaled[i][0], pts_scaled[i][1], 1.0, ang[i])
        all_points.extend(vi.tolist())
    all_points = np.array(all_points)
    
    S, beta, center = _compute_min_hexagon(all_points)
    cx, cy = center
    angle_deg = math.degrees(beta)
    
    inner = []
    for i in range(N):
        inner.append([float(pts_scaled[i][0]), float(pts_scaled[i][1]), float(math.degrees(ang[i]))])
    
    center_out = [float(cx), float(cy)]
    side = float(S)
    rotation = float(angle_deg)
    
    return inner, center_out, side, rotation
