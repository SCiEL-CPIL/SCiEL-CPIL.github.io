import numpy as np
import math
import time
import random

N = 12
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _hex_normals(angle_deg):
    """Edge normals for a unit hexagon with given rotation."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    normals = []
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        rnx = ca * nx - sa * ny
        rny = sa * nx + ca * ny
        normals.append((rnx, rny))
    return normals

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _separating_axis_check(vertsA, vertsB, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Return True if they overlap."""
    axesA = _hex_normals(0)  # We'll use the actual normals from the hexagons
    # Get normals from each polygon's edges
    n = len(vertsA)
    axes = []
    for i in range(n):
        x1, y1 = vertsA[i]
        x2, y2 = vertsA[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    n = len(vertsB)
    for i in range(n):
        x1, y1 = vertsB[i]
        x2, y2 = vertsB[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + tol or maxB <= minA + tol:
            return False  # Separated
    return True  # Overlapping

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles, tol=1e-9):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j], tol):
                return False
    return True

def _sdf_distance(p1x, p1y, a1, p2x, p2y, a2):
    """
    Compute the signed distance between two unit hexagons.
    Positive means separated, negative means overlapping.
    """
    dx = p2x - p1x
    dy = p2y - p1y
    dist = math.hypot(dx, dy)
    
    if dist < 1e-10:
        return 0.0
    
    def hex_radius(angle_deg, direction_deg):
        a_rad = math.radians(angle_deg)
        d_rad = math.radians(direction_deg)
        dx_dir = math.cos(d_rad)
        dy_dir = math.sin(d_rad)
        
        max_val = 0.0
        for k in range(6):
            theta = math.radians(60 * k)
            vx = math.cos(theta)
            vy = math.sin(theta)
            ca = math.cos(a_rad)
            sa = math.sin(a_rad)
            rvx = ca * vx - sa * vy
            rvy = sa * vx + ca * vy
            dot = rvx * dx_dir + rvy * dy_dir
            if dot > max_val:
                max_val = dot
        return max_val
    
    dir_deg = math.degrees(math.atan2(dy, dx))
    r1 = hex_radius(a1, dir_deg)
    r2 = hex_radius(a2, dir_deg + 180)
    
    return dist - r1 - r2

def _smooth_overlap_penalty(centers_angles, beta=50.0):
    """
    Smooth penalty for overlapping hexagons using soft-min.
    Returns (penalty_value, ) - we'll use Nelder-Mead so no gradient needed.
    """
    m = len(centers_angles)
    penalty = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            d = _sdf_distance(
                centers_angles[i][0], centers_angles[i][1], centers_angles[i][2],
                centers_angles[j][0], centers_angles[j][1], centers_angles[j][2]
            )
            # If d < 0, they overlap. Use soft penalty.
            if d < 0:
                # Smooth penalty: -d * exp(beta * d) is smooth but let's use quadratic
                # Actually, use a smooth min: penalty = max(0, -d)^2 with smoothing
                # For Nelder-Mead, we just need a smooth function
                penalty += d * d  # Quadratic penalty when overlapping
    return penalty

def _objective(params, beta=50.0):
    """
    Objective function for Nelder-Mead.
    params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, cx, cy, cangle]
    Returns: outer_side + penalty_weight * overlap_penalty
    """
    # Parse parameters
    centers_angles = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i + 1]
        a = params[3*i + 2]
        centers_angles.append((x, y, a))
    
    cx = params[3*N]
    cy = params[3*N + 1]
    cangle = params[3*N + 2]
    
    # Compute outer side length
    side = _outer_hex_side(centers_angles, (cx, cy), cangle)
    
    # Compute overlap penalty
    penalty = _smooth_overlap_penalty(centers_angles, beta)
    
    # Combined objective
    penalty_weight = 10.0
    return side + penalty_weight * penalty

def _make_hex_lattice():
    """3x4 hexagonal lattice arrangement."""
    pts = []
    row_height = math.sqrt(3) * 1.0
    col_width = 2.0 * 1.0
    for r in range(3):
        for c in range(4):
            x = c * col_width + (col_width / 2 if r % 2 else 0.0)
            y = r * row_height
            pts.append([x - 3.0, y - row_height, 0.0])
    return pts

def _make_ring():
    """12 hexagons in a ring, oriented outward."""
    pts = []
    R = 2.5
    for i in range(N):
        theta = 2 * math.pi * i / N
        x = R * math.cos(theta)
        y = R * math.sin(theta)
        angle_deg = math.degrees(theta)
        pts.append([x, y, angle_deg])
    return pts

def _make_flower():
    """Flower pattern: 1 center + 6 inner ring + 5 outer."""
    pts = []
    pts.append([0.0, 0.0, 0.0])
    R1 = math.sqrt(3)
    for i in range(6):
        theta = 2 * math.pi * i / 6
        x = R1 * math.cos(theta)
        y = R1 * math.sin(theta)
        pts.append([x, y, 0.0])
    R2 = 2 * math.sqrt(3)
    for i in range(5):
        theta = 2 * math.pi * (i + 0.5) / 5
        x = R2 * math.cos(theta)
        y = R2 * math.sin(theta)
        pts.append([x, y, 0.0])
    return pts

def _make_dense_cluster():
    """Tight cluster: center + 6 neighbors + 5 more."""
    pts = []
    pts.append([0.0, 0.0, 0.0])
    for i in range(6):
        theta = 2 * math.pi * i / 6
        x = math.sqrt(3) * math.cos(theta)
        y = math.sqrt(3) * math.sin(theta)
        pts.append([x, y, 0.0])
    for i in range(5):
        theta = 2 * math.pi * (i + 0.5) / 5
        x = 2 * math.sqrt(3) * math.cos(theta)
        y = 2 * math.sqrt(3) * math.sin(theta)
        pts.append([x, y, 0.0])
    return pts

def _make_hexagonal_12():
    """12 hexagons: center + 5 inner + 6 outer."""
    pts = []
    pts.append([0.0, 0.0, 0.0])
    R1 = math.sqrt(3)
    for i in range(5):
        theta = 2 * math.pi * i / 5 + math.pi/5
        x = R1 * math.cos(theta)
        y = R1 * math.sin(theta)
        pts.append([x, y, 0.0])
    R2 = 2 * math.sqrt(3)
    for i in range(6):
        theta = 2 * math.pi * i / 6
        x = R2 * math.cos(theta)
        y = R2 * math.sin(theta)
        pts.append([x, y, 0.0])
    return pts

def _make_random(seed_val):
    """Random configuration with rejection sampling."""
    np.random.seed(seed_val)
    random.seed(seed_val)
    pts = []
    max_attempts = 1000
    for _ in range(max_attempts):
        if len(pts) == N:
            break
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
        a = random.uniform(0, 60)
        valid = True
        for (px, py, pa) in pts:
            d = _sdf_distance(x, y, a, px, py, pa)
            if d < 0:
                valid = False
                break
        if valid:
            pts.append([x, y, a])
    # Pad if needed
    while len(pts) < N:
        pts.append([0.0, 0.0, 0.0])
    return pts[:N]

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Generate initial guesses
    initial_guesses = []
    
    # Various structured initializations
    initial_guesses.append(_make_hex_lattice())
    initial_guesses.append(_make_ring())
    initial_guesses.append(_make_flower())
    initial_guesses.append(_make_dense_cluster())
    initial_guesses.append(_make_hexagonal_12())
    
    # Random initializations
    for seed_val in [1, 2, 3, 4, 5]:
        initial_guesses.append(_make_random(seed_val))
    
    # Optimize each initial guess
    for guess in initial_guesses:
        if time.time() - t0 > TIME_LIMIT - 15:
            break
        
        # Convert guess to params vector: [x1,y1,a1,...,xN,yN,aN, cx,cy,cangle]
        params = []
        for (x, y, a) in guess:
            params.extend([x, y, a])
        params.extend([0.0, 0.0, 0.0])  # Outer center and angle
        
        # Try multiple outer angles
        for outer_angle in [0.0, 30.0, 60.0]:
            if time.time() - t0 > TIME_LIMIT - 10:
                break
            
            # Set outer angle in params
            params[3*N + 2] = outer_angle
            
            try:
                # Use Nelder-Mead
                from scipy.optimize import minimize
                
                # Initial simplex: perturb the initial point
                x0 = np.array(params)
                
                # Create initial simplex
                simplex = [x0.copy()]
                for i in range(len(x0)):
                    step = x0.copy()
                    step[i] += 0.1 * (1.0 + abs(x0[i]))
                    simplex.append(step)
                
                res = minimize(
                    _objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                x_opt = res.x
                centers_opt = []
                for i in range(N):
                    centers_opt.append((x_opt[3*i], x_opt[3*i+1], x_opt[3*i+2]))
                
                cx_opt = x_opt[3*N]
                cy_opt = x_opt[3*N + 1]
                cangle_opt = x_opt[3*N + 2]
                
                # Check validity
                if _is_valid(centers_opt):
                    side = _outer_hex_side(centers_opt, (cx_opt, cy_opt), cangle_opt)
                    
                    if side < best_side:
                        best_side = side
                        best_inner = [[centers_opt[i][0], centers_opt[i][1], centers_opt[i][2]] for i in range(N)]
                        best_outer_center = (cx_opt, cy_opt)
                        best_outer_angle = cangle_opt
                        
            except Exception as e:
                pass
    
    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback: return a valid but not optimal configuration
    fallback = _make_hex_lattice()
    return (fallback, (0.0, 0.0), 40.0, 0.0)
