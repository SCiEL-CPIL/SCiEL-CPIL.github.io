"""Packing 12 unit hexagons inside a regular hexagon; minimize outer side length."""
import math
import time
import numpy as np

N = 12
DEADLINE = 110.0  # seconds; safety margin before 120s kill

def _hex_vertices(cx, cy, s, angle_deg):
    """Return the 6 vertices of a regular hexagon centered at (cx,cy) with side s and rotation."""
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        pts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return pts

def _support_function(points, ux, uy):
    """Max of dot(p, u) over points."""
    return max(ux * px + uy * py for px, py in points)

def _convex_hull(points):
    """Andrew's monotone chain; returns hull in CCW order (no duplicate last point)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def _outer_side_from_hull(hull):
    """
    Given the convex hull of all inner-hexagon vertices, compute the minimum side length
    of a regular hexagon (any center, any rotation) that contains the hull.
    Use the support-function approach: for a regular hexagon of side s centered at c with rotation phi,
    the support in direction u is c·u + s * h_hex(u - phi), where h_hex is the support of a unit hexagon.
    We need s such that for all directions u in a dense set, max_p p·u <= c·u + s * h_hex(u - phi).
    This is equivalent to: s >= max_u (max_p p·u - c·u) / h_hex(u - phi).
    To minimize s over c and phi, we can note that the optimal c is the center of the minimal enclosing hexagon.
    For a fixed phi, the problem reduces to finding c that minimizes max_u (max_p p·u - c·u)/h_hex(u-phi).
    This is a linear programming problem in c. For simplicity and speed, we sample phi and for each
    compute the required s assuming c is the centroid of the hull (a good heuristic), then refine c
    by a few iterations of coordinate descent on the max constraint.
    """
    if len(hull) == 0:
        return 0.0, (0.0, 0.0), 0.0
    if len(hull) == 1:
        return 1.0, hull[0], 0.0
    if len(hull) == 2:
        dx = hull[1][0] - hull[0][0]
        dy = hull[1][1] - hull[0][1]
        length = math.hypot(dx, dy)
        return length, ((hull[0][0] + hull[1][0]) / 2, (hull[0][1] + hull[1][1]) / 2), 0.0

    # Sample rotations
    best_s = float('inf')
    best_c = (0.0, 0.0)
    best_phi = 0.0

    # Directions to check: dense set of 72 directions
    n_dirs = 72
    dirs = [math.radians(i * 360.0 / n_dirs) for i in range(n_dirs)]

    for phi_deg in range(0, 360, 15):
        phi = math.radians(phi_deg)
        # Support of unit hexagon (side 1) in direction theta:
        # A regular hexagon with side 1 has vertices at distance 1 from center at angles 0,60,...,300.
        # Support in direction u is max over vertices of cos(u_angle - vertex_angle).
        # For a hexagon, h(theta) = max_k cos(theta - k*pi/3).
        # This equals: if theta mod 60 is in [-30,30], h = cos(theta_mod_60).
        def h_hex(theta):
            t = theta % (2 * math.pi)
            # Reduce to [0, 2pi)
            # The max of cos(t - k*pi/3) for k=0..5
            # This is cos of the angle from t to the nearest multiple of pi/3, but capped.
            # Actually, for a hexagon, the support function is:
            # h(t) = cos(t - round(t/(pi/3))*(pi/3)) if |t - round(...)| <= pi/6
            # More simply: h(t) = max(cos(t), cos(t-pi/3), cos(t-2pi/3), cos(t-pi), cos(t-4pi/3), cos(t-5pi/3))
            best = -1.0
            for k in range(6):
                val = math.cos(t - k * math.pi / 3.0)
                if val > best:
                    best = val
            return best

        # For a given phi, compute the "required" side length for each direction.
        # s >= (max_p p·u - c·u) / h_hex(u - phi) for all u.
        # To minimize s, choose c to minimize the max over u of that expression.
        # Start with c = centroid.
        cx = sum(p[0] for p in hull) / len(hull)
        cy = sum(p[1] for p in hull) / len(hull)

        # A few iterations of gradient-free refinement: try small shifts in c.
        for _iter in range(3):
            # Compute current s
            s_cur = 0.0
            for u_deg in dirs:
                ux, uy = math.cos(u_deg), math.sin(u_deg)
                sup = max(ux * px + uy * py for px, py in hull)
                h_val = h_hex(u_deg - phi)
                if h_val < 1e-9:
                    h_val = 1e-9
                required = (sup - (ux * cx + uy * cy)) / h_val
                if required > s_cur:
                    s_cur = required
            if s_cur < 0:
                s_cur = 0.0

            # Try shifting c to reduce the max
            improved = False
            for dx in (-0.1, 0.0, 0.1):
                for dy in (-0.1, 0.0, 0.1):
                    nx, ny = cx + dx, cy + dy
                    s_new = 0.0
                    for u_deg in dirs:
                        ux, uy = math.cos(u_deg), math.sin(u_deg)
                        sup = max(ux * px + uy * py for px, py in hull)
                        h_val = h_hex(u_deg - phi)
                        if h_val < 1e-9:
                            h_val = 1e-9
                        required = (sup - (ux * nx + uy * ny)) / h_val
                        if required > s_new:
                            s_new = required
                    if s_new < s_cur - 1e-6:
                        cx, cy = nx, ny
                        s_cur = s_new
                        improved = True
            if not improved:
                break

        if s_cur < best_s:
            best_s = s_cur
            best_c = (cx, cy)
            best_phi = phi_deg

    return best_s, best_c, best_phi

def construct():
    start_time = time.time()
    
    # Initial configuration: 12 hexagons in a 3x4 grid-like arrangement
    # Hexagonal lattice: for side-1 hexagons, horizontal spacing is 3 (if flat-top) or sqrt(3) (if pointy-top).
    # Let's use pointy-top hexagons (angle 90 or 0). For pointy-top, vertices at 30,90,150,210,270,330.
    # Horizontal distance between adjacent columns: sqrt(3).
    # Vertical distance: 1.5.
    
    # Place 12 hexagons in a 4-column x 3-row hexagonal grid.
    # Columns 0,1,2,3. Rows depend on column parity.
    # For column c, x = c * sqrt(3).
    # For row r in column c, y = (r - 1) * 1.5 + (0 if c even else 0.75).
    
    sqrt3 = math.sqrt(3.0)
    inner = []
    for c in range(4):
        x = c * sqrt3
        for r in range(3):
            y = (r - 1) * 1.5 + (0.0 if c % 2 == 0 else 0.75)
            inner.append([x, y, 0.0])  # angle 0 = pointy-top? Let's check.
    
    # Actually, let's just use angle 0 for all. The orientation of the inner hexagons
    # can be optimized later. For now, compute the outer hexagon.
    
    # Compute all vertices of all inner hexagons.
    all_pts = []
    for (cx, cy, ang) in inner:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_pts.extend(verts)
    
    hull = _convex_hull(all_pts)
    s, c, phi = _outer_side_from_hull(hull)
    
    best_score = s
    best_inner = [list(x) for x in inner]
    best_c = c
    best_phi = phi
    
    # Simple local search: perturb positions and angles, accept if outer side decreases.
    rng = np.random.default_rng(42)
    
    def evaluate(inner_cfg):
        t0 = time.time()
        all_pts = []
        for (cx, cy, ang) in inner_cfg:
            verts = _hex_vertices(cx, cy, 1.0, ang)
            all_pts.extend(verts)
        hull = _convex_hull(all_pts)
        s, c, phi = _outer_side_from_hull(hull)
        return s, c, phi
    
    current_inner = [list(x) for x in inner]
    current_s, current_c, current_phi = evaluate(current_inner)
    
    step = 0.5
    while time.time() - start_time < DEADLINE:
        # Pick a random hexagon to perturb
        i = rng.integers(0, N)
        dx = rng.normal(0, step)
        dy = rng.normal(0, step)
        dangle = rng.normal(0, step * 10)
        
        new_inner = [list(x) for x in current_inner]
        new_inner[i][0] += dx
        new_inner[i][1] += dy
        new_inner[i][2] = (new_inner[i][2] + dangle) % 360
        
        new_s, new_c, new_phi = evaluate(new_inner)
        
        if new_s < current_s:
            current_inner = new_inner
            current_s = new_s
            current_c = new_c
            current_phi = new_phi
            if current_s < best_score:
                best_score = current_s
                best_inner = [list(x) for x in current_inner]
                best_c = current_c
                best_phi = current_phi
            step = min(step * 1.1, 2.0)
        else:
            step = max(step * 0.9, 0.05)
    
    return best_inner, best_c, best_score, best_phi

