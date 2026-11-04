<<<<<<< SEARCH
def _convex_hull(points):
    """Andrew's monotone chain; returns list of (x,y) in CCW order."""
    pts = sorted(set((round(p[0], 12), round(p[1], 12)) for p in points))
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

def _polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5

def _point_in_convex(poly, p, tol=1e-9):
    """Check if point p is inside convex polygon poly (CCW)."""
    if len(poly) < 3:
        return False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -tol:
            return False
    return True

def _poly_in_convex(outer, inner, tol=1e-9):
    """Check if all vertices of inner are inside outer (both convex, CCW)."""
    return all(_point_in_convex(outer, (v[0], v[1]), tol) for v in inner)

def _minkowski_sum(poly1, poly2):
    """Minkowski sum of two convex polygons (CCW)."""
    if not poly1 or not poly2:
        return []
    # Sum all vertex pairs
    pts = [(a[0] + b[0], a[1] + b[1]) for a in poly1 for b in poly2]
    return _convex_hull(pts)

def _separation_check(verts_a, verts_b, tol=1e-9):
    """Return True if convex polygons A and B are disjoint (no interior overlap)."""
    # Check if any vertex of A is strictly inside B, or vice versa
    for p in verts_a:
        if _point_in_convex(verts_b, p, tol):
            return False
    for p in verts_b:
        if _point_in_convex(verts_a, p, tol):
            return False
    # Check edge intersections
    def orient(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    def on_segment(p, q, r):
        return (min(p[0], r[0]) - tol <= q[0] <= max(p[0], r[0]) + tol and
                min(p[1], r[1]) - tol <= q[1] <= max(p[1], r[1]) + tol)
    def segments_intersect(p1, p2, p3, p4):
        d1 = orient(p3, p4, p1)
        d2 = orient(p3, p4, p2)
        d3 = orient(p1, p2, p3)
        d4 = orient(p1, p2, p4)
        if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and \
           ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)):
            return True
        if abs(d1) <= tol and on_segment(p1, p3, p4):
            return True
        if abs(d2) <= tol and on_segment(p2, p3, p4):
            return True
        if abs(d3) <= tol and on_segment(p3, p1, p2):
            return True
        if abs(d4) <= tol and on_segment(p4, p1, p2):
            return True
        return False
    n1 = len(verts_a)
    n2 = len(verts_b)
    for i in range(n1):
        a1 = verts_a[i]
        a2 = verts_a[(i + 1) % n1]
        for j in range(n2):
            b1 = verts_b[j]
            b2 = verts_b[(j + 1) % n2]
            if segments_intersect(a1, a2, b1, b2):
                return False
    return True

def _inner_hexagon(cx, cy, angle_deg, side=1.0):
    return _hex_verts(angle_deg, cx, cy, side)

def _outer_hexagon(cx, cy, side, angle_deg=0.0):
    return _hex_verts(angle_deg, cx, cy, side)

def _is_valid_packing(inner_configs, outer_center, outer_side, outer_angle=0.0, tol=1e-9):
    """Check all inner hexagons are disjoint and inside outer."""
    outer_verts = _outer_hexagon(outer_center[0], outer_center[1], outer_side, outer_angle)
    inner_verts_list = []
    for (cx, cy, ang) in inner_configs:
        v = _inner_hexagon(cx, cy, ang)
        inner_verts_list.append(v)
        if not _poly_in_convex(outer_verts, v, tol):
            return False
    for i in range(len(inner_verts_list)):
        for j in range(i + 1, len(inner_verts_list)):
            if not _separation_check(inner_verts_list[i], inner_verts_list[j], tol):
                return False
    return True

def _build_trajectory(t, base_configs):
    """Time-parameterized path: t in [0,1]. At t=0, spread out; at t=1, dense."""
    configs = []
    for k, (cx, cy, ang) in enumerate(base_configs):
        # Interpolate from spread position to target
        # Use a smooth path that avoids collisions by rotating and translating
        scale = 1.0 - 0.8 * t  # shrink spacing
        # Add a small rotational sweep to help escape local minima
        rot_offset = math.radians(30 * (1 - t))
        new_ang = ang + rot_offset
        # Spiral-like inward motion
        r = math.hypot(cx, cy)
        th = math.atan2(cy, cx)
        new_r = r * (1.0 - 0.5 * t * math.sin(math.pi * t))
        new_cx = cx * (1.0 - 0.3 * t) + 0.2 * math.cos(th + math.pi * t) * t
        new_cy = cy * (1.0 - 0.3 * t) + 0.2 * math.sin(th + math.pi * t) * t
        configs.append((new_cx, new_cy, math.degrees(new_ang)))
    return configs

def _optimize_packing(outer_side, outer_center=(0.0, 0.0), outer_angle=0.0, max_iters=200, tol=1e-9):
    """Optimize inner hexagon positions to fit inside outer hexagon."""
    # Start with a symmetric arrangement
    best_configs = None
    best_penalty = float('inf')
    
    for trial in range(5):
        # Initial configuration: ring + center
        random.seed(42 + trial)
        configs = []
        # Place hexagons in a hexagonal lattice pattern
        # For 12 hexagons, use a 3-4-3 ring arrangement
        positions = []
        # Center
        positions.append((0.0, 0.0, 0.0))
        # Inner ring of 6
        for k in range(6):
            th = math.radians(60 * k)
            positions.append((2.2 * math.cos(th), 2.2 * math.sin(th), 0.0))
        # Outer ring of 5 (to make 12 total: 1 + 6 + 5)
        for k in range(5):
            th = math.radians(60 * k + 30)
            positions.append((4.2 * math.cos(th), 4.2 * math.sin(th), 0.0))
        
        configs = positions[:]
        
        # Trajectory optimization
        base_configs = configs[:]
        current_configs = configs[:]
        
        for it in range(max_iters):
            t = it / max_iters
            # Move along trajectory
            traj_configs = _build_trajectory(t, base_configs)
            current_configs = traj_configs[:]
            
            # Local refinement: random perturbations
            if it > max_iters // 2:
                for _ in range(3):
                    new_configs = [list(c) for c in current_configs]
                    for i in range(N):
                        if random.random() < 0.3:
                            new_configs[i][0] += random.uniform(-0.5, 0.5)
                            new_configs[i][1] += random.uniform(-0.5, 0.5)
                        if random.random() < 0.1:
                            new_configs[i][2] += random.uniform(-15, 15)
                    new_configs = [tuple(c) for c in new_configs]
                    
                    # Check validity
                    if _is_valid_packing(new_configs, outer_center, outer_side, outer_angle, tol):
                        current_configs = new_configs
                        break
        
        # Final check
        if _is_valid_packing(current_configs, outer_center, outer_side, outer_angle, tol):
            return current_configs
    
    return None

def _minimize_side_length(outer_center=(0.0, 0.0), outer_angle=0.0, lo=3.0, hi=10.0, tol=1e-4):
    """Binary search for minimum outer side length."""
    # First find a valid upper bound
    while hi < 15.0:
        configs = _optimize_packing(hi, outer_center, outer_angle)
        if configs is not None:
            break
        hi *= 1.2
    
    if hi >= 15.0:
        return None, hi
    
    for _ in range(50):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2
        configs = _optimize_packing(mid, outer_center, outer_angle)
        if configs is not None:
            hi = mid
        else:
            lo = mid
    
    # Return the best configuration at hi
    configs = _optimize_packing(hi, outer_center, outer_angle)
    if configs is None:
        return None, hi
    return configs, hi

def construct():
    """Construct 12 unit hexagons inside a regular hexagon, minimizing outer side."""
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Try to find a good packing
    configs, side = _minimize_side_length(outer_center, outer_angle, lo=3.5, hi=8.0, tol=0.01)
    
    if configs is None:
        # Fallback: simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = [[c[0], c[1], c[2]] for c in configs]
    return inner, outer_center, side, outer_angle
=======
def _is_valid_packing(inner_configs, outer_center, outer_side, outer_angle=0.0, tol=1e-9):
    """Check all inner hexagons are disjoint and inside outer."""
    outer_verts = np.array(_hex_verts(outer_angle, outer_center[0], outer_center[1], outer_side))
    inner_verts_list = []
    for (cx, cy, ang) in inner_configs:
        v = np.array(_hex_verts(ang, cx, cy, 1.0))
        inner_verts_list.append(v)
        if not _polygon_in_convex_polygon(outer_verts, v):
            return False
    for i in range(len(inner_verts_list)):
        for j in range(i + 1, len(inner_verts_list)):
            if not _polygons_disjoint(
                [(p[0], p[1]) for p in inner_verts_list[i]],
                [(p[0], p[1]) for p in inner_verts_list[j]], tol
            ):
                return False
    return True

def _ccd_trajectory_step(configs, outer_side, outer_center, outer_angle, dt=0.02):
    """One CCD step: move configs along a target direction, detect collisions,
    and apply corrective impulses. Returns new configs."""
    n = len(configs)
    new_configs = [list(c) for c in configs]
    
    # Target: pull toward center and tighten
    for i in range(n):
        cx, cy, ang = new_configs[i]
        # Move slightly toward center
        r = math.hypot(cx, cy)
        if r > 0.1:
            new_configs[i][0] -= 0.05 * dt * cx / r
            new_configs[i][1] -= 0.05 * dt * cy / r
    
    # CCD: check pairwise collisions and resolve
    for _ in range(10):  # Multiple resolution iterations
        resolved = True
        for i in range(n):
            for j in range(i + 1, n):
                vi = np.array(_hex_verts(new_configs[i][2], new_configs[i][0], new_configs[i][1], 1.0))
                vj = np.array(_hex_verts(new_configs[j][2], new_configs[j][0], new_configs[j][1], 1.0))
                if not _polygons_disjoint(
                    [(p[0], p[1]) for p in vi],
                    [(p[0], p[1]) for p in vj], 1e-9
                ):
                    resolved = False
                    # Push apart along center-to-center direction
                    dx = new_configs[i][0] - new_configs[j][0]
                    dy = new_configs[i][1] - new_configs[j][1]
                    d = math.hypot(dx, dy)
                    if d < 1e-6:
                        dx, dy = 1.0, 0.0
                        d = 1.0
                    push = 0.1 * dt
                    new_configs[i][0] += push * dx / d
                    new_configs[i][1] += push * dy / d
                    new_configs[j][0] -= push * dx / d
                    new_configs[j][1] -= push * dy / d
        if resolved:
            break
    
    # Containment: push back inside outer hexagon
    outer_verts = np.array(_hex_verts(outer_angle, outer_center[0], outer_center[1], outer_side))
    for i in range(n):
        vi = np.array(_hex_verts(new_configs[i][2], new_configs[i][0], new_configs[i][1], 1.0))
        for v in vi:
            if not _point_in_convex_polygon(outer_verts, v):
                # Move toward center
                dx = outer_center[0] - new_configs[i][0]
                dy = outer_center[1] - new_configs[i][1]
                d = math.hypot(dx, dy)
                if d > 1e-6:
                    new_configs[i][0] += 0.1 * dt * dx / d
                    new_configs[i][1] += 0.1 * dt * dy / d
                break
    
    return [tuple(c) for c in new_configs]

def _trajectory_optimize(outer_side, outer_center=(0.0, 0.0), outer_angle=0.0,
                         n_steps=500, seed=42):
    """CCD-based trajectory optimization from sparse to dense config."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Initial sparse configuration: hexagonal lattice
    configs = []
    # 1 center + 6 ring + 5 outer = 12
    configs.append((0.0, 0.0, 0.0))
    for k in range(6):
        th = math.radians(60 * k)
        configs.append((3.0 * math.cos(th), 3.0 * math.sin(th), 0.0))
    for k in range(5):
        th = math.radians(60 * k + 30)
        configs.append((5.5 * math.cos(th), 5.5 * math.sin(th), 0.0))
    
    best_configs = None
    best_penalty = float('inf')
    
    for step in range(n_steps):
        # Gradually increase tightening force
        configs = _ccd_trajectory_step(configs, outer_side, outer_center, outer_angle, dt=0.02)
        
        # Occasionally try random perturbations to escape local minima
        if step % 50 == 49:
            perturbed = [list(c) for c in configs]
            for i in range(N):
                if random.random() < 0.3:
                    perturbed[i][0] += random.uniform(-0.3, 0.3)
                    perturbed[i][1] += random.uniform(-0.3, 0.3)
                if random.random() < 0.1:
                    perturbed[i][2] += random.uniform(-10, 10)
            perturbed = [tuple(c) for c in perturbed]
            # Accept if valid or less bad
            if _is_valid_packing(perturbed, outer_center, outer_side, outer_angle):
                configs = perturbed
                best_configs = configs[:]
                return configs
        
        if _is_valid_packing(configs, outer_center, outer_side, outer_angle):
            best_configs = configs[:]
            return configs
    
    return best_configs

def _minimize_side_length(outer_center=(0.0, 0.0), outer_angle=0.0, lo=3.0, hi=10.0, tol=0.01):
    """Binary search for minimum outer side length using trajectory optimizer."""
    # Find valid upper bound
    while hi < 15.0:
        configs = _trajectory_optimize(hi, outer_center, outer_angle, n_steps=300, seed=42)
        if configs is not None:
            break
        hi *= 1.15
    
    if hi >= 15.0:
        return None, hi
    
    # Binary search
    for _ in range(20):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2
        configs = _trajectory_optimize(mid, outer_center, outer_angle, n_steps=400, seed=42)
        if configs is not None:
            hi = mid
        else:
            lo = mid
    
    configs = _trajectory_optimize(hi, outer_center, outer_angle, n_steps=500, seed=42)
    if configs is None:
        return None, hi
    return configs, hi

def construct():
    """Construct 12 unit hexagons inside a regular hexagon, minimizing outer side."""
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    configs, side = _minimize_side_length(outer_center, outer_angle, lo=3.5, hi=8.0, tol=0.01)
    
    if configs is None:
        # Fallback: wide spacing
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = [[c[0], c[1], c[2]] for c in configs]
    return inner, outer_center, side, outer_angle
>>>>>>> REPLACE
