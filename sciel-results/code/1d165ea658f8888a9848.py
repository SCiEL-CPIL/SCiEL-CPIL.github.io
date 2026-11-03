"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0  # 120s safety margin within 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _get_hex_vertices_np(cx, cy, angle_deg=0.0):
    """Get vertices as numpy array for faster computation."""
    angle_rad = math.radians(angle_deg)
    angles = angle_rad + np.pi * np.arange(6) / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """Use LP to find optimal center for given outer hexagon angle.
    Returns (cx, cy, side_length)."""
    from scipy.optimize import linprog
    
    # Collect all vertices as numpy array
    all_verts = []
    for ix, iy, iangle in inner_hexes:
        all_verts.append(_get_hex_vertices_np(ix, iy, iangle))
    all_verts = np.vstack(all_verts)  # shape (6*N, 2)
    
    n_pts = all_verts.shape[0]
    
    # Variables: cx, cy, t (inradius)
    # Minimize t
    c = np.array([0.0, 0.0, 1.0])
    
    # Build constraint matrix efficiently
    # For each of 6 directions, we have 2 sets of N*6 constraints
    A_rows = []
    b_rows = []
    
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        
        # Constraint 1: (vx - cx)*nx + (vy - cy)*ny <= t for all vertices
        # => nx*vx + ny*vy - nx*cx - ny*cy - t <= 0
        # => -nx*cx - ny*cy - t <= -(nx*vx + ny*vy)
        projections = all_verts[:, 0] * nx + all_verts[:, 1] * ny
        A1 = np.column_stack([-nx * np.ones(n_pts), -ny * np.ones(n_pts), -np.ones(n_pts)])
        b1 = -projections
        
        # Constraint 2: -(vx - cx)*nx - (vy - cy)*ny <= t for all vertices
        # => nx*cx + ny*cy - nx*vx - ny*vy - t <= 0
        # => nx*cx + ny*cy - t <= nx*vx + ny*vy
        A2 = np.column_stack([nx * np.ones(n_pts), ny * np.ones(n_pts), -np.ones(n_pts)])
        b2 = projections
        
        A_rows.extend([A1, A2])
        b_rows.extend([b1, b2])
    
    A_ub = np.vstack(A_rows)
    b_ub = np.concatenate(b_rows)
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if result.success:
        cx, cy, t = result.x
        side = t / (math.sqrt(3) / 2)
        return cx, cy, side
    
    # Fallback: use centroid
    cx = all_verts[:, 0].mean()
    cy = all_verts[:, 1].mean()
    required_inradius = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        projections = (all_verts[:, 0] - cx) * nx + (all_verts[:, 1] - cy) * ny
        max_proj = projections.max()
        min_proj = projections.min()
        required_inradius = max(required_inradius, max(max_proj, -min_proj))
    side = required_inradius / (math.sqrt(3) / 2)
    return cx, cy, side

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try different angles for the outer hexagon
    # By symmetry, we only need to check [0, 60) degrees
    # Use finer resolution for better optimization
    for angle_deg in np.arange(0.0, 60.0, 5.0):
        angle_deg = float(angle_deg)
        cx, cy, side = _min_enclosing_hex_side_lp(inner_hexes, angle_deg)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = angle_deg
    
    # Refine around the best angle with finer steps
    if best_side < float('inf'):
        for da in np.arange(-5.0, 5.0, 1.0):
            test_angle = (best_angle + float(da)) % 60.0
            cx, cy, side = _min_enclosing_hex_side_lp(inner_hexes, test_angle)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = test_angle
    
    return best_center[0], best_center[1], best_side, best_angle

def _check_disjoint(inner_hexes):
    """Check if all inner hexagons have disjoint interiors using polygon intersection."""
    n = len(inner_hexes)
    polygons = [_get_hex_vertices(x, y, a) for x, y, a in inner_hexes]
    
    for i in range(n):
        for j in range(i + 1, n):
            if _polygons_overlap(polygons[i], polygons[j]):
                return False
    return True

def _polygons_overlap(poly1, poly2, eps=1e-9):
    """Check if two convex polygons have overlapping interiors."""
    # Check if any vertex of poly1 is inside poly2 (strictly)
    for v in poly1:
        if _point_in_convex_polygon(v, poly2, eps):
            return True
    # Check if any vertex of poly2 is inside poly1 (strictly)
    for v in poly2:
        if _point_in_convex_polygon(v, poly1, eps):
            return True
    # Check if any edges intersect
    for i in range(len(poly1)):
        for j in range(len(poly2)):
            if _segments_intersect(poly1[i], poly1[(i+1)%len(poly1)], 
                                   poly2[j], poly2[(j+1)%len(poly2)], eps):
                return True
    return False

def _point_in_convex_polygon(point, polygon, eps=1e-9):
    """Check if point is strictly inside a convex polygon."""
    px, py = point
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i+1) % n]
        # Cross product to check if point is on the right side of edge
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross > eps:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4, eps=1e-9):
    """Check if two line segments intersect (including endpoints)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    
    # Check collinear cases
    if abs(d1) <= eps and _on_segment(p3, p1, p4, eps):
        return True
    if abs(d2) <= eps and _on_segment(p3, p2, p4, eps):
        return True
    if abs(d3) <= eps and _on_segment(p1, p3, p2, eps):
        return True
    if abs(d4) <= eps and _on_segment(p1, p4, p2, eps):
        return True
    
    return False

def _on_segment(p, q, r, eps=1e-9):
    """Check if point q lies on segment pr."""
    if (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and
        min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps and
        abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])) <= eps):
        return True
    return False

def _smooth_separation_violation(inner_hexes, beta=100.0):
    """Compute smooth approximation of minimum separation violation.
    Returns 0 if all hexagons are sufficiently separated, positive if not."""
    min_required = math.sqrt(3) - 0.001  # required center-to-center distance
    n = len(inner_hexes)
    violations = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = inner_hexes[i][0] - inner_hexes[j][0]
            dy = inner_hexes[i][1] - inner_hexes[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            violations.append(min_required - dist)
    if not violations:
        return 0.0
    # Use log-sum-exp for smooth max
    max_v = max(violations)
    if max_v <= 0:
        return 0.0
    # Soft max: (1/beta) * log(sum(exp(beta*v)))
    scaled = [beta * v for v in violations if v > 0]
    if not scaled:
        return 0.0
    m = max(scaled)
    s = sum(math.exp(v - m) for v in scaled)
    return (m + math.log(s)) / beta

def _compute_enclosing_side_fast(inner_hexes, angle_deg):
    """Fast computation of enclosing hexagon side for given angle using numpy."""
    all_verts = []
    for ix, iy, iangle in inner_hexes:
        all_verts.append(_get_hex_vertices_np(ix, iy, iangle))
    all_verts = np.vstack(all_verts)
    
    # For a given outer hex angle, the inradius is determined by the max projection
    # across all 6 face normals, centered optimally
    # We can compute the required inradius for a fixed center (centroid) quickly
    cx = all_verts[:, 0].mean()
    cy = all_verts[:, 1].mean()
    
    required_inradius = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        projections = (all_verts[:, 0] - cx) * nx + (all_verts[:, 1] - cy) * ny
        max_proj = projections.max()
        min_proj = projections.min()
        required_inradius = max(required_inradius, max(max_proj, -min_proj))
    
    side = required_inradius / (math.sqrt(3) / 2)
    return cx, cy, side

def _slsqp_polish(inner_hexes, max_iter=100, tol=1e-8, beta=50.0, outer_angle=30.0):
    """Polish inner hexagon positions using SLSQP with smooth constraints."""
    from scipy.optimize import minimize
    
    n_params = N * 3
    
    def objective(params):
        new_inner = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            new_inner.append([x, y, a])
        
        # Smooth separation penalty
        sep_viol = _smooth_separation_violation(new_inner, beta)
        
        # Compute enclosing hexagon side using fast method with fixed angle
        cx, cy, side = _compute_enclosing_side_fast(new_inner, outer_angle)
        
        return side + 10.0 * sep_viol
    
    # Initial params
    x0 = []
    for h in inner_hexes:
        x0.extend([h[0], h[1], h[2]])
    
    result = minimize(objective, x0, method='SLSQP', 
                      options={'maxiter': max_iter, 'ftol': tol, 'disp': False})
    
    if result.success or result.fun < 100.0:
        polished = []
        for i in range(N):
            polished.append([result.x[i*3], result.x[i*3+1], result.x[i*3+2]])
        # Verify the result is valid
        if _check_disjoint(polished):
            return polished
    
    return inner_hexes

def _local_polish(inner_hexes, max_iter=200, tol=1e-9, outer_angle=30.0):
    """High-precision local polish using coordinate descent with fine steps."""
    best = [h[:] for h in inner_hexes]
    _, _, best_side = _compute_enclosing_side_fast(best, outer_angle)
    
    step = 0.01
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 10.0:
            break
        improved = False
        for i in range(N):
            for param_idx in range(3):
                for direction in [-1, 1]:
                    trial = [h[:] for h in best]
                    trial[i][param_idx] += direction * step
                    if _check_disjoint(trial):
                        _, _, side = _compute_enclosing_side_fast(trial, outer_angle)
                        if side < best_side - 1e-10:
                            best = trial
                            best_side = side
                            improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    
    return best

def _generate_hex_cluster_positions(scale=1.001, angle=30.0):
    """Generate 11 positions in a hexagonal cluster on triangular lattice."""
    sqrt3 = math.sqrt(3)
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
    
    return [[pos[0], pos[1], angle] for pos in positions]

def _generate_ring_positions(scale=1.001, angle=30.0):
    """Generate 11 positions: center + ring of 6 + ring of 4."""
    sqrt3 = math.sqrt(3)
    positions = [[0.0, 0.0, angle]]
    
    # First ring: 6 hexagons at distance sqrt(3)*scale
    for k in range(6):
        a = math.radians(60 * k)
        x = sqrt3 * scale * math.cos(a)
        y = sqrt3 * scale * math.sin(a)
        positions.append([x, y, angle])
    
    # Second ring: 4 hexagons (alternate positions)
    r2 = 2 * sqrt3 * scale
    for k in [0, 2, 4, 6]:
        a = math.radians(60 * k + 30)
        x = r2 * math.cos(a)
        y = r2 * math.sin(a)
        positions.append([x, y, angle])
    
    return positions[:N]

def _generate_compact_positions(scale=1.001, angle=30.0):
    """Generate 11 positions in a compact arrangement."""
    sqrt3 = math.sqrt(3)
    # Use a more compact packing
    positions = [[0.0, 0.0, angle]]
    
    # Ring 1: 6 hexagons
    for k in range(6):
        a = math.radians(60 * k)
        x = sqrt3 * scale * math.cos(a)
        y = sqrt3 * scale * math.sin(a)
        positions.append([x, y, angle])
    
    # Ring 2: 4 hexagons at 120-degree intervals
    r2 = sqrt3 * scale * math.sqrt(3)  # distance to second ring
    for k in [1, 3, 5, 7]:
        a = math.radians(60 * k)
        x = r2 * math.cos(a)
        y = r2 * math.sin(a)
        positions.append([x, y, angle])
    
    return positions[:N]

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate multiple candidate configurations
    configs = []
    
    # Config 1: Hexagonal cluster (triangular lattice)
    config1 = _generate_hex_cluster_positions(scale=1.001, angle=30.0)
    if _check_disjoint(config1):
        configs.append(config1)
    
    # Config 2: Ring arrangement
    config2 = _generate_ring_positions(scale=1.001, angle=30.0)
    if _check_disjoint(config2):
        configs.append(config2)
    
    # Config 3: Compact arrangement
    config3 = _generate_compact_positions(scale=1.001, angle=30.0)
    if _check_disjoint(config3):
        configs.append(config3)
    
    # Config 4: Different angle
    config4 = _generate_hex_cluster_positions(scale=1.001, angle=0.0)
    if _check_disjoint(config4):
        configs.append(config4)
    
    # Config 5: Slightly larger scale for more breathing room
    config5 = _generate_hex_cluster_positions(scale=1.005, angle=30.0)
    if _check_disjoint(config5):
        configs.append(config5)
    
    if not configs:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Evaluate all configs and find the best
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for config in configs:
        if time.time() > _DEADLINE - 15.0:
            break
        cx, cy, side, angle = _min_enclosing_hex_side(config)
        if side < best_side:
            best_side = side
            best_inner = [h[:] for h in config]
            best_center = (cx, cy)
            best_angle = angle
    
    if best_inner is None:
        best_inner = configs[0]
        cx, cy, side, angle = _min_enclosing_hex_side(best_inner)
        best_side = side
        best_center = (cx, cy)
        best_angle = angle
    
    # Try SLSQP polish on the best config
    if time.time() < _DEADLINE - 40.0:
        polished = _slsqp_polish(best_inner, max_iter=150, tol=1e-9, beta=50.0, outer_angle=best_angle)
        if polished is not best_inner:
            pcx, pcy, pside, pangle = _min_enclosing_hex_side(polished)
            if pside < best_side - 1e-6:
                best_side = pside
                best_inner = polished
                best_center = (pcx, pcy)
                best_angle = pangle
    
    # Try SLSQP on second-best config if time permits
    if len(configs) > 1 and time.time() < _DEADLINE - 40.0:
        second_best = configs[1] if best_inner is not configs[0] else configs[0]
        polished2 = _slsqp_polish(second_best, max_iter=100, tol=1e-9, beta=50.0, outer_angle=best_angle)
        if polished2 is not second_best:
            pcx2, pcy2, pside2, pangle2 = _min_enclosing_hex_side(polished2)
            if pside2 < best_side - 1e-6:
                best_side = pside2
                best_inner = polished2
                best_center = (pcx2, pcy2)
                best_angle = pangle2
    
    # Final high-precision LP polish for outer hexagon
    if time.time() < _DEADLINE - 15.0:
        # Try finer angles around the best angle
        for da in np.arange(-5.0, 5.0, 0.5):
            test_angle = (best_angle + float(da)) % 60.0
            tcx, tcy, tside = _min_enclosing_hex_side_lp(best_inner, test_angle)
            if tside < best_side - 1e-10:
                best_side = tside
                best_center = (tcx, tcy)
                best_angle = test_angle
        
        # High-precision local polish if time permits
        if time.time() < _DEADLINE - 15.0:
            polished3 = _local_polish(best_inner, max_iter=300, tol=1e-10, outer_angle=best_angle)
            if polished3 is not best_inner:
                pcx3, pcy3, pside3, pangle3 = _min_enclosing_hex_side(polished3)
                if pside3 < best_side - 1e-8:
                    best_side = pside3
                    best_inner = polished3
                    best_center = (pcx3, pcy3)
                    best_angle = pangle3
    
    return best_inner, (best_center[0], best_center[1]), best_side, best_angle
# EVOLVE-BLOCK-END
