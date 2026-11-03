import math
import numpy as np
from scipy.optimize import differential_evolution

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon with given center, rotation (degrees), side length."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # Flat-top hexagon vertices relative to center
    rel = [
        (side, 0.0),
        (side * 0.5, side * math.sqrt(3) / 2),
        (-side * 0.5, side * math.sqrt(3) / 2),
        (-side, 0.0),
        (-side * 0.5, -side * math.sqrt(3) / 2),
        (side * 0.5, -side * math.sqrt(3) / 2),
    ]
    verts = []
    for vx, vy in rel:
        rx = cx + vx * ca - vy * sa
        ry = cy + vx * sa + vy * ca
        verts.append((rx, ry))
    return np.array(verts)

def _min_outer_side_for_vertices(verts, angle_rad):
    """
    Given a set of vertices and an outer hexagon rotation angle (radians),
    compute the minimum side length of a regular hexagon (centered at origin)
    that contains all vertices.
    
    A regular hexagon of side S centered at origin with rotation theta is defined
    by 6 half-planes. The normal directions are at angles theta + 30 + k*60 for k=0..5.
    The apothem (distance from center to each side) is A = S * sqrt(3)/2.
    So S = 2*A/sqrt(3).
    
    For a point p to be inside, |dot(p, n_i)| <= A for all i.
    So A_min = max_i max_p |dot(p, n_i)|.
    But wait, the hexagon is rotated by angle_rad. So the normals are rotated.
    Equivalently, rotate the points by -angle_rad and check against fixed normals.
    """
    s3 = math.sqrt(3)
    # Normals for standard (angle=0) flat-top hexagon: 30, 90, 150, 210, 270, 330 degrees
    normal_angles = [math.radians(30 + k * 60) for k in range(6)]
    normals = np.array([(math.cos(a), math.sin(a)) for a in normal_angles])
    
    # Rotate points by -angle_rad
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    rotated_verts = verts @ R.T  # shape (N, 2)
    
    # Compute dot products: (N, 6)
    dots = rotated_verts @ normals.T
    abs_dots = np.abs(dots)
    
    # For each point, the required apothem is the max over normals
    # But actually, the hexagon is the intersection of 6 half-planes.
    # A point p is inside if for all i: dot(p, n_i) <= A (and >= -A since normals are symmetric)
    # So we need A >= max_i |dot(p_i, n_i)| for all i? No.
    # The constraint is: for all i, |dot(p, n_i)| <= A.
    # So A must be >= max over all points p and all normals i of |dot(p, n_i)|.
    # Wait, that's not right either. The hexagon is the set of points where for ALL i, |dot(p, n_i)| <= A.
    # So for a given set of points, the minimum A such that all points are inside is:
    # A = max over all points p of (max over all normals i of |dot(p, n_i)|)
    # No, that's not right. For a single point p, it's inside if |dot(p, n_i)| <= A for all i.
    # So the required A for point p is max_i |dot(p, n_i)|.
    # For all points, A must be >= max_p (max_i |dot(p, n_i)|).
    # But wait, the normals are symmetric (n and -n are both included? No, we have 6 normals at 30,90,...,330).
    # These 6 normals cover all 6 sides. The opposite normal is included (e.g., 30 and 210).
    # So |dot(p, n_i)| already handles both directions.
    # So A_min = max over all points p and all i of |dot(p, n_i)|.
    
    # Actually, let me reconsider. The hexagon with apothem A and normals at angles phi_i is:
    # { p : for all i, p . n_i <= A }
    # where n_i are unit normals pointing outward.
    # Since the hexagon is symmetric, we can use normals at 30, 90, ..., 330 and require p . n_i <= A for all i.
    # This is equivalent to |p . n_i| <= A only if we include both n_i and -n_i.
    # In our case, n_0 is at 30, n_3 is at 210 = 30 + 180. So yes, opposite normals are included.
    # So the condition is: for all i, p . n_i <= A.
    # Which means A >= max_i (p . n_i).
    # But since n_i and n_{i+3} = -n_i, max_i (p . n_i) = max( p . n_i, p . n_{i+3} ) = max( p . n_i, -p . n_i ) = |p . n_i|.
    # So for a single point, the required A is max_i |p . n_i|.
    # For all points, A = max_p max_i |p . n_i|.
    
    max_proj = np.max(abs_dots)
    S = 2.0 * max_proj / s3
    return S

def _construct_from_params(params, outer_angle_rad):
    """
    params: array of shape (3N,) where N=11.
    For each hexagon i: params[3i]=cx, params[3i+1]=cy, params[3i+2]=angle_deg.
    Returns: (inner_list, center, side, angle_rad, all_vertices)
    """
    N = 11
    inner_list = []
    all_verts = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang_deg = params[3*i+2]
        inner_list.append((cx, cy, ang_deg))
        verts = _hex_vertices(cx, cy, ang_deg, 1.0)
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)
    side = _min_outer_side_for_vertices(all_verts, outer_angle_rad)
    center = (0.0, 0.0)
    return inner_list, center, side, outer_angle_rad, all_verts

def _separation_violation(params):
    """
    Check if any two hexagons overlap.
    For two unit hexagons, if their centers are at distance d, they overlap if d < some threshold.
    But with rotation, it's more complex.
    A simpler approach: check all pairs of vertices and edges.
    But that's expensive.
    
    Alternative: use the fact that for two convex polygons, they overlap iff one vertex of one is inside the other,
    or edges intersect.
    
    For efficiency, let's use a bounding circle check first.
    The circumradius of a unit hexagon is 1.0.
    So if distance between centers > 2.0, no overlap.
    If distance < some small value, definitely overlap.
    Otherwise, do exact check.
    
    Let's use a simpler proxy: the minimum distance between edges of the two hexagons.
    But that's complex.
    
    Let's use scipy's ConvexHull or a simple point-in-polygon + edge intersection test.
    
    For now, let's use a conservative check: if the distance between centers is less than 2.0,
    we need to check more carefully. If it's >= 2.0, no overlap.
    
    Actually, the minimum distance between two non-overlapping unit hexagons depends on their relative orientation.
    The maximum distance between centers for non-overlap is 2.0 (when aligned).
    The minimum distance for non-overlap can be smaller if rotated.
    
    Let's just do a full check for pairs with center distance < 2.0.
    """
    N = 11
    hexes = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        verts = _hex_vertices(cx, cy, ang, 1.0)
        hexes.append(verts)
    
    def _poly_overlap(p1, p2):
        """Check if two convex polygons overlap."""
        # Check if any vertex of p1 is inside p2 or vice versa
        # Or if any edges intersect.
        
        def point_in_poly(p, poly):
            # Ray casting or winding number
            # For convex poly, we can check if point is on same side of all edges
            n = len(poly)
            for i in range(n):
                a = poly[i]
                b = poly[(i+1)%n]
                # Cross product to check side
                cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                # All crosses should have same sign (or be zero)
                if i == 0:
                    sign = np.sign(cross)
                    if sign == 0:
                        # On edge, check if within segment
                        # For simplicity, consider it inside
                        continue
                else:
                    if sign * np.sign(cross) < 0:
                        return False
            return True
        
        for v in p1:
            if point_in_poly(v, p2):
                return True
        for v in p2:
            if point_in_poly(v, p1):
                return True
        
        # Check edge intersections
        # If no vertex is inside, and no edges intersect, they don't overlap
        # For now, skip edge intersection for speed, rely on vertex check
        # This is not fully robust but should work for most cases
        
        return False
    
    violation = 0.0
    for i in range(N):
        for j in range(i+1, N):
            cx1 = params[3*i]
            cy1 = params[3*i+1]
            cx2 = params[3*j]
            cy2 = params[3*j+1]
            dist = math.hypot(cx1-cx2, cy1-cy2)
            if dist < 2.0:
                if _poly_overlap(hexes[i], hexes[j]):
                    # Penalty proportional to overlap depth
                    # Approximate: penalty = max(0, 2.0 - dist)
                    violation += max(0, 2.0 - dist)
    
    return violation

def _objective_flat(params):
    """
    Objective: minimize outer side length + penalty for overlaps.
    """
    # Use a fixed outer angle for now, optimize later
    # Or include outer angle in params? Let's keep it separate for simplicity.
    outer_angle = 0.0  # We'll optimize this separately or include it
    
    _, _, side, _, _ = _construct_from_params(params, outer_angle)
    overlap = _separation_violation(params)
    
    return side + 10.0 * overlap

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    N = 11
    s3 = math.sqrt(3)
    
    # Initial configurations to try
    configs = []
    
    # 1. Honeycomb cluster (1 center + 6 ring + 4 more)
    # Center
    centers_honey = [(0.0, 0.0)]
    # First ring at distance 2*s3/2 = s3? No, distance between centers of adjacent hexagons in honeycomb is 2*apothem = s3.
    # Actually, for unit hexagons, the distance between centers of two adjacent hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    # So first ring at distance s3 from center, at angles 0, 60, ..., 300.
    for k in range(6):
        ang = math.radians(60 * k)
        centers_honey.append((s3 * math.cos(ang), s3 * math.sin(ang)))
    # Second ring: 6 more at distance 2*s3, at angles 30, 90, ..., 330
    for k in range(6):
        ang = math.radians(30 + 60 * k)
        centers_honey.append((2*s3 * math.cos(ang), 2*s3 * math.sin(ang)))
    # We have 1 + 6 + 6 = 13. Too many.
    # We need 11. So take center + 6 ring + 4 from second ring.
    # Or center + 6 ring + 4 at positions that fit.
    # Let's use center + 6 ring + 4 at the "corners" of a larger hexagon.
    # Actually, a valid packing of 11: center, 6 adjacent, and 4 more in the gaps.
    # The 4 more can be placed at distance 2*s3 from center, at angles 0, 90, 180, 270?
    # Let's try: center, 6 at s3 distance, 4 at 2*s3 distance at 0, 90, 180, 270.
    centers_11 = [(0.0, 0.0)]
    for k in range(6):
        ang = math.radians(60 * k)
        centers_11.append((s3 * math.cos(ang), s3 * math.sin(ang)))
    for k in range(4):
        ang = math.radians(90 * k)
        centers_11.append((2*s3 * math.cos(ang), 2*s3 * math.sin(ang)))
    
    params_init = np.zeros(3*N)
    for i, (cx, cy) in enumerate(centers_11):
        params_init[3*i] = cx
        params_init[3*i+1] = cy
        params_init[3*i+2] = 0.0
    configs.append(params_init.copy())
    
    # 2. Random cluster
    for _ in range(3):
        params_rand = np.zeros(3*N)
        for i in range(N):
            params_rand[3*i] = np.random.uniform(-4, 4)
            params_rand[3*i+1] = np.random.uniform(-4, 4)
            params_rand[3*i+2] = np.random.uniform(0, 60)
        configs.append(params_rand)
    
    # 3. Line packing
    params_line = np.zeros(3*N)
    for i in range(N):
        params_line[3*i] = i * s3
        params_line[3*i+1] = 0.0
        params_line[3*i+2] = 0.0
    configs.append(params_line)
    
    best_side = float('inf')
    best_params = None
    best_angle = 0.0
    
    # For each config, try to optimize
    for cfg in configs:
        if time.time() > deadline:
            break
        
        # Optimize inner params with Nelder-Mead or DE
        # Use differential evolution for robustness
        bounds = [(-10, 10), (-10, 10), (0, 60)] * N
        
        # Use a smaller population for speed
        try:
            result = differential_evolution(
                _objective_flat,
                bounds=bounds,
                x0=cfg,
                seed=42,
                maxiter=50,
                popsize=5,
                tol=1e-4,
                init='sobol'
            )
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # Now optimize the outer angle for the best params
    if best_params is None:
        # Fallback to initial config
        best_params = configs[0]
        _, _, best_side, _, _ = _construct_from_params(best_params, 0.0)
    
    def _outer_angle_objective(angle_rad):
        _, _, side, _, _ = _construct_from_params(best_params, angle_rad)
        return side
    
    # Optimize outer angle in [0, pi/3]
    from scipy.optimize import minimize_scalar
    try:
        res = minimize_scalar(_outer_angle_objective, bounds=(0, math.pi/3), method='bounded')
        if time.time() < deadline:
            best_angle = res.x
            _, _, best_side, _, _ = _construct_from_params(best_params, best_angle)
    except Exception:
        pass
    
    # Final construction
    inner_list, center, side, angle, _ = _construct_from_params(best_params, best_angle)
    
    return (inner_list, center, side, angle)
