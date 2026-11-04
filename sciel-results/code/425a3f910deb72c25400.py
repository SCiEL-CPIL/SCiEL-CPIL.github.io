import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    """Get vertices of a regular hexagon with given center, side length, and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and rotation."""
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle = np.radians(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    # In local frame, hexagon vertices are at angles 0, 60, 120, 180, 240, 300
    # The hexagon can be defined by |lx| <= side and |ly| <= side*sqrt(3)/2 for certain orientations
    # Actually, let's use the half-plane method
    # For a regular hexagon with "pointy" top (vertex at top), the constraints are:
    # |x| <= side, and |y| <= side * sqrt(3) for the flat-top orientation
    # Let me use a more robust approach: check against all 6 edges
    
    verts = hexagon_vertices(0, 0, side, angle_deg)
    # Transform point to local coordinates
    local_p = np.array([lx, ly])
    
    # Check if point is inside using cross products
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        edge = v2 - v1
        to_point = local_p - v1
        cross = edge[0] * to_point[1] - edge[1] * to_point[0]
        # For CCW polygon, all crosses should be >= 0 (or <= 0 depending on winding)
        if cross < -1e-10:
            return False
    return True

def min_hexagon_side_for_points(points, outer_angle_deg=0):
    """Find minimum side length of regular hexagon (with given rotation) that contains all points."""
    # The minimum enclosing hexagon with fixed rotation can be found by
    # computing the maximum extent in the 6 directions normal to the sides
    angle = np.radians(outer_angle_deg)
    
    # The 6 outward normals for a hexagon with vertex at angle 0
    # For a regular hexagon, the sides are perpendicular to directions at 30, 90, 150, 210, 270, 330 degrees
    # Actually, let me think more carefully.
    
    # A regular hexagon with "flat" top (side horizontal at top) has vertices at angles 30, 90, 150, 210, 270, 330
    # The outward normals to the sides are at angles 0, 60, 120, 180, 240, 300
    
    # For a hexagon with rotation angle, the outward normals are at angle + 0, angle + 60, etc.
    # The constraint for each side: dot(point - center, normal) <= side_length
    
    # So we need: for all points p, for all 6 normals n_i:
    # dot(p - c, n_i) <= s
    # where s is the side length (apothem = s * sqrt(3)/2, but let me be careful)
    
    # Actually, for a regular hexagon with side length s:
    # - The apothem (distance from center to side) is s * sqrt(3) / 2
    # - The circumradius (distance from center to vertex) is s
    
    # So the constraint is: dot(p - c, n_i) <= s * sqrt(3) / 2 for each normal direction
    
    # To minimize s, we center the hexagon optimally.
    # For fixed center c, s >= max over all points and normals of dot(p - c, n_i) / (sqrt(3)/2)
    
    # To find optimal c, we can note that this is a convex optimization.
    # Let's use the fact that the optimal center minimizes the maximum extent.
    
    # For simplicity, let's try centering at the centroid of the points first, then refine.
    centroid = np.mean(points, axis=0)
    
    # Compute the 6 normal directions
    normals = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    # For a given center c, the required side length is:
    # s = max over i, p of dot(p - c, n_i) / (sqrt(3)/2)
    # We want to minimize this over c.
    
    # This is a linear programming problem: minimize s subject to
    # dot(p - c, n_i) <= s * sqrt(3)/2 for all i, p
    # dot(c - p, n_i) <= s * sqrt(3)/2 for all i, p (symmetric constraint)
    
    # Let me use scipy's linear programming
    from scipy.optimize import linprog
    
    # Variables: c_x, c_y, s
    # Constraints: for each point p and normal n_i:
    # n_i_x * (p_x - c_x) + n_i_y * (p_y - c_y) <= s * sqrt(3)/2
    # => -n_i_x * c_x - n_i_y * c_y - sqrt(3)/2 * s <= -n_i_x * p_x - n_i_y * p_y
    
    # And symmetric:
    # n_i_x * (c_x - p_x) + n_i_y * (c_y - p_y) <= s * sqrt(3)/2
    # => n_i_x * c_x + n_i_y * c_y - sqrt(3)/2 * s <= n_i_x * p_x + n_i_y * p_y
    
    A = []
    b = []
    for p in points:
        for n in normals:
            # First constraint: dot(p - c, n) <= s * sqrt(3)/2
            A.append([-n[0], -n[1], -np.sqrt(3)/2])
            b.append(-n[0]*p[0] - n[1]*p[1])
            
            # Second constraint: dot(c - p, n) <= s * sqrt(3)/2
            A.append([n[0], n[1], -np.sqrt(3)/2])
            b.append(n[0]*p[0] + n[1]*p[1])
    
    # Objective: minimize s
    c_obj = [0, 0, 1]
    
    # Bounds: c_x, c_y unbounded, s >= 0
    bounds = [(None, None), (None, None), (0, None)]
    
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.success:
        return res.x[2]
    else:
        # Fallback: use centroid
        s = 0
        for p in points:
            for n in normals:
                val = np.dot(p - centroid, n) / (np.sqrt(3)/2)
                s = max(s, val)
        return s

def total_vertices(inner_positions):
    """Get all vertices of all inner hexagons."""
    all_verts = []
    for x, y, angle in inner_positions:
        verts = hexagon_vertices(x, y, 1.0, angle)
        all_verts.append(verts)
    return all_verts

def objective(params):
    """Objective function: minimize outer hexagon side length."""
    # params: 12 * 3 = 36 values (x, y, angle for each hexagon)
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        angle = params[3*i+2]
        inner.append([x, y, angle])
    
    # Get all vertices
    all_verts_list = total_vertices(inner)
    all_verts = np.vstack(all_verts_list)
    
    # Find minimum enclosing hexagon side length
    # Try a few outer rotations
    best_side = float('inf')
    for outer_angle in [0, 30, 60, 90]:
        side = min_hexagon_side_for_points(all_verts, outer_angle)
        best_side = min(best_side, side)
    
    return best_side

def check_validity(inner):
    """Check if hexagons are disjoint and we can compute a valid outer hexagon."""
    # Check pairwise non-overlap
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            
            # Quick distance check
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            # Minimum distance between centers for non-overlapping unit hexagons
            # The inradius is sqrt(3)/2, so if centers are further than 2*inradius = sqrt(3) ~ 1.732,
            # they might still overlap depending on orientation
            # For a robust check, we'd need to check all vertex-edge pairs
            # For now, use a conservative threshold
            if dist < 1.5:  # Very close, likely overlapping
                # More detailed check
                vi = hexagon_vertices(xi, yi, 1.0, ai)
                vj = hexagon_vertices(xj, yj, 1.0, aj)
                # Check if any vertex of i is inside j
                for v in vi:
                    if point_in_hexagon(v[0], v[1], xj, yj, 1.0, aj):
                        return False
                # Check if any vertex of j is inside i
                for v in vj:
                    if point_in_hexagon(v[0], v[1], xi, yi, 1.0, ai):
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial guess: hexagonal arrangement
    # Place hexagons in a honeycomb pattern
    # Center hexagon at origin, 6 around it, then 5 more
    
    # Hexagonal lattice spacing: for unit hexagons, the distance between centers
    # in a honeycomb is 2 * inradius = sqrt(3) for adjacent hexagons
    # Actually, for regular hexagons of side 1, the center-to-center distance
    # for edge-sharing is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3)
    
    # Let's create a hexagonal arrangement
    # Row 0: 1 hexagon at center
    # Row 1: 6 hexagons around
    # Row 2: 5 more hexagons
    
    # Actually, let's use a simpler initial guess and let optimization do the work
    # Start with hexagons in a rough hexagonal pattern
    
    sqrt3 = np.sqrt(3)
    
    # Initial positions: hexagonal lattice
    # For a hexagonal lattice with spacing d = sqrt(3) (edge-sharing)
    # Positions in axial coordinates (q, r) -> cartesian:
    # x = d * (q + r/2), y = d * r * sqrt(3)/2
    
    positions = []
    # Center
    positions.append([0.0, 0.0, 0.0])
    
    # 6 neighbors
    for i in range(6):
        angle = i * np.pi / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        positions.append([x, y, 0.0])
    
    # 5 more: next ring, pick 5 of the 12
    for i in range(5):
        angle = i * 2 * np.pi / 5  # Spread them out
        x = 2 * sqrt3 * np.cos(angle)
        y = 2 * sqrt3 * np.sin(angle)
        positions.append([x, y, 0.0])
    
    # Convert to params
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = positions[i][0]
        params[3*i+1] = positions[i][1]
        params[3*i+2] = positions[i][2]
    
    # Optimize
    best_params = params.copy()
    best_obj = objective(params)
    
    # Use Nelder-Mead for local optimization (derivative-free)
    while time.time() < deadline:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
            
            # Check if we're still improving
            if best_obj < 4.0:  # Getting close to good solution
                break
        except Exception:
            break
        
        # Perturb for restart
        if time.time() > deadline - 20:
            break
        best_params = best_params + np.random.normal(0, 0.5, size=3*N)
    
    # Extract final positions
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        angle = best_params[3*i+2]
        inner.append([x, y, angle])
    
    # Compute outer hexagon
    all_verts_list = total_vertices(inner)
    all_verts = np.vstack(all_verts_list)
    
    best_side = float('inf')
    best_outer_angle = 0
    for outer_angle in np.arange(0, 180, 15):
        side = min_hexagon_side_for_points(all_verts, outer_angle)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle
    
    # Center of outer hexagon
    # Find optimal center
    angle = np.radians(best_outer_angle)
    normals = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    # The optimal center is the centroid of the "extreme" points
    # For simplicity, use the centroid of all vertices
    center = np.mean(all_verts, axis=0)
    
    return inner, (center[0], center[1]), best_side, best_outer_angle

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
