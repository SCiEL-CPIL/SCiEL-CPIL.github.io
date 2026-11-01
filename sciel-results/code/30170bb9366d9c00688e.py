"""Place disjoint unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon (flat-top or pointy depending on angle)."""
    a = math.radians(angle_deg)
    # Standard regular hexagon with side length s, vertices at angles 0,60,...,300
    # For a "flat-top" hexagon, vertices are at 0,60,120,180,240,300 degrees
    # The apothem (distance from center to side midpoint) = s * sqrt(3)/2
    # The circumradius (center to vertex) = s
    r = side
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return verts

def _hex_apothem(side):
    return side * math.sqrt(3) / 2.0

def _point_in_hex(px, py, cx, cy, side, angle_deg, eps=1e-9):
    """Check if point (px,py) is inside the regular hexagon centered at (cx,cy)."""
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate point into hexagon's local frame
    ca, sa = math.cos(-a), math.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # For a regular hexagon with circumradius r=side, the six sides are at
    # distance apothem = side*sqrt(3)/2 from center, in directions 30,90,150,210,270,330 (for flat-top with a=0)
    # Actually for vertices at angles 0,60,...,300, the side normals are at 30,90,150,210,270,330
    apothem = _hex_apothem(side)
    for i in range(6):
        theta = math.pi / 6.0 + i * math.pi / 3.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        if lx * nx + ly * ny > apothem + eps:
            return False
    return True

def _hexagon_intersection_area(v1, v2):
    """
    Compute the area of intersection of two convex polygons (hexagons).
    Uses Sutherland-Hodgman or clipping approach.
    Returns 0 if no intersection.
    """
    # Use a simple polygon clipping algorithm
    def clip_polygon(subject, clip):
        """Clip subject polygon by clip polygon (convex)."""
        output = list(subject)
        for i in range(len(clip)):
            if not output:
                return []
            input_list = output
            output = []
            S = input_list[0]
            for j in range(len(input_list)):
                E = input_list[j]
                S_idx = i
                E_idx = (i + 1) % len(clip)
                # Determine if S and E are inside or outside the edge
                def inside(p, a, b):
                    # Check if p is to the left of edge a->b (for CCW polygon)
                    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= -1e-9
                
                S_inside = inside(S, clip[i], clip[(i+1)%len(clip)])
                E_inside = inside(E, clip[i], clip[(i+1)%len(clip)])
                
                if S_inside and E_inside:
                    output.append(E)
                elif S_inside and not E_inside:
                    # Leaving: add intersection
                    inter = _line_intersect(S, E, clip[i], clip[(i+1)%len(clip)])
                    if inter:
                        output.append(inter)
                elif not S_inside and E_inside:
                    # Entering: add intersection then E
                    inter = _line_intersect(S, E, clip[i], clip[(i+1)%len(clip)])
                    if inter:
                        output.append(inter)
                    output.append(E)
                S = E
        return output
    
    def _line_intersect(p1, p2, p3, p4):
        """Intersection of line p1-p2 and line p3-p4."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-12:
            return None
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        if -1e-9 <= t <= 1+1e-9:
            return (x1 + t*(x2-x1), y1 + t*(y2-y1))
        return None
    
    def polygon_area(poly):
        if len(poly) < 3:
            return 0.0
        area = 0.0
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1)%n]
            area += x1*y2 - x2*y1
        return abs(area) / 2.0
    
    # Ensure both polygons are CCW
    def ensure_ccw(poly):
        area = 0.0
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1)%n]
            area += x1*y2 - x2*y1
        if area < 0:
            poly = poly[::-1]
        return poly
    
    v1 = ensure_ccw(list(v1))
    v2 = ensure_ccw(list(v2))
    
    # Quick bounding box check
    minx1 = min(p[0] for p in v1); maxx1 = max(p[0] for p in v1)
    miny1 = min(p[1] for p in v1); maxy1 = max(p[1] for p in v1)
    minx2 = min(p[0] for p in v2); maxx2 = max(p[0] for p in v2)
    miny2 = min(p[1] for p in v2); maxy2 = max(p[1] for p in v2)
    
    if maxx1 < minx2 - 1e-9 or maxx2 < minx1 - 1e-9 or maxy1 < miny2 - 1e-9 or maxy2 < miny1 - 1e-9:
        return 0.0
    
    clipped = clip_polygon(v1, v2)
    if not clipped:
        return 0.0
    return polygon_area(clipped)

def construct():
    """
    Place N unit hexagons inside a regular hexagon, minimizing the outer side length.
    Returns: (inner, center, side, angle)
    inner: list of [x, y, angle_degrees] for each unit hexagon
    center: (x, y) of outer hexagon
    side: side length of outer hexagon
    angle: rotation of outer hexagon in degrees
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Strategy: Use a hexagonal lattice arrangement
    # For N hexagons, find the optimal hexagonal packing arrangement
    # The unit hexagons have side length 1, so they tile the plane.
    # In a hexagonal lattice, adjacent hexagons are at distance 2*sqrt(3)/2 = sqrt(3) apart
    # (center-to-center distance for edge-sharing hexagons of side 1)
    
    # Actually, for regular hexagons of side s, the center-to-center distance
    # for edge-adjacent hexagons is 2 * apothem = 2 * s * sqrt(3)/2 = s * sqrt(3)
    # For s=1, this is sqrt(3) ≈ 1.732
    
    # Let's try a hexagonal cluster arrangement
    # For N=11, a good arrangement is a central hexagon surrounded by 6,
    # plus 4 more in the next ring (or some other configuration)
    
    # Let's use a systematic approach: try different arrangements and compute
    # the minimum enclosing hexagon
    
    best_score = float('inf')
    best_result = None
    
    # Generate candidate arrangements
    arrangements = []
    
    # Arrangement 1: Linear row
    # N hexagons in a row, center-to-center distance sqrt(3)
    # Total width ≈ (N-1)*sqrt(3) + 2*apothem = (N-1)*sqrt(3) + sqrt(3) = N*sqrt(3)
    # But we need to enclose in a hexagon, so this is not optimal for large N
    
    # Arrangement 2: Hexagonal cluster
    # For N=11, we can use a 3-4-3 or similar arrangement
    
    # Let's use a grid-based approach with hexagonal coordinates
    # Hexagonal lattice basis vectors:
    # a1 = (sqrt(3), 0)  (horizontal)
    # a2 = (sqrt(3)/2, 3/2)  (60 degrees)
    # Wait, for hexagons of side 1:
    # The distance between centers of edge-adjacent hexagons is sqrt(3)
    # Direction 1: (sqrt(3), 0)
    # Direction 2: (sqrt(3)/2, 3/2)  -- this has length sqrt(3/4 + 9/4) = sqrt(3)
    # Actually: sqrt((sqrt(3)/2)^2 + (3/2)^2) = sqrt(3/4 + 9/4) = sqrt(3)  ✓
    
    a1 = (math.sqrt(3), 0)
    a2 = (math.sqrt(3)/2, 1.5)
    
    # Generate hexagonal cluster positions
    # For a hexagonal cluster with "radius" r (in terms of number of hexagons from center),
    # the total count is 1 + 3*r*(r+1)
    # r=0: 1, r=1: 7, r=2: 19, r=3: 37
    
    # For N=11, we need a partial cluster
    # Let's generate positions in a hexagonal grid and pick N of them
    
    def hex_to_xy(q, r):
        """Convert axial hex coordinates to xy."""
        x = a1[0]*q + a2[0]*r
        y = a1[1]*q + a2[1]*r
        return (x, y)
    
    # Generate all positions in a large enough hexagonal grid
    max_r = 10
    positions = []
    for q in range(-max_r, max_r+1):
        for r in range(-max_r, max_r+1):
            # Axial coordinate constraint: |q+r| <= max_r for a hexagon of radius max_r
            if abs(q + r) <= max_r:
                positions.append((q, r))
    
    # For each subset of N positions, compute the enclosing hexagon
    # This is too expensive for N=11 from many candidates
    
    # Better approach: Try specific structured arrangements
    
    # Arrangement A: 3 rows of 4,3,4 (centered)
    # Arrangement B: 4 rows of 3,4,4,3 (centered)
    # Arrangement C: Linear with some offset
    
    # Let's try a more direct optimization approach
    # Use a simple local search: start with a hexagonal cluster and optimize positions
    
    # Initial guess: hexagonal cluster
    # For N=11, use positions:
    # Center: (0,0)
    # Ring 1 (6 hexagons): all 6 neighbors
    # Ring 2 (4 hexagons): pick 4 from the 12 available
    
    # Let's define the hexagonal cluster more carefully
    # Using axial coordinates (q, r):
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1)
    
    # For N=11: center + 6 in ring 1 + 4 in ring 2
    
    # Let's try all possible selections of 4 from ring 2 and find the best enclosing hexagon
    # But first, let's compute the enclosing hexagon for a given set of points
    
    def compute_enclosing_hexagon(points, side_range=None):
        """
        Find the minimum side length regular hexagon that encloses all given points.
        The hexagon can be translated and rotated.
        Returns (side_length, center_x, center_y, angle_deg)
        """
        if not points:
            return (0, 0, 0, 0)
        
        # For a regular hexagon with side length s and rotation angle theta,
        # centered at (cx, cy), a point (px, py) is inside if:
        # For each of the 6 sides, the dot product with the outward normal <= apothem
        # The 6 outward normals are at angles theta + 30 + k*60 for k=0..5
        # Apothem = s * sqrt(3)/2
        
        # We need to find s, cx, cy, theta such that all points are inside
        # and s is minimized
        
        # This is a 4D optimization problem. Let's use a grid search over theta
        # and then for each theta, solve for cx, cy, s
        
        # For a fixed theta, the constraint for each point (px, py) is:
        # For each of 6 normals n_k: (px - cx)*n_kx + (py - cy)*n_ky <= s * sqrt(3)/2
        # This is equivalent to:
        # cx * n_kx + cy * n_ky >= px * n_kx + py * n_ky - s * sqrt(3)/2
        # and
        # -cx * n_kx - cy * n_ky >= -px * n_kx - py * n_ky - s * sqrt(3)/2
        
        # Let u = (cx, cy). For each normal n_k, we have:
        # u · n_k >= (px * n_kx + py * n_ky) - s*apothem  for all points
        # -u · n_k >= -(px * n_kx + py * n_ky) - s*apothem  for all points
        
        # So: u · n_k >= max_p (px*n_kx + py*n_ky) - s*apothem
        #     -u · n_k >= max_p (-(px*n_kx + py*n_ky)) - s*apothem
        #     => u · n_k <= -max_p (-(px*n_kx + py*n_ky)) + s*apothem
        #     => u · n_k <= min_p (px*n_kx + py*n_ky) + s*apothem
        
        # So for each k:
        # max_p (p·n_k) - s*apothem <= u·n_k <= min_p (p·n_k) + s*apothem
        
        # For a feasible u to exist, we need these 6 constraints to be consistent.
        # The minimum s for which there exists a u satisfying all 6 constraints
        # can be found by checking if the intersection of the 6 strips is non-empty.
        
        # Actually, this is getting complex. Let's use a simpler approach:
        # For a fixed theta, the minimum enclosing hexagon has its center at
        # the midpoint of the "extreme" points in each of the 6 normal directions.
        
        # For a regular hexagon, the minimum side length s for a given orientation theta
        # is determined by the maximum over the 6 directions of:
        # (max_p (p·n_k) - min_p (p·n_k)) / 2 / apothem_ratio
        # where apothem_ratio = 1 (since apothem = s*sqrt(3)/2, so s = 2*half_range/sqrt(3))
        
        # Wait, let me reconsider. For a hexagon with side s, the distance from center
        # to a side (apothem) is s*sqrt(3)/2. The distance from center to a vertex is s.
        
        # For a point to be inside, its projection onto each of the 6 side normals
        # must be within [-apothem, apothem] of the center's projection.
        
        # So for normal n_k, we need:
        # max_p (p·n_k) - u·n_k <= s*sqrt(3)/2
        # u·n_k - min_p (p·n_k) <= s*sqrt(3)/2
        
        # Adding these: max_p (p·n_k) - min_p (p·n_k) <= s*sqrt(3)
        # So s >= (max_p (p·n_k) - min_p (p·n_k)) / sqrt(3)
        
        # And for the center u, we need:
        # u·n_k >= max_p (p·n_k) - s*sqrt(3)/2
        # u·n_k <= min_p (p·n_k) + s*sqrt(3)/2
        
        # For the minimum s, the center is at the midpoint:
        # u·n_k = (max_p (p·n_k) + min_p (p·n_k)) / 2
        
        # So for a fixed theta, the minimum s is:
        # s_min(theta) = max_k [ (max_p (p·n_k) - min_p (p·n_k)) / sqrt(3) ]
        
        # And the center satisfies u·n_k = (max + min)/2 for all k.
        # This gives us 6 equations for 2 unknowns (cx, cy), which is overdetermined.
        # The center is the point that best satisfies these, or we can solve least squares.
        
        # Actually, for a regular hexagon, the 6 normals are not independent.
        # n_0 and n_3 are opposite, n_1 and n_4 are opposite, n_2 and n_5 are opposite.
        # So we only have 3 independent constraints on u.
        
        # Let's compute s_min(theta) and then find u.
        
        best_s = float('inf')
        best_cx = 0
        best_cy = 0
        best_theta = 0
        
        # Search over theta from 0 to 60 degrees (hexagon has 6-fold symmetry)
        # Use a fine grid
        num_theta = 360
        for i in range(num_theta):
            theta = i * 60.0 / num_theta
            theta_rad = math.radians(theta)
            
            # Compute the 6 normals
            normals = []
            for k in range(6):
                angle = theta_rad + math.pi/6 + k * math.pi/3
                normals.append((math.cos(angle), math.sin(angle)))
            
            # For each normal, compute max and min of p·n over all points
            max_projs = []
            min_projs = []
            for nx, ny in normals:
                projs = [px*nx + py*ny for px, py in points]
                max_projs.append(max(projs))
                min_projs.append(min(projs))
            
            # s_min for this theta
            s_candidate = 0
            for k in range(6):
                range_k = max_projs[k] - min_projs[k]
                s_k = range_k / math.sqrt(3)
                if s_k > s_candidate:
                    s_candidate = s_k
            
            if s_candidate < best_s:
                best_s = s_candidate
                best_theta = theta
                
                # Find center u = (cx, cy)
                # u·n_k = (max_projs[k] + min_projs[k])/2 for k=0,1,2 (independent)
                # Use least squares to solve for cx, cy
                A = []
                b = []
                for k in range(3):
                    nx, ny = normals[k]
                    A.append([nx, ny])
                    b.append((max_projs[k] + min_projs[k]) / 2)
                
                # Solve A @ [cx, cy] = b
                try:
                    A_arr = np.array(A)
                    b_arr = np.array(b)
                    sol = np.linalg.lstsq(A_arr, b_arr, rcond=None)[0]
                    best_cx = sol[0]
                    best_cy = sol[1]
                except:
                    best_cx = 0
                    best_cy = 0
        
        return (best_s, best_cx, best_cy, best_theta)
    
    # Now try various arrangements
    
    # Arrangement 1: Hexagonal cluster (center + ring 1 + 4 from ring 2)
    center_pos = hex_to_xy(0, 0)
    ring1 = [hex_to_xy(q, r) for q, r in [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]]
    ring2_all = [hex_to_xy(q, r) for q, r in [(2,0),(1,1),(0,2),(-1,2),(-2,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1),(2,1)]]
    
    # Try all combinations of 4 from ring 2
    from itertools import combinations
    
    for combo in combinations(range(12), 4):
        points = [center_pos] + ring1 + [ring2_all[i] for i in combo]
        s, cx, cy, theta = compute_enclosing_hexagon(points)
        if s < best_score:
            best_score = s
            best_result = (points, cx, cy, theta)
        
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        break
    
    # Arrangement 2: 3-4-3 grid (3 rows: 3, 4, 3 hexagons)
    # Row 0 (3 hexagons): y = 0, x = -sqrt(3)/2, 0, sqrt(3)/2  (wait, let me use the lattice)
    # Actually, let's use axial coordinates for a 3-4-3 arrangement
    # This is tricky. Let's just try a few more structured arrangements.
    
    # Arrangement 3: Linear row of 11
    points = [hex_to_xy(k, 0) for k in range(-5, 6)]
    s, cx, cy, theta = compute_enclosing_hexagon(points)
    if s < best_score:
        best_score = s
        best_result = (points, cx, cy, theta)
    
    # Arrangement 4: Two rows of 6 and 5
    points = [hex_to_xy(k, 0) for k in range(-3, 3)] + [hex_to_xy(k, 1) for k in range(-2, 3)]
    s, cx, cy, theta = compute_enclosing_hexagon(points)
    if s < best_score:
        best_score = s
        best_result = (points, cx, cy, theta)
    
    # Arrangement 5: 4-3-4 arrangement
    points = [hex_to_xy(k, 0) for k in range(-2, 2)] + [hex_to_xy(k, 1) for k in range(-1, 2)] + [hex_to_xy(k, 2) for k in range(-2, 2)]
    s, cx, cy, theta = compute_enclosing_hexagon(points)
    if s < best_score:
        best_score = s
        best_result = (points, cx, cy, theta)
    
    if best_result is None:
        # Fallback: linear row
        points = [hex_to_xy(k, 0) for k in range(-(N//2), N - N//2)]
        s, cx, cy, theta = compute_enclosing_hexagon(points)
        best_score = s
        best_result = (points, cx, cy, theta)
    
    points, cx, cy, theta = best_result
    
    # Build the inner hexagons list
    inner = []
    for px, py in points:
        inner.append([px, py, 0.0])
    
    return inner, (cx, cy), best_score, theta
# EVOLVE-BLOCK-END
