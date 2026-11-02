import math
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _min_enclosing_hexagon(points):
    """
    Computes the minimum side length of a regular hexagon that encloses the given points.
    points: list of (x, y)
    Returns: (side_length, center_x, center_y, angle_deg)
    """
    if not points:
        return 0, 0, 0, 0
    
    # If all points are the same, return 0
    if all(p == points[0] for p in points):
        return 0, points[0][0], points[0][1], 0

    # Compute convex hull
    pts = np.array(points)
    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
    except Exception:
        # If hull fails (e.g., collinear), use all points
        hull_pts = pts

    # We need to find the minimum enclosing regular hexagon.
    # The optimal hexagon's sides are parallel to the sides of the convex hull's "supporting" structure.
    # A regular hexagon has 6 sides. The normals are at angles alpha, alpha+60, ..., alpha+300.
    # For a given angle alpha, the width of the set in direction alpha is w(alpha).
    # The side length s of the enclosing hexagon must satisfy s >= w(alpha) / sqrt(3) for all 6 normals?
    # No. The distance from center to side is apothem a = s * sqrt(3) / 2.
    # So s = 2 * a / sqrt(3).
    # For a fixed orientation, the center must be placed such that the projections of all points
    # onto the 6 normal directions lie within [-a, a].
    # This is a 2D linear programming problem for the center (cx, cy) for a fixed alpha.
    # However, we can iterate over alpha.
    # It is known that the minimum enclosing regular polygon occurs when one of its sides is flush with an edge of the convex hull.
    # So we only need to check angles alpha that are perpendicular to the edges of the convex hull.
    
    # Edges of convex hull
    n_hull = len(hull_pts)
    if n_hull < 3:
        # Degenerate
        return 0, 0, 0, 0

    best_s = float('inf')
    best_cx = 0
    best_cy = 0
    best_angle = 0

    # Candidate angles: perpendicular to each hull edge
    # Also include 0, 30, 60, 90, 120, 150 just in case?
    # The normals of the hexagon must align with the "critical" directions.
    # The width function w(theta) is piecewise cosine. The maximum of the required apothem
    # A(alpha) = max_{k=0..5} ( w(alpha + k*pi/3) / sqrt(3) ) ? No.
    
    # Let's use the property:
    # For a fixed rotation theta of the hexagon (angle of one normal),
    # the required apothem is A = max_{i=0..5} ( max( dot(p, n_i) - min( dot(p, n_i) ) ) / 2 )?
    # No, the center is variable.
    
    # Let's use a simpler robust method:
    # Iterate over candidate angles theta (0 to 180, step 1 degree? No, too slow maybe, but N is small).
    # Actually, the optimal theta is such that one side of the hexagon is parallel to an edge of the convex hull.
    # So theta = edge_angle + 90.
    
    candidate_angles = []
    for i in range(n_hull):
        p1 = hull_pts[i]
        p2 = hull_pts[(i+1) % n_hull]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx*dx + dy*dy > 1e-12:
            edge_angle = math.atan2(dy, dx)
            # Normal angle
            normal_angle = edge_angle + math.pi / 2
            candidate_angles.append(normal_angle)
            
    # Also add 0, pi/6, pi/3, pi/2, 2pi/3, 5pi/6 to be safe?
    # The set of critical angles for enclosing a convex polygon with a regular hexagon
    # is indeed the normals to the hull edges.
    
    if not candidate_angles:
        return 0, 0, 0, 0

    # For each candidate angle, find the optimal center and side length.
    # For a fixed set of normals n_0...n_5,
    # We want to minimize s such that there exists (cx, cy) with
    # max_p ( n_k . (p - c) ) <= s * sqrt(3) / 2 for all k.
    # Let m_k = max_p ( n_k . p ) and M_k = min_p ( n_k . p ).
    # The width in direction n_k is W_k = m_k - M_k.
    # The center projection c_k = n_k . c must be in [M_k + A, m_k - A] where A is apothem.
    # This interval must be non-empty: M_k + A <= m_k - A => 2A <= W_k => A <= W_k / 2.
    # So A must be <= min_k (W_k / 2).
    # Wait, this logic is for a fixed center. But we can choose the center.
    # Actually, for a fixed orientation, the minimal enclosing hexagon is determined by the minimal A
    # such that the intersection of the strips defined by the normals is non-empty?
    # No, the hexagon is the intersection of 6 strips.
    # The points must be inside the hexagon.
    # So for each normal n_k, the projection of all points onto n_k must lie within [c_k - A, c_k + A].
    # This implies 2A >= W_k for all k. So A >= max_k (W_k / 2).
    # Is it possible to choose c_k such that this holds?
    # c_k is not independent; c_k = n_k . c.
    # So we need to find c such that for all k:
    # M_k - n_k . c <= A and n_k . c - m_k <= A.
    # This is a 2D LP problem: minimize A subject to these constraints.
    # However, for a regular hexagon, the normals are fixed relative to each other.
    
    # Simpler approach:
    # The minimal enclosing regular hexagon for a convex polygon is found by rotating the hexagon.
    # We can just iterate over the candidate angles (normals to hull edges).
    # For each angle, we can find the minimal A by solving the 2D LP or using a geometric property.
    # Actually, there is a known result: the minimal enclosing regular polygon of a convex polygon
    # has a side parallel to an edge of the polygon.
    
    # Let's implement the check for a given angle.
    # Given angle theta (for normal 0), normals are at theta + k*pi/3.
    # We need to find c=(cx, cy) and A such that:
    # For all k in 0..5, for all p in hull_pts:
    # | n_k . (p - c) | <= A
    
    # This is equivalent to:
    # For all k, max_p (n_k . p) - n_k . c <= A
    # and n_k . c - min_p (n_k . p) <= A
    
    # Let u_k = max_p (n_k . p) and l_k = min_p (n_k . p).
    # Constraints:
    # n_k . c >= u_k - A
    # n_k . c <= l_k + A
    
    # We want to minimize A.
    # This is a linear program in variables cx, cy, A.
    # Variables: x, y, A
    # Constraints:
    # For each k:
    # n_kx * x + n_ky * y + A >= u_k
    # n_kx * x + n_ky * y - A <= l_k  => -n_kx * x - n_ky * y + A >= -l_k
    
    # We can use scipy.optimize.linprog? Or just iterate over A?
    # Since N is small, we can use a simpler method.
    # Note that A must be at least max_k ( (u_k - l_k) / 2 ). Let A_min0 = max_k ( (u_k - l_k) / 2 ).
    # If we set A = A_min0, is there a feasible c?
    # The constraints become:
    # n_k . c >= u_k - A_min0
    # n_k . c <= l_k + A_min0
    # Note that u_k - A_min0 <= l_k + A_min0 is guaranteed by A_min0 >= (u_k - l_k)/2.
    # So the intervals for n_k . c are non-empty.
    # We need to find c such that n_k . c lies in [u_k - A, l_k + A] for all k.
    # This is a 2D feasibility problem.
    
    # Let's use a grid search for c? No, too slow.
    # Let's use the fact that the optimal c is often the centroid? No.
    # Let's use scipy.optimize.minimize_scalar or similar?
    
    # Alternative:
    # The minimal A is the minimal value such that the intersection of the 6 strips is non-empty.
    # The strips are defined by pairs of parallel lines.
    # The intersection of 6 strips in 2D is a convex polygon (possibly empty).
    # We want the minimal A such that this polygon is non-empty.
    # This happens when the 6 lines form a closed loop.
    
    # Actually, for regular hexagons, there is a simpler formula.
    # The width of the hexagon in any direction is 2A.
    # The width of the set in direction n_k is W_k.
    # So we need 2A >= W_k for all k.
    # So A >= max_k (W_k / 2).
    # Is it always possible to center the hexagon to achieve this?
    # Yes, if the intersection of the strips of width 2A centered at c is non-empty.
    # The condition for non-emptiness of the intersection of strips is that the sum of the widths in any direction is sufficient?
    # Actually, for 3 pairs of parallel lines (which is what a hexagon is), the intersection is non-empty
    # if and only if the "triangle" formed by the lines is non-empty?
    
    # Let's just use a numerical solver for the 2D LP for each candidate angle.
    # Or even simpler: since we only have 6 constraints on 2 variables (for fixed A),
    # we can check feasibility.
    # But we want to minimize A.
    
    # Let's try a different approach:
    # The optimal enclosing hexagon for a convex polygon has at least 3 points of contact.
    # We can iterate over pairs of hull edges?
    
    # Given the time constraint, let's use a robust but slightly slower method:
    # For each candidate angle, find the minimal A by binary search on A?
    # No, A is determined by the width.
    # Wait, if A = max_k (W_k / 2), is it always feasible?
    # Counter-example: Two points far apart in one direction, close in others.
    # The width in the far direction is large.
    # The strips for other directions might not intersect at the required center.
    
    # Let's use scipy.optimize.minimize to find c for a given A?
    # No, let's use the property that the optimal hexagon is "tight".
    
    # Actually, for the specific case of enclosing a set of points with a regular hexagon,
    # the minimum side length s is given by:
    # s = (2 / sqrt(3)) * max_{theta} ( W(theta) / 2 ) ? No.
    
    # Let's stick to the candidate angles and solve the 2D LP using a simple method.
    # We can use the fact that the feasible region for c is a convex polygon.
    # We can compute the intersection of the 6 strips.
    
    def _intersect_strips(normals, u, l, A):
        # normals: list of (nx, ny)
        # u, l: max and min projections
        # Returns (cx, cy) if feasible, else None
        # Constraints:
        # nx * cx + ny * cy >= u_k - A
        # nx * cx + ny * cy <= l_k + A
        
        # We can use scipy.spatial.HalfspaceIntersection?
        # Or just use a simple linear programming solver.
        
        # Let's use scipy.optimize.linprog
        # Minimize 0
        # Subject to:
        # A_ub * x <= b_ub
        # A_eq * x = b_eq
        
        # Variables: cx, cy
        # Constraints:
        # nx * cx + ny * cy >= u_k - A  => -nx * cx - ny * cy <= -(u_k - A)
        # nx * cx + ny * cy <= l_k + A
        
        A_ub = []
        b_ub = []
        
        for k in range(6):
            nx, ny = normals[k]
            # -nx * cx - ny * cy <= -(u_k - A)
            A_ub.append([-nx, -ny])
            b_ub.append(-(u[k] - A))
            # nx * cx + ny * cy <= l_k + A
            A_ub.append([nx, ny])
            b_ub.append(l[k] + A)
            
        try:
            res = scipy.optimize.linprog(c=[0, 0], A_ub=A_ub, b_ub=b_ub, method='highs')
            if res.success:
                return res.x
        except Exception:
            pass
        return None

    import scipy.optimize
    
    for angle in candidate_angles:
        normals = []
        for k in range(6):
            theta = angle + k * math.pi / 3
            normals.append((math.cos(theta), math.sin(theta)))
            
        u = []
        l = []
        for nx, ny in normals:
            projs = [nx * p[0] + ny * p[1] for p in hull_pts]
            u.append(max(projs))
            l.append(min(projs))
            
        # Binary search for A?
        # Lower bound: max((u_k - l_k)/2)
        A_low = max((u[k] - l[k]) / 2 for k in range(6))
        # Upper bound: something large
        A_high = 100.0
        
        # Actually, we can just try A = A_low. If feasible, it's optimal.
        # If not, we need to increase A.
        # But for a regular hexagon, is A_low always feasible?
        # No.
        
        # Let's use a simple iteration.
        # Since the problem is small, we can use a grid search for A?
        # Or use the fact that the optimal A is such that the intersection of strips is a single point?
        
        # Let's try A = A_low.
        c = _intersect_strips(normals, u, l, A_low)
        if c is not None:
            A = A_low
        else:
            # Increase A
            # We can use a binary search between A_low and A_high
            lo = A_low
            hi = A_high
            for _ in range(50):
                mid = (lo + hi) / 2
                c = _intersect_strips(normals, u, l, mid)
                if c is not None:
                    hi = mid
                else:
                    lo = mid
            A = hi
            c = _intersect_strips(normals, u, l, A)
            if c is None:
                continue # Should not happen
        
        s = 2 * A / math.sqrt(3)
        if s < best_s:
            best_s = s
            best_cx = c[0]
            best_cy = c[1]
            best_angle = math.degrees(angle)

    return best_s, best_cx, best_cy, best_angle

def _generate_configurations():
    """
    Generates candidate configurations of 11 unit hexagons on a hexagonal lattice.
    Returns a list of lists of (cx, cy, angle).
    """
    configs = []
    
    # Lattice vectors for unit hexagons (side 1)
    # Distance between centers of touching hexagons is sqrt(3)? No.
    # For regular hexagons of side 1, the distance between centers of touching hexagons is sqrt(3).
    # Wait, apothem is sqrt(3)/2. Distance = 2 * apothem = sqrt(3).
    # Lattice basis:
    # e1 = (sqrt(3), 0)
    # e2 = (sqrt(3)/2, 3/2)
    
    sqrt3 = math.sqrt(3)
    e1 = (sqrt3, 0)
    e2 = (sqrt3/2, 1.5)
    
    def lattice_to_cart(i, j):
        x = i * e1[0] + j * e2[0]
        y = i * e1[1] + j * e2[1]
        return (x, y)
    
    # Configuration 1: Flower (1 center, 6 ring, 4 more)
    # This is not a valid lattice tiling if we just pick 11.
    # We need disjoint hexagons.
    # A valid packing is a subset of the hexagonal lattice.
    # Let's generate all connected subsets of size 11 from a large lattice patch.
    # This is combinatorially expensive.
    
    # Instead, let's use known optimal packings for N=11.
    # For N=11, the optimal packing is likely a "flower" with some extensions.
    
    # Let's define a few specific topologies.
    
    # Topology 1: Center + 6 neighbors + 4 from second ring.
    # Center: (0,0)
    # Ring 1: 6 neighbors.
    # Ring 2: 12 neighbors. Pick 4.
    
    # Let's just generate random connected subsets of size 11 from a 5x5 lattice patch.
    # And also some structured ones.
    
    # Structured 1: Line of 11? No, too long.
    # Structured 2: 3x3 block + 2?
    
    # Let's use a simple heuristic:
    # Start with center (0,0).
    # Add neighbors greedily to keep the cluster compact.
    
    # We will generate multiple random compact clusters.
    
    import random
    random.seed(42)
    
    for _ in range(100):
        # Start with a center
        centers = [(0.0, 0.0)]
        # Available neighbors
        # Directions: 0, 60, 120, 180, 240, 300
        # In lattice coords:
        # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        
        lattice_centers = {(0,0)}
        
        for _ in range(N - 1):
            # Find all potential neighbors
            candidates = set()
            for (i, j) in lattice_centers:
                for di, dj in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                    ni, nj = i + di, j + dj
                    if (ni, nj) not in lattice_centers:
                        candidates.add((ni, nj))
            
            if not candidates:
                break
                
            # Pick a random candidate
            # To keep it compact, we could pick the one that minimizes the distance to the centroid,
            # but random is fine for diversity.
            choice = random.choice(list(candidates))
            lattice_centers.add(choice)
            centers.append(lattice_to_cart(*choice))
            
            if len(centers) == N:
                break
        
        if len(centers) == N:
            # Check if it's a valid packing (no overlaps).
            # Since it's a lattice subset, they are disjoint.
            # Check if it's connected? Yes, by construction.
            configs.append(centers)
            
    # Remove duplicates
    unique_configs = []
    seen = set()
    for c in configs:
        # Round coordinates for hashing
        key = tuple((round(x, 5), round(y, 5)) for x, y in c)
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
            
    return unique_configs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    configs = _generate_configurations()
    
    best_score = float('inf')
    best_inner = None
    best_outer = None
    
    for centers in configs:
        if time.time() > deadline:
            break
            
        # All inner hexagons have angle 0 (aligned with lattice)
        inner = []
        for cx, cy in centers:
            inner.append((cx, cy, 0.0))
            
        # Get vertices of all inner hexagons
        points = []
        for cx, cy, a in inner:
            verts = _hex_vertices(cx, cy, 1.0, a)
            points.extend(verts)
            
        # Find minimum enclosing hexagon
        s, ocx, ocy, oang = _min_enclosing_hexagon(points)
        
        if s < best_score:
            best_score = s
            best_inner = inner
            best_outer = (ocx, ocy, s, oang)
            
    if best_inner is None:
        # Fallback: simple lattice
        centers = _generate_configurations()[0]
        inner = [(cx, cy, 0.0) for cx, cy in centers]
        points = []
        for cx, cy, a in inner:
            verts = _hex_vertices(cx, cy, 1.0, a)
            points.extend(verts)
        s, ocx, ocy, oang = _min_enclosing_hexagon(points)
        best_inner = inner
        best_outer = (ocx, ocy, s, oang)
        
    return best_inner, best_outer[:2], best_outer[2], best_outer[3]

if __name__ == "__main__":
    r = construct()
    print(r)
