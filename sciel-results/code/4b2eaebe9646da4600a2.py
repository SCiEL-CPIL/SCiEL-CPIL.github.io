import math
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate a set of diverse initial configurations
    # We will use a mix of grid-like and random perturbations
    def generate_configurations():
        configs = []
        
        # Configuration 1: Simple hexagonal packing (center + 6 neighbors)
        # This is the most natural starting point
        base = [(0, 0)]
        for i in range(6):
            angle = i * math.pi / 3
            x = 2 * APOTHEM * math.cos(angle)
            y = 2 * APOTHEM * math.sin(angle)
            base.append((x, y))
        # Add 4 more in a ring further out? Or just use the 7 and let optimization move them.
        # Actually, for 11, a common pattern is a central hex, 6 around it, and 4 more.
        # Let's just generate random points within a reasonable radius to let DE find the structure.
        
        # Random initialization
        np.random.seed(42)
        for _ in range(5):
            pts = []
            for i in range(N):
                # Random position in a hexagon of side 2
                r = 2.0 * np.sqrt(np.random.random())
                theta = 2 * np.pi * np.random.random()
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                angle = 0.0
                pts.append((x, y, angle))
            configs.append(pts)
            
        # Structured initialization: Hex grid
        # Center (0,0)
        # Ring 1: distance 2*APOTHEM
        # Ring 2: distance 4*APOTHEM
        hex_pts = []
        hex_pts.append((0.0, 0.0, 0.0))
        for i in range(6):
            angle = i * math.pi / 3
            x = 2 * APOTHEM * math.cos(angle)
            y = 2 * APOTHEM * math.sin(angle)
            hex_pts.append((x, y, 0.0))
        # Need 4 more. Place them at distance 4*APOTHEM at angles between ring 1 vertices
        for i in range(4):
            angle = (i + 0.5) * math.pi / 3
            x = 4 * APOTHEM * math.cos(angle)
            y = 4 * APOTHEM * math.sin(angle)
            hex_pts.append((x, y, 0.0))
        configs.append(hex_pts)
        
        # Another structured one: Linear-ish
        line_pts = []
        # 5 in a row
        for i in range(5):
            line_pts.append((i * 2 * APOTHEM, 0.0, 0.0))
        # 3 above
        for i in range(3):
            line_pts.append((i * 2 * APOTHEM + APOTHEM, 1.5, 0.0))
        # 3 below
        for i in range(3):
            line_pts.append((i * 2 * APOTHEM + APOTHEM, -1.5, 0.0))
        configs.append(line_pts)
        
        return configs

    def get_vertices(x, y, angle_deg):
        """
        Get the 6 vertices of a unit regular hexagon centered at (x,y) with rotation angle_deg.
        """
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            vx = x + math.cos(theta)
            vy = y + math.sin(theta)
            verts.append((vx, vy))
        return verts

    def is_convex_polygon(vertices):
        """
        Check if a polygon is convex. Vertices should be ordered.
        """
        n = len(vertices)
        if n < 3:
            return False
        sign = 0
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            p3 = vertices[(i + 2) % n]
            cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
            if cross != 0:
                if sign == 0:
                    sign = 1 if cross > 0 else -1
                elif (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                    return False
        return True

    def get_convex_hull(points):
        """
        Simple convex hull for a set of 2D points. Returns list of (x,y) in CCW order.
        Uses monotone chain.
        """
        points = sorted(set(points))
        if len(points) <= 1:
            return points
        
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
            
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
            
        return lower[:-1] + upper[:-1]

    def point_in_polygon(point, polygon):
        """
        Check if a point is inside a convex polygon.
        """
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            # Cross product to check if point is to the left of edge p1->p2
            # For CCW polygon, interior is to the left.
            cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
            if cross < -1e-9: # If strictly to the right (outside)
                return False
        return True

    def distance_point_to_segment(p, a, b):
        """
        Calculate distance from point p to segment ab.
        """
        px, py = p
        ax, ay = a
        bx, by = b
        
        dx = bx - ax
        dy = by - ay
        
        if dx == 0 and dy == 0:
            return math.sqrt((px - ax)**2 + (py - ay)**2)
            
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def calculate_min_distance_to_hull(hull, inner_pts):
        """
        Calculate the minimum distance from any inner point to the hull boundary.
        Returns negative distance if a point is outside the hull.
        """
        min_dist = float('inf')
        n_hull = len(hull)
        
        for p in inner_pts:
            # Check if inside
            if not point_in_polygon(p, hull):
                # Calculate distance to hull edges
                for i in range(n_hull):
                    a = hull[i]
                    b = hull[(i + 1) % n_hull]
                    d = distance_point_to_segment(p, a, b)
                    if d < min_dist:
                        min_dist = d
                # If outside, the distance is effectively negative for our objective?
                # No, we want to penalize being outside.
                # The "clearance" is negative.
                # But we need to know how far outside.
                # For the objective, we just want the min clearance.
                # If outside, clearance is -distance_to_boundary.
                # So we return -min_dist if outside?
                # Actually, if any point is outside, the configuration is invalid or has negative clearance.
                # We want to maximize min_clearance.
                # If point is outside, clearance is -distance.
                # So we take the min over all points.
                # If point is inside, clearance is distance to boundary.
                # If point is outside, clearance is -distance to boundary.
                # This works.
                pass
            
            # Even if inside, we need distance to boundary for the min clearance.
            # Wait, if it's inside, the distance to boundary is positive.
            # If it's outside, the distance to boundary is negative (conceptually).
            # So we just calculate distance to boundary for all points.
            for i in range(n_hull):
                a = hull[i]
                b = hull[(i + 1) % n_hull]
                d = distance_point_to_segment(p, a, b)
                if not point_in_polygon(p, hull):
                    d = -d
                if d < min_dist:
                    min_dist = d
                    
        return min_dist

    def objective(params):
        """
        Objective function for optimization.
        params: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, cx, cy, S, beta]
        Total length: 3*N + 4 = 37
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10

        # Unpack
        inner_params = params[:3 * N]
        cx = params[3 * N]
        cy = params[3 * N + 1]
        S = params[3 * N + 2]
        beta = params[3 * N + 3]

        # Ensure S is positive
        if S < 0.1:
            return 1e10

        # Extract inner hexagons
        inner_hexes = []
        for i in range(N):
            x = inner_params[3 * i]
            y = inner_params[3 * i + 1]
            a = inner_params[3 * i + 2]
            inner_hexes.append((x, y, a))

        # Generate all vertices
        all_verts = []
        inner_verts = []
        for i, (x, y, a) in enumerate(inner_hexes):
            verts = get_vertices(x, y, a)
            for v in verts:
                all_verts.append(v)
                if i < N: # All are inner
                    inner_verts.append(v)

        # Calculate Convex Hull
        if len(all_verts) < 3:
            return 1e10
        hull = get_convex_hull(all_verts)
        
        if len(hull) < 3:
            return 1e10

        # Check if hull is convex (it should be by definition of convex hull, but just in case of degenerate cases)
        # Actually, monotone chain returns a convex hull, so it is convex.

        # Calculate min clearance
        min_clearance = calculate_min_distance_to_hull(hull, inner_verts)
        
        # Calculate outer side length from hull
        # We need to find the smallest regular hexagon that contains the hull.
        # The objective in the previous program was to minimize S given the points.
        # Here, we have S as a variable. We need to check if the hull fits inside a hexagon of side S at center (cx,cy) and rotation beta.
        # If it fits, we want to minimize S.
        # If it doesn't fit, we want to penalize heavily.
        
        # Check containment
        # Outer hexagon vertices
        outer_verts = []
        for i in range(6):
            angle = beta + i * math.pi / 3
            vx = cx + S * math.cos(angle)
            vy = cy + S * math.sin(angle)
            outer_verts.append((vx, vy))
            
        # Check if all hull points are inside outer hexagon
        # Since outer hexagon is convex, we can use point_in_polygon
        # But we need to handle the rotation and center.
        # The point_in_polygon function assumes the polygon is given in CCW order.
        # Our outer_verts are generated in CCW order.
        
        is_inside = True
        for v in hull:
            if not point_in_polygon(v, outer_verts):
                is_inside = False
                break
        
        if not is_inside:
            # Penalty for not fitting
            # Calculate how much it sticks out? Or just a large penalty.
            # A large penalty is simpler.
            # To help the optimizer, we can calculate the max distance outside.
            # For now, just return a large value.
            return 1e6 + S
        
        # If it fits, we want to minimize S.
        # We also want to maximize min_clearance to ensure disjointness.
        # The task says "disjoint". This means the inner hexagons must not overlap.
        # My current objective only checks if they fit in the outer hexagon and calculates clearance to the hull.
        # It does NOT check if inner hexagons overlap each other!
        
        # This is a critical missing piece.
        # We need to check pairwise distance between inner hexagons.
        # Two convex polygons are disjoint if their distance is > 0.
        # We can approximate this by checking if any vertex of one is inside the other,
        # or if any edge intersects.
        # Given the small number of hexagons (11), we can do a pairwise check.
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Get vertices
                vi = get_vertices(inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][2])
                vj = get_vertices(inner_hexes[j][0], inner_hexes[j][1], inner_hexes[j][2])
                
                # Check intersection
                # Simple check: is any vertex of i inside j?
                overlap = False
                for p in vi:
                    if point_in_polygon(p, vj):
                        overlap = True
                        break
                if not overlap:
                    for p in vj:
                        if point_in_polygon(p, vi):
                            overlap = True
                            break
                
                if overlap:
                    overlap_penalty += 1000.0
                    break # No need to check more for this pair? Or sum them up?
                    # Summing up might be better for gradient.
                    # But 1000 is large.
                
                # Also check edge intersections?
                # If they don't contain each other's vertices, they might still intersect.
                # For regular hexagons, if they are close, they likely contain vertices or are far.
                # A robust check is needed.
                # Let's use a simpler heuristic: distance between centers.
                # Min distance for disjoint unit hexagons is 2 * APOTHEM? No.
                # If they are aligned, distance between centers must be >= 2 * APOTHEM?
                # Actually, the width of a unit hexagon is 2 * APOTHEM.
                # So if centers are further than 2 * APOTHEM, they might still overlap if rotated?
                # No, the minimum distance between two disjoint unit hexagons is related to their relative orientation.
                # The safest check is geometric intersection.
                
                # Let's implement a proper segment intersection check.
                # This is expensive but N=11 is small.
                
                def segments_intersect(p1, p2, p3, p4):
                    def orientation(a, b, c):
                        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
                        if val > 0: return 1
                        if val < 0: return 2
                        return 0
                    def on_segment(a, b, c):
                        return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
                                min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))
                        
                    o1 = orientation(p1, p2, p3)
                    o2 = orientation(p1, p2, p4)
                    o3 = orientation(p3, p4, p1)
                    o4 = orientation(p3, p4, p2)
                    
                    if o1 != o2 and o3 != o4:
                        return True
                    if o1 == 0 and on_segment(p1, p3, p2): return True
                    if o2 == 0 and on_segment(p1, p4, p2): return True
                    if o3 == 0 and on_segment(p3, p1, p4): return True
                    if o4 == 0 and on_segment(p3, p2, p4): return True
                    return False

                # Check all edge pairs
                n_vi = len(vi)
                n_vj = len(vj)
                for k in range(n_vi):
                    if overlap: break
                    for m in range(n_vj):
                        if segments_intersect(vi[k], vi[(k+1)%n_vi], vj[m], vj[(m+1)%n_vj]):
                            overlap_penalty += 1000.0
                            overlap = True
                            break

        if overlap_penalty > 0:
            return 1e6 + overlap_penalty + S

        # If valid, return S. We want to minimize S.
        # To encourage compactness, we can add a small term for min_clearance?
        # No, min_clearance is for the hull.
        # The objective is just S.
        return S

    configs = generate_configurations()
    
    best_S = float('inf')
    best_params = None

    # Differential Evolution bounds
    # x, y in [-5, 5]
    # angle in [0, 180] (since hexagon has 6-fold symmetry, 0-60 is enough, but 0-180 is safer for DE)
    # cx, cy in [-5, 5]
    # S in [1.0, 10.0]
    # beta in [0, 60]
    
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, 180.0))
    bounds.append((-5.0, 5.0)) # cx
    bounds.append((-5.0, 5.0)) # cy
    bounds.append((1.0, 10.0)) # S
    bounds.append((0.0, 60.0)) # beta

    # Run DE for a limited number of iterations
    # maxiter=100, popsize=20
    # This might be slow. 11*3+4 = 37 variables.
    # Let's try to be efficient.
    
    # We will run DE on the best few configs.
    # Or just run DE once with multiple initial populations?
    # DE doesn't take multiple initial populations directly, but we can seed the population.
    # Or we can just run DE with a good bound.
    
    # Let's try running DE with the structured config as the initial guess?
    # DE generates random initial population.
    # We can force the first few individuals to be our configs.
    
    # Actually, let's just run DE and hope it finds the structure.
    # To speed up, we can reduce popsize and maxiter.
    
    # Time limit check
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=15, 
            seed=42,
            tol=1e-6,
            init='sobol'
        )
        
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x
            
    except Exception as e:
        pass

    # If DE failed or didn't find a good solution, try local optimization from best config
    if best_params is None:
        # Use the structured config
        best_params = []
        for x, y, a in configs[1]: # The hex grid one
            best_params.extend([x, y, a])
        # Calculate hull and fit hexagon
        # This is complex. Let's just pick a large S.
        best_params.extend([0.0, 0.0, 5.0, 0.0])
        
    # Local optimization to polish
    # We can use minimize on the best_params found by DE
    # But the objective is non-smooth (due to hull and intersection checks).
    # Nelder-Mead might work better for non-smooth.
    
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            # We need to make sure the current best is valid (no overlaps, fits in S)
            # If not, minimize will fail.
            
            # Let's try a few local optimizations from slightly perturbed best_params
            for _ in range(3):
                perturbed = list(best_params)
                for i in range(len(perturbed)):
                    if i < 3 * N:
                        perturbed[i] += np.random.normal(0, 0.1)
                    else:
                        perturbed[i] += np.random.normal(0, 0.01)
                        
                res_local = minimize(objective, perturbed, method='Nelder-Mead', 
                                     options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
                if res_local.fun < best_S:
                    best_S = res_local.fun
                    best_params = res_local.x
                    
        except Exception:
            pass

    # Extract final solution
    if best_params is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    inner = []
    for i in range(N):
        x = best_params[3 * i]
        y = best_params[3 * i + 1]
        a = best_params[3 * i + 2]
        inner.append([float(x), float(y), float(a)])
        
    cx = float(best_params[3 * N])
    cy = float(best_params[3 * N + 1])
    S = float(best_params[3 * N + 2])
    beta = float(best_params[3 * N + 3])
    
    # Ensure S is not too small (validity)
    if S < 1.0:
        S = 1.0
        
    center = [cx, cy]
    rotation = beta
    
    return inner, center, S, rotation

