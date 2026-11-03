import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals are at angles theta + 30 + k*60
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint(poly1, poly2, tol=1e-7):
        """
        Fast check for disjointness of two convex hexagons.
        Returns True if disjoint (with tolerance), False if overlapping.
        """
        min_x1, min_y1 = np.min(poly1, axis=0)
        max_x1, max_y1 = np.max(poly1, axis=0)
        min_x2, min_y2 = np.min(poly2, axis=0)
        max_x2, max_y2 = np.max(poly2, axis=0)
        
        if max_x1 < min_x2 - tol or max_x2 < min_x1 - tol or max_y1 < min_y2 - tol or max_y2 < min_y1 - tol:
            return True

        # Point in Polygon check (Ray casting)
        def point_in_poly(pt, poly):
            x, y = pt
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                    inside = not inside
            return inside

        # Check vertices of v1 in v2
        for pt in poly1:
            if point_in_poly(pt, poly2):
                return False
        
        # Check vertices of v2 in v1
        for pt in poly2:
            if point_in_poly(pt, poly1):
                return False
        
        # Check edge intersections
        def segments_intersect(p1, p2, p3, p4):
            def ccw(a, b, c):
                return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

        for i in range(6):
            p1 = poly1[i]
            p2 = poly1[(i+1)%6]
            for j in range(6):
                p3 = poly2[j]
                p4 = poly2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        
        return True

    def min_separation_distance(angle1, angle2):
        """
        Calculate the minimum center-to-center distance required for two unit hexagons
        with given angles to be disjoint (touching allowed).
        We approximate this by sampling relative rotations. 
        For regular hexagons, the min distance is min over all directions u of (h1(u) + h2(-u)).
        Since they are congruent, h1(u) = h2(u).
        Actually, it's easier to just precompute or use a safe bound.
        The minimum width of a unit hexagon is sqrt(3).
        The maximum width is 2.
        If they are aligned, min dist is sqrt(3) (flat-to-flat).
        If rotated by 30 deg, the "corners" might protrude?
        Let's check: 
        Hex A: flats at 30, 90...
        Hex B: rotated 30 deg -> flats at 60, 120... (vertices at 30, 90...)
        If B is rotated 30, its vertices point towards A's flats.
        Distance from center of A to flat is sqrt(3)/2.
        Distance from center of B to vertex is 1.
        If aligned along the normal of A's flat (30 deg), B's vertex is at 30 deg relative to B's center?
        If B is rotated 30 deg, its vertices are at 30, 90, 150... relative to global frame?
        Wait. Vertices of unit hex at angle 0 are at 0, 60, 120...
        If rotated 30 deg, vertices are at 30, 90, 150...
        Flats of hex at angle 0 are at 30, 90... (normals).
        So if A is at 0, its flat normal is at 30.
        If B is at 30, its vertex is at 30.
        So B's vertex points directly at A's flat.
        Distance = h_A(30) + h_B(150)? No.
        Distance = h_A(u) + h_B(-u).
        Let u be direction from A to B.
        h_A(u) is distance from center A to support plane in dir u.
        h_B(-u) is distance from center B to support plane in dir -u.
        
        For regular hexagon side 1:
        h(theta) = (sqrt(3)/2) / cos((theta mod 60) - 30) ? No.
        h(theta) = 1 / cos( (theta mod 60) - 30 ) * (sqrt(3)/2) ?
        Let's use the formula: h(theta) = (sqrt(3)/2) / cos( (theta mod 60) - 30 ) is for circumradius 1?
        No, for side length s=1:
        Apothem a = sqrt(3)/2.
        h(theta) = a / cos( (theta mod 60) - 30 ) is incorrect.
        
        Correct formula for support function of regular hexagon (side 1, centered at origin):
        h(theta) = (sqrt(3)/2) / cos( (theta mod 60) - 30 ) is actually for the *circumradius* if the denominator was 1.
        Let's derive.
        Vertex at (1, 0). Edge from (1,0) to (1/2, sqrt(3)/2).
        Normal to this edge is at 30 deg.
        Distance from origin to this edge (apothem) is sqrt(3)/2.
        So h(30) = sqrt(3)/2.
        h(0) = 1.
        h(theta) = (sqrt(3)/2) / cos( (theta mod 60) - 30 ) ?
        At theta=0: (sqrt(3)/2) / cos(-30) = (sqrt(3)/2) / (sqrt(3)/2) = 1. Correct.
        At theta=30: (sqrt(3)/2) / cos(0) = sqrt(3)/2. Correct.
        
        So h(theta) = (sqrt(3)/2) / cos( (theta mod 60) - 30 ).
        
        We need min over theta of h(theta - angle1) + h(-(theta) - angle2).
        Since h is periodic with 60, we can just sample theta in [0, 60].
        """
        def h(theta, angle):
            # Normalize theta relative to angle
            t = (theta - angle) % 60.0
            # t is in [0, 60).
            # The function is symmetric around 30?
            # h(theta) peaks at vertices (0, 60) and dips at flats (30).
            # cos((t - 30)) is max at 30, min at 0, 60.
            # So h is max at 0, 60 and min at 30.
            # We need to be careful with the modulo.
            # If t is close to 0, cos(t-30) = cos(-30) = sqrt(3)/2. h=1.
            # If t is 30, cos(0)=1. h=sqrt(3)/2.
            return (math.sqrt(3)/2) / math.cos(math.radians(t - 30))
        
        min_dist = float('inf')
        # Sample theta from 0 to 60
        for k in range(61):
            theta = k
            d = h(theta, angle1) + h(theta + 180, angle2) # h(-theta) is same as h(theta+180) due to symmetry?
            # h(-theta) = h(360-theta). Due to 60 period, h(-theta) = h(60-theta) if we adjust?
            # Actually h(-theta) for hexagon:
            # h(-theta) = (sqrt(3)/2) / cos( (-theta mod 60) - 30 ).
            # Let's just use the property h(-u) = h(u) for centrally symmetric bodies?
            # Hexagon is centrally symmetric. h(-u) = h(u).
            # So we need h(theta - angle1) + h(theta - angle2).
            d = h(theta, angle1) + h(theta, angle2)
            if d < min_dist:
                min_dist = d
        return min_dist

    # Strategy:
    # 1. Generate a good initial guess (honeycomb).
    # 2. Use SLSQP to minimize the outer side length.
    # 3. Constraints: For each pair (i, j), distance(pos_i, pos_j) >= min_sep(angle_i, angle_j).
    #    Also, we can fix the center of mass at (0,0) to remove translational degrees of freedom.
    #    And fix the global rotation? The outer hexagon is fixed at 0. 
    #    The inner hexagons can rotate.
    
    best_score = float('inf')
    best_positions = None
    best_angles = None

    sqrt3 = math.sqrt(3)
    
    # Initial guess: Honeycomb lattice
    # Centers at distance sqrt(3) apart.
    # v1 = (0, sqrt3), v2 = (1.5, sqrt3/2)
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
            
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    # Verify initial guess
    polys = [get_hex_verts(init_positions[i][0], init_positions[i][1], 1.0, 0.0) for i in range(N)]
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not check_disjoint(polys[i], polys[j]):
                valid = False
                break
        if not valid: break
    
    if valid:
        score = calc_min_outer_side(init_positions, init_angles, 0.0)
        if score < best_score:
            best_score = score
            best_positions = init_positions.copy()
            best_angles = init_angles.copy()

    # Optimization
    if best_positions is not None:
        
        def objective(params):
            if time.time() > deadline - 5:
                return 1e6
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            
            # Center of mass constraint is handled by SLSQP constraints, but we can also just minimize side.
            return calc_min_outer_side(positions, angles, 0.0)
            
        def constraint_com(params):
            if time.time() > deadline - 5:
                return 0.0
            positions = params[:2*N].reshape(N, 2)
            return np.sum(positions, axis=0) # Should be 0
            
        # Pairwise separation constraints
        # There are N*(N-1)/2 = 66 constraints.
        # This might be slow for SLSQP.
        # We can reduce constraints by only checking nearby pairs in the initial lattice?
        # Or just use all of them. SLSQP handles sparse constraints well if we provide them.
        
        def constraints():
            cons = []
            # COM constraint
            cons.append({'type': 'eq', 'fun': constraint_com, 'jac': lambda p: None}) # Jac is tricky, let SLSQP estimate or provide.
            
            # Separation constraints
            for i in range(N):
                for j in range(i+1, N):
                    def sep_con(params, i=i, j=j):
                        if time.time() > deadline - 5:
                            return 0.0
                        positions = params[:2*N].reshape(N, 2)
                        angles = params[2*N:]
                        d = np.linalg.norm(positions[i] - positions[j])
                        min_d = min_separation_distance(angles[i], angles[j])
                        return d - min_d
                    cons.append({'type': 'ineq', 'fun': sep_con})
            return cons

        # SLSQP requires Jacobians for efficiency, but can estimate them.
        # Given 66 constraints + 1 eq, and 36 variables, this is a dense problem.
        # Let's try it.
        
        x0 = np.concatenate([best_positions.flatten(), best_angles])
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                constraints=constraints(),
                options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.fun < best_score:
                new_positions = res.x[:2*N].reshape(N, 2)
                new_angles = res.x[2*N:]
                
                # Verify validity
                polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint(polys[i], polys[j]):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    best_score = res.fun
                    best_positions = new_positions
                    best_angles = new_angles
        except Exception as e:
            pass

    # Fallback
    if best_positions is None:
        best_positions = init_positions
        best_angles = init_angles
        best_score = calc_min_outer_side(best_positions, best_angles, 0.0)

    final_positions = best_positions
    final_angles = best_angles
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
