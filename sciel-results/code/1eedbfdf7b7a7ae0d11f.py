import math
import time
import numpy as np
from scipy.optimize import linprog

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute vertices for unit hexagon at angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # For a regular hexagon with side 1, circumradius is 1.
    # Standard "flat-top" if angle 0 is a vertex?
    # Let's define angle 0 such that vertices are at 0, 60, ...
    # This is "pointy-top" if we consider x-axis horizontal?
    # Actually, vertex at (1,0), (1/2, sqrt(3)/2), etc.
    # This is a hexagon with a vertex on the x-axis.
    
    def get_hex_vertices_unit(angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            x = math.cos(theta)
            y = math.sin(theta)
            verts.append([x, y])
        return np.array(verts)
    
    # Helper to check if a point is inside a hexagon
    # For convex polygon defined by vertices in CCW order
    # We can use the half-plane intersection property.
    # A point P is inside if for every edge (V_i, V_{i+1}), P is on the left (or right) side.
    # Since vertices are CCW, P is inside if cross(V_{i+1}-V_i, P-V_i) >= 0 for all i.
    
    def point_in_hexagon(px, py, hx, hy, side, angle_deg, verts_unit):
        # Transform point to local coordinates of hexagon
        # Translate center to origin
        dx = px - hx
        dy = py - hy
        # Rotate by -angle
        angle_rad = -math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rx = cos_a * dx - sin_a * dy
        ry = sin_a * dx + cos_a * dy
        # Scale by 1/side
        rx /= side
        ry /= side
        
        # Check against unit hexagon vertices (CCW)
        # verts_unit is CCW
        for i in range(6):
            v1 = verts_unit[i]
            v2 = verts_unit[(i + 1) % 6]
            # Edge vector
            ex = v2[0] - v1[0]
            ey = v2[1] - v1[1]
            # Vector from v1 to point
            px_local = rx - v1[0]
            py_local = ry - v1[1]
            # Cross product z-component
            cross = ex * py_local - ey * px_local
            if cross < -1e-9: # Allow small tolerance
                return False
        return True

    def hexagons_overlap(v1, v2, side1, angle1, side2, angle2, verts1_unit, verts2_unit):
        """
        Check if two hexagons overlap.
        v1, v2: centers (x,y)
        side1, side2: side lengths
        angle1, angle2: angles in degrees
        """
        # 1. Check if any vertex of hex1 is in hex2
        verts1 = v1 + side1 * verts1_unit
        for v in verts1:
            if point_in_hexagon(v[0], v[1], v2[0], v2[1], side2, angle2, verts2_unit):
                return True
        # 2. Check if any vertex of hex2 is in hex1
        verts2 = v2 + side2 * verts2_unit
        for v in verts2:
            if point_in_hexagon(v[0], v[1], v1[0], v1[1], side1, angle1, verts1_unit):
                return True
        # 3. Check edge intersections
        # If no vertex is inside, and they are convex, they might still intersect.
        # However, for disjoint convex polygons, if no vertex is inside the other, 
        # and no edges intersect, they are disjoint.
        
        def segments_intersect(p1, p2, p3, p4):
            def ccw(a, b, c):
                return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

        for i in range(6):
            p1 = verts1[i]
            p2 = verts1[(i+1)%6]
            for j in range(6):
                p3 = verts2[j]
                p4 = verts2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return True
        return False

    def evaluate_solution(params):
        """
        params: 
        - 12 * 3 for inner hexagons (x, y, angle_index) -- angle_index 0 or 1?
          Let's restrict angles to 0 and 30 degrees? Or just 0?
          Actually, for hexagons, rotating by 30 degrees swaps "flat" and "pointy" sides.
          Let's allow angles 0, 30, 60, 90, 120, 150. But due to symmetry, 0 and 30 are distinct.
          Let's try just 0 degrees first to reduce search space, or 0 and 30.
          Let's use 0 and 30.
        - 2 for outer hexagon (side, angle) -- fix outer center to (0,0).
        """
        # Extract inner hexagons
        inner_hexes = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            angle_idx = params[i*3+2]
            # Map index to angle: 0->0, 1->30
            angle = 0.0 if angle_idx < 0.5 else 30.0
            inner_hexes.append((x, y, 1.0, angle))
        
        # Extract outer hexagon
        outer_side = params[N*3]
        outer_angle = params[N*3+1]
        
        if outer_side <= 0:
            return 1e10
            
        # Constraint 1: Inner hexagons must be disjoint
        # Precompute unit vertices for angles 0 and 30
        verts_0 = get_hex_vertices_unit(0.0)
        verts_30 = get_hex_vertices_unit(30.0)
        
        for i in range(N):
            for j in range(i+1, N):
                ih_i = inner_hexes[i]
                ih_j = inner_hexes[j]
                # Get correct vertices
                v1 = np.array([ih_i[0], ih_i[1]])
                v2 = np.array([ih_j[0], ih_j[1]])
                
                # Rotate unit vertices to match angle
                # Since angle is 0 or 30, we can just use precomputed
                # But we need to rotate the *positions*? No, the vertices are defined relative to center.
                # get_hex_vertices_unit returns vertices for angle 0.
                # For angle 30, we need rotated vertices.
                
                # Let's just compute vertices on the fly for simplicity in this check, 
                # or precompute for 0 and 30.
                if ih_i[3] == 0.0:
                    v1_unit = verts_0
                else:
                    v1_unit = verts_30
                    
                if ih_j[3] == 0.0:
                    v2_unit = verts_0
                else:
                    v2_unit = verts_30

                if hexagons_overlap(v1, v2, ih_i[2], ih_i[3], ih_j[2], ih_j[3], v1_unit, v2_unit):
                    return 1e10 # Large penalty for overlap
        
        # Constraint 2: Inner hexagons must be inside outer hexagon
        outer_center = (0.0, 0.0)
        # Outer vertices
        outer_verts_unit = get_hex_vertices_unit(outer_angle)
        
        for ih in inner_hexes:
            verts = np.array([ih[0], ih[1]]) + ih[2] * get_hex_vertices_unit(ih[3])
            for v in verts:
                if not point_in_hexagon(v[0], v[1], outer_center[0], outer_center[1], outer_side, outer_angle, outer_verts_unit):
                    return 1e10 # Large penalty for containment violation
        
        # Objective: Minimize outer side length
        return outer_side

    # Strategy:
    # 1. Fix outer hexagon angle to 0. (Symmetry suggests this is optimal or close).
    # 2. Discretize inner hexagon angles to 0 and 30.
    # 3. Use a grid search or local optimization?
    #    Given the complexity, a simple local search (hill climbing) on positions might be better than DE.
    #    Or, we can use the LP idea: 
    #    For a fixed set of angles, the problem is:
    #    Minimize R (outer side)
    #    Subject to:
    #    - Inner hexagons disjoint (non-linear constraints)
    #    - Inner hexagons inside outer (linear constraints on positions given R and angles)
    
    # Since disjointness is non-linear, LP doesn't directly apply to the whole problem.
    # However, we can use a "best-fit" packing algorithm.
    
    # Let's try a multi-start local optimization using Nelder-Mead or similar.
    # But DE was tried. Let's try a different initialization and a faster local optimizer.
    
    # Actually, let's stick to a robust heuristic:
    # 1. Generate random configurations of angles (0 or 30) for each inner hex.
    # 2. Use a simple gradient-free local search (e.g., Nelder-Mead) to minimize outer side.
    # 3. Repeat multiple times.
    
    # To make it faster, we can precompute the "required" outer side for a given configuration.
    # The required outer side is the minimum R such that all inner hexes are inside.
    # This is a convex problem? 
    # For fixed inner positions and angles, the minimum R is the maximum over all vertices of inner hexes 
    # of the "radial distance" in the hexagonal norm?
    # Actually, for a regular hexagon container, the condition "point P is inside hexagon centered at 0 with side R"
    # is equivalent to:
    # max( |x|, |y|/sqrt(3) + |x|/sqrt(3), ... ) <= R ?
    # Let's derive the inequality for a point (x,y) to be inside a regular hexagon of side R centered at origin, angle 0.
    # Vertices at R*(1,0), R*(1/2, sqrt(3)/2), etc.
    # The hexagon is the intersection of 6 half-planes.
    # Edges are at distance R*sqrt(3)/2 from center.
    # The normal vectors to the edges are at angles 30, 90, 150, 210, 270, 330 degrees.
    # Inequality: n_i . p <= R * sqrt(3) / 2
    # So, for a point p, the required R is max_i (n_i . p) / (sqrt(3)/2) = max_i (n_i . p) * 2/sqrt(3).
    # This is a linear function of p!
    
    # So, for a fixed configuration of inner hexagons (positions and angles), the minimal outer side R is:
    # R = max over all vertices v of all inner hexagons of ( required_R(v) )
    # where required_R(v) = max_i ( n_i . v ) * 2/sqrt(3)
    
    # This allows us to evaluate the objective function very quickly without checking containment explicitly!
    # We only need to check disjointness.
    
    def get_required_outer_side(inner_hexes, outer_angle=0.0):
        # Precompute normals for outer hexagon
        angle_rad = math.radians(outer_angle)
        # Normals are at angle + 30 + k*60
        normals = []
        for k in range(6):
            theta = angle_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(theta), math.sin(theta)])
        normals = np.array(normals)
        
        max_r = 0.0
        for ih in inner_hexes:
            x, y, side, angle_deg = ih
            verts = np.array([x, y]) + side * get_hex_vertices_unit(angle_deg)
            # For each vertex, calculate required R
            # Dots = verts @ normals.T
            dots = verts @ normals.T
            max_dot = np.max(dots)
            # R = max_dot * 2 / sqrt(3)
            r = max_dot * 2.0 / math.sqrt(3)
            if r > max_r:
                max_r = r
        return max_r

    def evaluate_solution_fast(params):
        # params: 12*3 (x, y, angle_idx) + 2 (outer_side, outer_angle)
        # We ignore outer_side in params and calculate it.
        # We fix outer_angle to 0.0 for now.
        inner_hexes = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            angle_idx = params[i*3+2]
            angle = 0.0 if angle_idx < 0.5 else 30.0
            inner_hexes.append((x, y, 1.0, angle))
        
        # Check disjointness
        verts_0 = get_hex_vertices_unit(0.0)
        verts_30 = get_hex_vertices_unit(30.0)
        
        for i in range(N):
            for j in range(i+1, N):
                ih_i = inner_hexes[i]
                ih_j = inner_hexes[j]
                v1 = np.array([ih_i[0], ih_i[1]])
                v2 = np.array([ih_j[0], ih_j[1]])
                
                if ih_i[3] == 0.0:
                    v1_unit = verts_0
                else:
                    v1_unit = verts_30
                    
                if ih_j[3] == 0.0:
                    v2_unit = verts_0
                else:
                    v2_unit = verts_30

                if hexagons_overlap(v1, v2, ih_i[2], ih_i[3], ih_j[2], ih_j[3], v1_unit, v2_unit):
                    return 1e10
        
        # Calculate minimal outer side
        return get_required_outer_side(inner_hexes, 0.0)

    # Optimization:
    # Use Nelder-Mead on the positions and angle indices.
    # Angle indices are discrete. We can treat them as continuous in [0, 1] and round.
    # Or, we can fix angles to 0 for all, and just optimize positions.
    # Let's try fixing angles to 0 first.
    
    # Initialization: Honeycomb lattice
    sqrt3 = math.sqrt(3)
    coords = [
        (-2, 0), (0, 0), (2, 0),
        (-1, sqrt3), (1, sqrt3), (3, sqrt3),
        (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
        (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
    ]
    cx = sum(c[0] for c in coords) / N
    cy = sum(c[1] for c in coords) / N
    
    init_params = []
    for c in coords:
        init_params.append(c[0] - cx)
        init_params.append(c[1] - cy)
        init_params.append(0.0) # angle idx 0
        
    # Bounds for Nelder-Mead are tricky. 
    # Let's use a simple grid search over angle configurations? No, 2^12 is 4096, manageable.
    # But positions are continuous.
    
    # Let's try a few random angle configurations and optimize positions with Nelder-Mead.
    
    best_score = 1e10
    best_params = None
    
    # Try a few random angle assignments
    np.random.seed(42)
    
    num_trials = 50
    trial_count = 0
    
    while time.time() < deadline - 10 and trial_count < num_trials:
        trial_count += 1
        
        # Random angles: 0 or 30
        angles = np.random.randint(0, 2, N)
        
        # Initial positions: Honeycomb, slightly perturbed
        x0 = []
        for i, c in enumerate(coords):
            x0.append(c[0] - cx + np.random.normal(0, 0.1))
            x0.append(c[1] - cy + np.random.normal(0, 0.1))
            x0.append(angles[i])
            
        # Use Nelder-Mead
        # We need to handle the discrete angle part.
        # We can just optimize positions, keeping angles fixed.
        # Define a function that only takes positions (2N variables)
        
        def obj_pos(pos):
            params = []
            for i in range(N):
                params.append(pos[2*i])
                params.append(pos[2*i+1])
                params.append(angles[i])
            return evaluate_solution_fast(params)
        
        try:
            from scipy.optimize import minimize
            res = minimize(obj_pos, x0[:2*N], method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_score:
                best_score = res.fun
                # Reconstruct params
                best_params = []
                for i in range(N):
                    best_params.append(res.x[2*i])
                    best_params.append(res.x[2*i+1])
                    best_params.append(angles[i])
                
        except Exception as e:
            pass

    # If we found a good solution, use it. Otherwise, fallback to initial honeycomb.
    if best_params is None:
        best_params = init_params
        best_score = evaluate_solution_fast(init_params)
        
    # Final verification and extraction
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        angle_idx = best_params[i*3+2]
        angle = 0.0 if angle_idx < 0.5 else 30.0
        inner.append([x, y, angle])
        
    outer_side = best_score
    outer_angle = 0.0
    
    # If score is invalid, fallback
    if outer_side > 1e9:
        inner = [[c[0]-cx, c[1]-cy, 0.0] for c in coords]
        # Calculate side for this
        test_params = []
        for c in coords:
            test_params.append(c[0]-cx)
            test_params.append(c[1]-cy)
            test_params.append(0.0)
        outer_side = evaluate_solution_fast(test_params)
        if outer_side > 1e9:
             # Absolute fallback
             outer_side = 10.0

    return inner, (0.0, 0.0), outer_side, outer_angle

# EVOLVE-BLOCK-END
