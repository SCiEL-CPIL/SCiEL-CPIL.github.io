import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Regular hexagon geometry helper
    def get_hex_vertices(center_x, center_y, side, angle_deg):
        """
        Returns vertices of a regular hexagon.
        center: (x, y)
        side: side length
        angle_deg: rotation in degrees
        """
        angle_rad = math.radians(angle_deg)
        # Vertices at angles 0, 60, 120, 180, 240, 300 relative to rotation
        # Standard flat-top or pointy-top? 
        # Let's use vertices at angles: angle + i*60
        # For a regular hexagon with side s, distance from center to vertex is s.
        vertices = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            x = center_x + side * math.cos(theta)
            y = center_y + side * math.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def point_in_hexagon(px, py, hx, hy, side, angle_deg):
        """
        Check if point (px, py) is inside hexagon centered at (hx, hy) with side 'side' and angle 'angle_deg'.
        Uses the property that a regular hexagon is the intersection of 6 half-planes.
        """
        verts = get_hex_vertices(hx, hy, side, angle_deg)
        # For each edge, check if point is on the correct side (inside)
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            # Edge vector
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            # Normal vector pointing inward
            # The cross product (p2-p1) x (p-p1) should have consistent sign for all points inside.
            # Let's determine the "inside" side by checking the center.
            cx = hx
            cy = hy
            # Vector from p1 to center
            v_center_x = cx - p1[0]
            v_center_y = cy - p1[1]
            # Vector from p1 to point
            v_point_x = px - p1[0]
            v_point_y = py - p1[1]
            
            # Cross product z-component: ex * vy - ey * vx
            cross_center = ex * v_center_y - ey * v_center_x
            cross_point = ex * v_point_y - ey * v_point_x
            
            if cross_center == 0:
                continue
            # Signs must match
            if (cross_center > 0 and cross_point < 0) or (cross_center < 0 and cross_point > 0):
                return False
        return True

    def hexagons_overlap(v1, v2, side1, angle1, side2, angle2, tol=1e-6):
        """
        Check if two hexagons overlap.
        v1, v2: centers (x,y)
        """
        # 1. Check if any vertex of hex1 is in hex2
        verts1 = get_hex_vertices(v1[0], v1[1], side1, angle1)
        for v in verts1:
            if point_in_hexagon(v[0], v[1], v2[0], v2[1], side2, angle2):
                return True
        # 2. Check if any vertex of hex2 is in hex1
        verts2 = get_hex_vertices(v2[0], v2[1], side2, angle2)
        for v in verts2:
            if point_in_hexagon(v[0], v[1], v1[0], v1[1], side1, angle1):
                return True
        # 3. Check edge intersections (if neither contains the other's vertices)
        # This is a robust check for convex polygons
        # We can use a simpler distance check or segment intersection.
        # Given time constraints, vertex containment is often sufficient for "disjoint" if we add a small buffer, 
        # but strictly disjoint means no intersection.
        # Let's implement a quick segment intersection check for safety.
        
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
        - 12 * 3 for inner hexagons (x, y, angle)
        - 3 for outer hexagon (x, y, side, angle) -- wait, 4 params? 
          Let's say outer is centered at (0,0) for symmetry to reduce params? 
          No, better to allow movement but symmetry suggests (0,0) is optimal. 
          Let's fix outer center to (0,0) to reduce dimensionality and search space.
          Outer params: side, angle. (2 params)
        Total params: 36 + 2 = 38.
        """
        # Extract inner hexagons
        inner_hexes = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            angle = params[i*3+2]
            inner_hexes.append((x, y, 1.0, angle)) # side=1
        
        # Extract outer hexagon
        outer_side = params[N*3]
        outer_angle = params[N*3+1]
        
        # Constraint 1: Outer side length must be positive
        if outer_side <= 0:
            return 1e10
            
        # Constraint 2: Inner hexagons must be disjoint
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(inner_hexes[i][:2], inner_hexes[j][:2], 
                                    inner_hexes[i][2], inner_hexes[i][3], 
                                    inner_hexes[j][2], inner_hexes[j][3]):
                    return 1e10 # Large penalty for overlap
        
        # Constraint 3: Inner hexagons must be inside outer hexagon
        outer_center = (0.0, 0.0)
        for ih in inner_hexes:
            verts = get_hex_vertices(ih[0], ih[1], ih[2], ih[3])
            for v in verts:
                if not point_in_hexagon(v[0], v[1], outer_center[0], outer_center[1], outer_side, outer_angle):
                    return 1e10 # Large penalty for containment violation
        
        # Objective: Minimize outer side length
        return outer_side

    # Define bounds for differential evolution
    # Inner hexagons: x, y in [-10, 10], angle in [0, 360]
    # Outer side: [3, 20], angle: [0, 360]
    
    bounds = []
    for _ in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
        bounds.append((0, 360))  # angle
    
    bounds.append((3, 20))   # outer side
    bounds.append((0, 360))  # outer angle

    # Run differential evolution
    # Use a reasonable maxiter and popsize to fit in time
    # 38 dimensions is high for DE.
    # Let's try a smaller population and fewer iterations, or use a smarter initialization.
    
    # Initialization: Honeycomb lattice
    sqrt3 = math.sqrt(3)
    init_params = []
    # 4 rows of 3
    # Row 0: y=0, x = -2, 0, 2
    # Row 1: y=sqrt(3), x = -1, 1, 3 -> Centered? 
    # Let's center the honeycomb block.
    # Total width approx 4, height approx 3*sqrt(3) ~ 5.2
    # Centers:
    coords = [
        (-2, 0), (0, 0), (2, 0),
        (-1, sqrt3), (1, sqrt3), (3, sqrt3),
        (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
        (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
    ]
    # Shift to center at (0,0)
    cx = sum(c[0] for c in coords) / N
    cy = sum(c[1] for c in coords) / N
    for c in coords:
        init_params.append(c[0] - cx)
        init_params.append(c[1] - cy)
        init_params.append(0.0) # angle
    
    # Estimate optimal outer side for this config
    # Max distance from center to vertex of inner hexagons + 1
    max_r = 0
    for c in coords:
        x = c[0] - cx
        y = c[1] - cy
        # Distance to vertex is 1
        # Max extent in any direction
        # Rough estimate: max(|x|, |y|) + 1
        r = math.sqrt(x*x + y*y) + 1
        if r > max_r:
            max_r = r
    # Side length of regular hexagon with circumradius R is R
    # So outer side ~ max_r
    init_params.append(max_r)
    init_params.append(0.0) # outer angle

    # Check time
    if time.time() > deadline - 20:
        # Return a safe fallback
        return [[c[0]-cx, c[1]-cy, 0.0] for c in coords], (0.0, 0.0), max_r, 0.0

    # Run optimization
    # DE is stochastic. We'll run it with a time limit check inside the callback?
    # DE doesn't easily support time-based stopping in scipy, but we can limit maxiter.
    # 38 dims, popsize=20, maxiter=100 might be slow.
    # Let's try popsize=10, maxiter=50.
    
    try:
        result = differential_evolution(
            evaluate_solution, 
            bounds, 
            x0=init_params,
            popsize=10, 
            maxiter=30, 
            tol=1e-3, 
            seed=42,
            polish=True # Use L-BFGS-B for final polish
        )
        
        if result.success or result.fun < 1e9:
            best_params = result.x
            # Extract result
            inner = []
            for i in range(N):
                x = best_params[i*3]
                y = best_params[i*3+1]
                angle = best_params[i*3+2]
                inner.append([x, y, angle])
            outer_side = best_params[N*3]
            outer_angle = best_params[N*3+1]
            
            # Final verification
            # If invalid, fall back to init
            if evaluate_solution(best_params) > 1e9:
                raise ValueError("Optimization result invalid")
                
            return inner, (0.0, 0.0), outer_side, outer_angle
        else:
            raise ValueError("Optimization failed")
            
    except Exception as e:
        # Fallback to honeycomb lattice
        inner = [[c[0]-cx, c[1]-cy, 0.0] for c in coords]
        return inner, (0.0, 0.0), max_r, 0.0

# EVOLVE-BLOCK-END
