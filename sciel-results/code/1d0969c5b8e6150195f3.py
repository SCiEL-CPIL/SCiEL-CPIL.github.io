<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        """Return vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside regular hexagon centered at (cx,cy)."""
        # Transform to hexagon's local frame
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # For a regular hexagon with side s, inradius = s * sqrt(3)/2
        # The hexagon in local frame (pointing right) has vertices at angles 0,60,120,180,240,300
        # Edge normals are at angles 30,90,150,210,270,330
        inradius = side * math.sqrt(3) / 2.0
        
        # Check against each edge
        for i in range(6):
            normal_angle = math.radians(30 + i * 60)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            dist = lx * nx + ly * ny
            if dist > inradius:
                return False
        return True
    
    def hexagons_overlap(cx1, cy1, a1, cx2, cy2, a2, margin=1e-9):
        """Check if two unit hexagons overlap. Returns True if they overlap."""
        # Quick distance check
        dx = cx1 - cx2
        dy = cy1 - cy2
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 2.0 + margin:
            return False
        if dist < 1e-6:
            return True
        
        # Check if any vertex of hex1 is inside hex2 or vice versa
        verts1 = get_hex_vertices(cx1, cy1, 1.0, a1)
        verts2 = get_hex_vertices(cx2, cy2, 1.0, a2)
        
        for v in verts1:
            if point_in_hexagon(v[0], v[1], cx2, cy2, 1.0, a2):
                return True
        for v in verts2:
            if point_in_hexagon(v[0], v[1], cx1, cy1, 1.0, a1):
                return True
        
        # Check edge intersections
        def segments_intersect(p1, p2, p3, p4):
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False
        
        for i in range(6):
            e1 = (verts1[i], verts1[(i+1)%6])
            for j in range(6):
                e2 = (verts2[j], verts2[(j+1)%6])
                if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                    return True
        return False
    
    def max_vertex_distance(cx, cy, side, angle_deg):
        """Max distance from center to any vertex of the hexagon."""
        return side  # For regular hexagon, circumradius = side
    
    def required_outer_side(centers, angles, outer_angle_deg):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        # For each inner hexagon, compute max distance from outer center to any of its vertices
        # The outer hexagon must contain all these vertices
        # For a regular hexagon with side S, a point at distance r from center is inside
        # if r <= S (circumradius) and the angular position allows it
        
        # Actually, for a regular hexagon, the maximum distance from center to a point
        # on the boundary depends on the angle. The inradius is S*sqrt(3)/2 and circumradius is S.
        # A point at distance r from center is inside if r <= S * (something depending on angle)
        
        # Simpler: for each vertex of each inner hexagon, compute the minimum outer side
        # length needed such that vertex is inside the outer hexagon.
        
        max_side = 0.0
        for (cx, cy, a) in zip(centers, angles):
            verts = get_hex_vertices(cx, cy, 1.0, a)
            for (vx, vy) in verts:
                # Distance from origin (outer center at 0,0) to this vertex
                r = math.sqrt(vx*vx + vy*vy)
                # Angle of this vertex
                theta = math.atan2(vy, vx)
                # For outer hexagon with rotation outer_angle_deg, 
                # the boundary at angle theta has distance:
                # d(theta) = S * sqrt(3) / (2 * cos(theta - outer_angle - 30*k)) for appropriate k
                # Actually, the distance from center to edge at angle phi is:
                # inradius / cos(phi - edge_normal_angle)
                # Edge normals are at outer_angle + 30 + 60*k degrees
                # The minimum S such that r <= S * sqrt(3)/2 / cos(delta) where delta is angle diff
                
                outer_angle_rad = math.radians(outer_angle_deg)
                # Find the closest edge normal
                best_ratio = float('inf')
                for k in range(6):
                    normal_angle = outer_angle_rad + math.radians(30 + k * 60)
                    delta = theta - normal_angle
                    # Normalize delta to [-pi/2, pi/2]
                    delta = ((delta + math.pi) % (2*math.pi)) - math.pi
                    if abs(delta) < math.pi/2:
                        ratio = 1.0 / math.cos(delta)
                        if ratio < best_ratio:
                            best_ratio = ratio
                
                if best_ratio < float('inf'):
                    # r <= S * sqrt(3)/2 * best_ratio
                    # S >= r / (sqrt(3)/2 * best_ratio)
                    required_S = r / (math.sqrt(3)/2.0 * best_ratio)
                    if required_S > max_side:
                        max_side = required_S
        
        return max_side
    
    def objective(params):
        """Objective: minimize outer side length. params = [x1,y1,a1, x2,y2,a2, ...]"""
        if time.time() > deadline:
            return 1e10
        
        centers = []
        angles = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            centers.append(x)
            angles.append(a)
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(centers[i], centers[i+0]*0+0, 0, 0, 0, 0, 0):
                    pass  # placeholder, will do proper check
                # Proper check
                if hexagons_overlap(centers[i], 0, angles[i], centers[j], 0, angles[j]):
                    return 1e10
        
        # Wait, I messed up the hexagons_overlap call. Let me fix.
        # Actually let me rewrite this more carefully.
        return 1e10
    
    # Let me restructure with a cleaner approach
    
    def evaluate_config(centers, angles, outer_angle_deg):
        """Evaluate a configuration. Returns (valid, outer_side)."""
        if time.time() > deadline:
            return False, 1e10
        
        # Check all pairs for overlap
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(centers[i], 0, angles[i], 0, 0, 0):
                    pass
                # Need proper 2D check
                pass
        
        # Let me just do a simple bounding box check for now and refine
        # Actually, let me use a simpler overlap check based on center distance
        # For two unit hexagons, if centers are > 2 apart, they don't overlap
        # If centers are < sqrt(3) apart, they definitely overlap (for aligned hexagons)
        # For arbitrary orientations, it's more complex
        
        # Simple conservative check: if center distance < 1.5, assume overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1.4:
                    return False, 1e10
        
        outer_side = required_outer_side(centers, angles, outer_angle_deg)
        return True, outer_side
    
    # Initial symmetric configuration: 3-4-3 arrangement
    # Rows of hexagons at different y positions
    # For hexagonal packing, row spacing = sqrt(3)
    # Within a row, spacing = 2
    
    def make_symmetric_config(scale=1.0):
        """Create a symmetric 3-4-3 configuration."""
        row_spacing = math.sqrt(3) * scale
        col_spacing = 2.0 * scale
        
        centers = []
        angles = []
        
        # Row 0 (top): 3 hexagons
        y0 = row_spacing
        for i in range(3):
            x = (i - 1) * col_spacing
            centers.append((x, y0))
            angles.append(0.0)
        
        # Row 1 (middle): 4 hexagons
        y1 = 0.0
        for i in range(4):
            x = (i - 1.5) * col_spacing
            centers.append((x, y1))
            angles.append(0.0)
        
        # Row 2 (bottom): 3 hexagons
        y2 = -row_spacing
        for i in range(3):
            x = (i - 1) * col_spacing
            centers.append((x, y2))
            angles.append(0.0)
        
        return centers, angles
    
    # Start with a reasonable configuration
    centers0, angles0 = make_symmetric_config(1.0)
    
    # Flatten for optimization
    def flatten(centers, angles):
        params = []
        for (cx, cy), a in zip(centers, angles):
            params.extend([cx, cy, a])
        return params
    
    def unflatten(params):
        centers = []
        angles = []
        for i in range(N):
            centers.append((params[3*i], params[3*i+1]))
            angles.append(params[3*i+2])
        return centers, angles
    
    # Objective function for optimization
    def obj_func(params):
        if time.time() > deadline:
            return 1e10
        
        centers, angles = unflatten(params)
        
        # Check overlaps with a simple distance check
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1.8:  # Conservative threshold
                    return 1e10
        
        # Compute required outer side
        outer_side = required_outer_side(centers, angles, 0.0)
        return outer_side
    
    # Try differential evolution
    x0 = flatten(centers0, angles0)
    bounds = [(-5, 5)] * (3 * N)
    
    best_side = float('inf')
    best_params = x0.copy()
    
    try:
        result = differential_evolution(
            obj_func,
            bounds,
            maxiter=50,
            popsize=15,
            tol=0.01,
            seed=42,
            polish=True,
            time_limit=100
        )
        if result.fun < best_side:
            best_side = result.fun
            best_params = result.x
    except Exception as e:
        pass
    
    # Also try local refinement from the initial config
    try:
        result2 = minimize(
            obj_func,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 0.01, 'fatol': 0.01}
        )
        if result2.fun < best_side:
            best_side = result2.fun
            best_params = result2.x
    except Exception as e:
        pass
    
    # Use best params found
    centers_best, angles_best = unflatten(best_params)
    
    # Verify and compute final outer side
    final_valid = True
    for i in range(N):
        for j in range(i+1, N):
            dx = centers_best[i][0] - centers_best[j][0]
            dy = centers_best[i][1] - centers_best[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1.8:
                final_valid = False
                break
        if not final_valid:
            break
    
    if not final_valid:
        # Fall back to initial config
        centers_best, angles_best = centers0, angles0
    
    outer_side = required_outer_side(centers_best, angles_best, 0.0)
    
    # Build inner list
    inner = []
    for (cx, cy), a in zip(centers_best, angles_best):
        inner.append([cx, cy, a])
    
    return inner, (0.0, 0.0), outer_side, 0.0
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def get_hex_vertices(cx, cy, side, angle_deg):
        """Return vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside regular hexagon centered at (cx,cy)."""
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a

        inradius = side * math.sqrt(3) / 2.0

        for i in range(6):
            normal_angle = math.radians(30 + i * 60)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            dist = lx * nx + ly * ny
            if dist > inradius + 1e-9:
                return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def hexagons_overlap(cx1, cy1, a1, cx2, cy2, a2):
        """Check if two unit hexagons overlap. Returns True if they overlap."""
        dx = cx1 - cx2
        dy = cy1 - cy2
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 2.0 + 1e-9:
            return False
        if dist < 1e-6:
            return True

        verts1 = get_hex_vertices(cx1, cy1, 1.0, a1)
        verts2 = get_hex_vertices(cx2, cy2, 1.0, a2)

        # Check if any vertex of hex1 is inside hex2
        for v in verts1:
            if point_in_hexagon(v[0], v[1], cx2, cy2, 1.0, a2):
                return True
        # Check if any vertex of hex2 is inside hex1
        for v in verts2:
            if point_in_hexagon(v[0], v[1], cx1, cy1, 1.0, a1):
                return True

        # Check edge intersections
        for i in range(6):
            e1 = (verts1[i], verts1[(i+1)%6])
            for j in range(6):
                e2 = (verts2[j], verts2[(j+1)%6])
                if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                    return True
        return False

    def required_outer_side(centers, angles, outer_angle_deg):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        max_side = 0.0
        outer_angle_rad = math.radians(outer_angle_deg)
        
        for idx in range(len(centers)):
            cx, cy = centers[idx]
            a = angles[idx]
            verts = get_hex_vertices(cx, cy, 1.0, a)
            for (vx, vy) in verts:
                r = math.sqrt(vx*vx + vy*vy)
                if r == 0:
                    theta = 0
                else:
                    theta = math.atan2(vy, vx)
                
                # Find the constraint from the outer hexagon edges
                # The outer hexagon has edges with normals at angles: outer_angle + 30 + 60*k
                # For a point at angle theta and distance r to be inside,
                # it must satisfy r * cos(theta - normal_angle) <= inradius_outer
                # where inradius_outer = S * sqrt(3)/2
                # So S >= r * cos(theta - normal_angle) / (sqrt(3)/2) for the binding constraint
                # But we need the MAX over all edges? No, the point is inside if it satisfies ALL edge inequalities.
                # The binding constraint is the one where the "margin" is smallest.
                # Actually, for a convex polygon, a point is inside if it is on the "inside" of all edges.
                # The distance from center to the edge along the normal is inradius.
                # The projection of the point onto the normal is r * cos(theta - normal_angle).
                # We need r * cos(theta - normal_angle) <= inradius for ALL edges.
                # This implies r * max(cos(theta - normal_angle)) <= inradius ? No.
                # We need r * cos(theta - normal_angle) <= S * sqrt(3)/2 for all k.
                # This is equivalent to S >= max_k (r * cos(theta - normal_angle) / (sqrt(3)/2))
                # BUT only for cos > 0? If cos < 0, the point is on the "back" side of that edge normal,
                # which is automatically satisfied if it's inside the other edges?
                # Actually, for a regular hexagon centered at origin, the condition is:
                # |r * cos(theta - normal_k)| <= inradius is NOT correct.
                # The half-plane defined by edge k is: x * nx_k + y * ny_k <= inradius.
                # So r * cos(theta - normal_k) <= inradius.
                # Since the point must satisfy this for ALL 6 edges,
                # S * sqrt(3)/2 >= max_k (r * cos(theta - normal_k))
                # Note: cos can be negative, but the max will be positive if the point is in the hexagon.
                
                max_proj = -float('inf')
                for k in range(6):
                    normal_angle = outer_angle_rad + math.radians(30 + k * 60)
                    proj = r * math.cos(theta - normal_angle)
                    if proj > max_proj:
                        max_proj = proj
                
                if max_proj > 0:
                    required_S = max_proj / (math.sqrt(3)/2.0)
                    if required_S > max_side:
                        max_side = required_S
        
        return max_side

    def evaluate_config(centers, angles, outer_angle_deg):
        """Evaluate a configuration. Returns (valid, outer_side)."""
        if time.time() > deadline:
            return False, float('inf')
        
        # Check all pairs for overlap
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(centers[i][0], centers[i][1], angles[i], 
                                    centers[j][0], centers[j][1], angles[j]):
                    return False, float('inf')
        
        outer_side = required_outer_side(centers, angles, outer_angle_deg)
        return True, outer_side

    def make_symmetric_config():
        """Create a symmetric 3-4-3 configuration."""
        row_spacing = math.sqrt(3)
        col_spacing = 2.0
        
        centers = []
        angles = []
        
        # Row 0 (top): 3 hexagons
        y0 = row_spacing
        for i in range(3):
            x = (i - 1) * col_spacing
            centers.append((x, y0))
            angles.append(0.0)
        
        # Row 1 (middle): 4 hexagons
        y1 = 0.0
        for i in range(4):
            x = (i - 1.5) * col_spacing
            centers.append((x, y1))
            angles.append(0.0)
        
        # Row 2 (bottom): 3 hexagons
        y2 = -row_spacing
        for i in range(3):
            x = (i - 1) * col_spacing
            centers.append((x, y2))
            angles.append(0.0)
        
        return centers, angles

    # Initial configuration
    centers0, angles0 = make_symmetric_config()
    
    # Verify initial config
    valid0, side0 = evaluate_config(centers0, angles0, 0.0)
    if not valid0:
        # If initial is invalid (shouldn't be), try a looser config
        # But 3-4-3 with standard spacing should be valid.
        # Let's just proceed, but if invalid, we return a very large side.
        pass

    best_side = side0 if valid0 else float('inf')
    best_centers = centers0.copy()
    best_angles = angles0.copy()

    # Optimization using Nelder-Mead on the flattened parameters
    # We optimize the positions only, keeping angles fixed at 0 for simplicity in this structural change?
    # Or we can optimize angles too. Let's optimize positions and angles.
    
    def flatten(centers, angles):
        params = []
        for i in range(N):
            params.extend([centers[i][0], centers[i][1], angles[i]])
        return params

    def unflatten(params):
        centers = []
        angles = []
        for i in range(N):
            centers.append((params[3*i], params[3*i+1]))
            angles.append(params[3*i+2])
        return centers, angles

    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        centers, angles = unflatten(params)
        valid, side = evaluate_config(centers, angles, 0.0)
        if not valid:
            return 1e10
        return side

    x0 = flatten(centers0, angles0)
    
    # Try a few random starts + local optimization
    import random
    random.seed(42)
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        if r == 0:
            current_params = x0
        else:
            # Perturb initial config
            current_params = x0.copy()
            for i in range(len(current_params)):
                if i % 3 != 2: # Don't perturb angles too much initially
                    current_params[i] += random.uniform(-0.5, 0.5)
                else:
                    current_params[i] += random.uniform(-15, 15)
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective, current_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 0.001, 'fatol': 0.001})
            if res.fun < best_side:
                best_side = res.fun
                best_centers, best_angles = unflatten(res.x)
        except Exception:
            pass

    # Final verification
    valid_final, side_final = evaluate_config(best_centers, best_angles, 0.0)
    if not valid_final:
        # Fallback to initial
        best_centers, best_angles = centers0, angles0
        side_final = side0

    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])
    
    return inner, (0.0, 0.0), side_final, 0.0
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
