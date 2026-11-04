"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # A unit regular hexagon has vertices at distance 1 from center (circumradius = side length = 1)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_hex = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)
    ])
    
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return base_hex @ R.T + np.array([cx, cy])
    
    def min_dist_hexagons(v1, v2):
        """Compute minimum distance between two convex hexagons (6x2 arrays).
        Returns 0 if they overlap or touch, positive if separated."""
        # Check if any vertex of v1 is inside v2, or vice versa
        # For convex polygons, we can use separating axis theorem
        # But simpler: compute min distance between all edge pairs
        # Edges of polygon with vertices v: v[i] to v[(i+1)%6]
        
        def point_in_convex_poly(p, poly):
            """Check if point p is inside convex polygon poly (6 vertices)."""
            n = len(poly)
            for i in range(n):
                a = poly[i]
                b = poly[(i+1) % n]
                # Cross product to check if p is on the same side
                cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                if cross < -1e-10:
                    return False
            return True
        
        # Check if any vertex of v1 is inside v2
        for v in v1:
            if point_in_convex_poly(v, v2):
                return 0.0
        # Check if any vertex of v2 is inside v1
        for v in v2:
            if point_in_convex_poly(v, v1):
                return 0.0
        
        # Compute min distance between all edge pairs
        min_d = float('inf')
        for i in range(6):
            a1 = v1[i]
            b1 = v1[(i+1) % 6]
            for j in range(6):
                a2 = v2[j]
                b2 = v2[(j+1) % 6]
                # Min distance between segment a1-b1 and a2-b2
                # Parametrize: p1(t) = a1 + t*(b1-a1), t in [0,1]
                #               p2(s) = a2 + s*(b2-a2), s in [0,1]
                # Minimize |p1(t) - p2(s)|^2
                d1 = b1 - a1
                d2 = b2 - a2
                r = a1 - a2
                a = d1 @ d1
                b = d1 @ d2
                c = d2 @ d2
                d = d1 @ r
                e = d2 @ r
                denom = a*c - b*b
                if abs(denom) < 1e-12:
                    s_num = 0.0
                    t_num = 0.0
                else:
                    s_num = b*e - c*d
                    t_num = a*e - b*d
                    if s_num < 0:
                        s_num = 0
                    elif s_num > c:
                        s_num = c
                    if t_num < 0:
                        t_num = 0
                    elif t_num > a:
                        t_num = a
                s = s_num / c if c > 1e-12 else 0
                t = t_num / a if a > 1e-12 else 0
                # Clamp to [0,1]
                s = max(0, min(1, s))
                t = max(0, min(1, t))
                p1 = a1 + t * d1
                p2 = a2 + s * d2
                dist = np.linalg.norm(p1 - p2)
                if dist < min_d:
                    min_d = dist
        return min_d
    
    def outer_hex_contains_point(p, center, side, angle_deg):
        """Check if point p is inside the outer regular hexagon.
        Outer hexagon has center, side length, and rotation angle_deg.
        A regular hexagon with side length S has apothem = S * sqrt(3)/2.
        In its local frame, it contains points where for each of the 6 directions,
        the projection is <= apothem."""
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        # Rotate point to outer hexagon's local frame
        rel = p - np.array(center)
        local_x = c * rel[0] + s * rel[1]
        local_y = -s * rel[0] + c * rel[1]
        
        apothem = side * math.sqrt(3) / 2
        # The six normals to the edges of a regular hexagon (in local frame)
        # Edge normals at angles 0, 60, 120, 180, 240, 300
        for k in range(6):
            theta = math.radians(60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            proj = local_x * nx + local_y * ny
            if proj > apothem + 1e-10:
                return False
        return True
    
    def max_violation(inner_positions, center, side, angle_deg):
        """Return the maximum amount by which any inner hexagon vertex
        exceeds the outer hexagon boundary. 0 if all contained."""
        cx, cy = center
        max_viol = 0.0
        for pos in inner_positions:
            px, py, ang = pos
            verts = hex_vertices(px, py, ang)
            for v in verts:
                if not outer_hex_contains_point(v, (cx, cy), side, angle_deg):
                    # Compute how far outside
                    angle_rad = math.radians(angle_deg)
                    c, s = math.cos(angle_rad), math.sin(angle_rad)
                    rel = v - np.array([cx, cy])
                    local_x = c * rel[0] + s * rel[1]
                    local_y = -s * rel[0] + c * rel[1]
                    apothem = side * math.sqrt(3) / 2
                    for k in range(6):
                        theta = math.radians(60 * k)
                        nx = math.cos(theta)
                        ny = math.sin(theta)
                        proj = local_x * nx + local_y * ny
                        viol = proj - apothem
                        if viol > max_viol:
                            max_viol = viol
        return max_viol
    
    def objective(params):
        """
        params: [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11, cx,cy, side, angle]
        Total length: 12*3 + 4 = 40
        """
        inner_positions = []
        for i in range(12):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_positions.append([x, y, a])
        cx = params[36]
        cy = params[37]
        side = params[38]
        angle = params[39]
        
        if side < 1.0:
            return 1e6
        
        # Compute all hexagon vertices
        all_verts = []
        for pos in inner_positions:
            all_verts.append(hex_vertices(pos[0], pos[1], pos[2]))
        
        # Compute pairwise minimum distances and penalty for overlap
        overlap_penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                d = min_dist_hexagons(all_verts[i], all_verts[j])
                if d < 0.01:
                    # Smooth penalty: use (0.01 - d)^2 if overlapping
                    if d < 0.01:
                        overlap_penalty += (0.01 - d) ** 2
        
        # Max violation of outer boundary
        viol = max_violation(inner_positions, (cx, cy), side, angle)
        
        # Objective: minimize side length, penalize violations and overlaps
        # We want to minimize side, but if there's violation, we must increase side or move inner
        # Use: side + large_penalty * (max_viol) + overlap_penalty
        # Actually, we want to find the minimal side such that all hexagons fit.
        # Reformulate: minimize side, subject to no overlap and no boundary violation.
        # Using penalty method:
        penalty = 100.0 * max(0, viol) + 10.0 * overlap_penalty
        
        return side + penalty
    
    # Initial guess: arrange hexagons in a compact pattern
    # Start with a 2x3 grid or similar compact arrangement
    # For 12 hexagons, a 3x4 grid of centers spaced ~2 apart
    # Better: use a hexagonal packing pattern
    # Let's start with a reasonable compact arrangement
    
    # Hexagonal lattice positions for 12 hexagons
    # Row 0: 4 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 4 hexagons
    # Row 3: 1 hexagon? No, let's do 3 rows of 4
    # Actually, for hexagonal packing:
    # Row 0: x = -1.5, -0.5, 0.5, 1.5 (4 hexagons), y = 0
    # Row 1: x = -1.0, 0.0, 1.0 (3 hexagons), y = sqrt(3)
    # Row 2: x = -1.5, -0.5, 0.5, 1.5 (4 hexagons), y = 2*sqrt(3)
    # Row 3: x = -0.5, 0.5 (2 hexagons)? That's 4+3+4+2=13, too many.
    # Let's do:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 4 hexagons
    # With hexagonal offset
    
    sqrt3 = math.sqrt(3)
    init_params = []
    # 3 rows of 4, hexagonally packed
    for row in range(3):
        y = row * sqrt3
        for col in range(4):
            x = col * 2.0 + (0.0 if row % 2 == 0 else 1.0) - 3.0
            init_params.extend([x, y, 0.0])
    
    # Center and outer hexagon
    init_params.extend([0.0, sqrt3, 8.0, 0.0])
    
    # Optimize
    result = minimize(objective, init_params, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    # Extract results
    best_params = result.x
    inner = []
    for i in range(12):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    cx = best_params[36]
    cy = best_params[37]
    side = best_params[38]
    angle = best_params[39]
    
    # Verify and possibly refine side to be just enough
    # Check if current configuration is valid
    if max_violation(inner, (cx, cy), side, angle) < 1e-6:
        # Try to shrink side slightly
        for _ in range(20):
            new_side = side * 0.99
            if max_violation(inner, (cx, cy), new_side, angle) < 1e-6:
                side = new_side
            else:
                break
    
    return inner, (cx, cy), side, angle
# EVOLVE-BLOCK-END
