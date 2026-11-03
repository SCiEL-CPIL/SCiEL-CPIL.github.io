"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    
    # Helper functions
    def get_hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon with side length 'side' centered at (cx, cy).
        Angle 0 means one vertex points right (pointy-top in some conventions, but here
        we define angle as rotation from x-axis).
        Standard hexagon vertices at angle a: a + i*pi/3 for i=0..5
        """
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        # Rotate point relative to center by -angle
        rel_x = px - cx
        rel_y = py - cy
        cos_a = math.cos(-math.radians(angle_deg))
        sin_a = math.sin(-math.radians(angle_deg))
        rx = rel_x * cos_a - rel_y * sin_a
        ry = rel_x * sin_a + rel_y * cos_a
        
        # For a regular hexagon centered at origin with "pointy-top" or "flat-top"?
        # Vertices are at angle 0, 60, 120...
        # The hexagon is the intersection of 6 half-planes.
        # The normal to the side between vertex i and i+1 is at angle (i + 0.5)*60 deg.
        # Let's use the distance to sides.
        # Side length is 'side'. Distance from center to side (apothem) is side * sqrt(3)/2.
        # The normals to the sides are at angles: 30, 90, 150, 210, 270, 330 degrees.
        # Wait, if vertices are at 0, 60, 120..., the sides are perpendicular to angles 30, 90, 150...
        # Let's check: Vertex 0 (1,0), Vertex 1 (0.5, sqrt(3)/2).
        # Midpoint (0.75, sqrt(3)/4). Vector from center to mid is (0.75, 0.433).
        # Angle is atan2(0.433, 0.75) = 30 deg. Correct.
        
        apothem = side * math.sqrt(3) / 2.0
        for i in range(6):
            theta = math.radians(30 + i * 60)
            nx = math.cos(theta)
            ny = math.sin(theta)
            dist = rx * nx + ry * ny
            if dist > apothem:
                return False
        return True

    def hexagons_disjoint(c1, c2, a1, a2, side=1.0, eps=1e-9):
        """Check if two hexagons are disjoint (interiors don't overlap).
        They can touch.
        We check if any vertex of hex1 is strictly inside hex2, or vice versa.
        Also check if edges cross? For convex polygons, if no vertex is inside and they don't overlap, they are disjoint or touching.
        Actually, if two convex polygons are disjoint, no vertex of one is inside the other.
        If they overlap, either a vertex is inside, or edges cross.
        However, for hexagons on a lattice, they often touch at sides or vertices.
        A robust check: 
        1. Check if any vertex of A is inside B (including boundary? No, 'disjoint' usually means interiors disjoint. So if vertex is on boundary, it's ok. If strictly inside, not ok).
        2. Check if any vertex of B is inside A.
        3. Check edge intersections.
        
        Let's stick to: if dist(center1, center2) < 2 * apothem? No, that's not sufficient.
        
        Better: 
        Compute vertices of both.
        Check if any vertex of A is strictly inside B.
        Check if any vertex of B is strictly inside A.
        If neither, and they are not intersecting... 
        Actually, for convex polygons, if the interiors are disjoint, the set of points in A intersect B is either empty or on the boundary.
        So checking vertices strictly inside is a necessary condition for intersection.
        Is it sufficient? No. Two squares can cross without vertices inside.
        But for regular hexagons, is it possible?
        Yes, if they are rotated 30 degrees relative to each other, one vertex of one might stick into the other without being the "deep" vertex?
        Actually, if interiors overlap, either a vertex is inside, or an edge crosses.
        
        Let's implement a proper convex polygon intersection test or just use a simpler distance-based check if possible?
        No, rotation matters.
        
        Let's use a standard polygon intersection check.
        """
        v1 = get_hex_vertices(c1[0], c1[1], side, a1)
        v2 = get_hex_vertices(c2[0], c2[1], side, a2)
        
        # Check if any vertex of v1 is strictly inside v2
        for p in v1:
            if point_in_hexagon(p[0], p[1], c2[0], c2[1], side, a2):
                # It's inside. We need to ensure it's not just on the boundary.
                # Due to floating point, 'point_in_hexagon' returns True if dist <= apothem.
                # We want to fail if dist < apothem - eps.
                # Let's re-calculate distance to sides to check strictness.
                rel_x = p[0] - c2[0]
                rel_y = p[1] - c2[1]
                cos_a = math.cos(-math.radians(a2))
                sin_a = math.sin(-math.radians(a2))
                rx = rel_x * cos_a - rel_y * sin_a
                ry = rel_x * sin_a + rel_y * cos_a
                apothem = side * math.sqrt(3) / 2.0
                inside_strict = False
                for i in range(6):
                    theta = math.radians(30 + i * 60)
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    dist = rx * nx + ry * ny
                    if dist < apothem - 1e-8:
                        inside_strict = True
                        break
                if inside_strict:
                    return False
        
        # Check if any vertex of v2 is strictly inside v1
        for p in v2:
            if point_in_hexagon(p[0], p[1], c1[0], c1[1], side, a1):
                rel_x = p[0] - c1[0]
                rel_y = p[1] - c1[1]
                cos_a = math.cos(-math.radians(a1))
                sin_a = math.sin(-math.radians(a1))
                rx = rel_x * cos_a - rel_y * sin_a
                ry = rel_x * sin_a + rel_y * cos_a
                apothem = side * math.sqrt(3) / 2.0
                inside_strict = False
                for i in range(6):
                    theta = math.radians(30 + i * 60)
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    dist = rx * nx + ry * ny
                    if dist < apothem - 1e-8:
                        inside_strict = True
                        break
                if inside_strict:
                    return False
        
        # Check edge intersections
        def edges_intersect(p1, p2, p3, p4):
            # Check if segments p1p2 and p3p4 intersect
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
            for j in range(6):
                if edges_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                    return False
        
        return True

    def outer_side_length(centers, outer_center, outer_angle):
        """Calculate minimum side length of outer hexagon to contain all inner hexagons.
        Inner hexagons have side 1.
        Outer hexagon is centered at outer_center, rotated by outer_angle.
        We need the minimum 'S' such that all inner hexagons are inside the outer hexagon.
        
        An inner hexagon is inside if all its vertices are inside the outer hexagon.
        The outer hexagon is defined by S.
        The distance from the outer center to the nearest side of the outer hexagon (apothem) is S * sqrt(3)/2.
        The outer hexagon is the intersection of 6 half-planes.
        For a given vertex V of an inner hexagon, the distance from outer_center to V projected onto the normal of each side must be <= apothem.
        So, for each side normal direction n_i, the max projection of (V - outer_center) dot n_i over all V and all inner hexagons must be <= S * sqrt(3)/2.
        
        Therefore, S >= max_projection / (sqrt(3)/2).
        """
        max_proj = 0.0
        # Normals for outer hexagon sides
        # If outer hexagon has vertices at angle outer_angle + i*pi/3,
        # the side normals are at outer_angle + pi/6 + i*pi/3 (30 + 60i).
        # Let's compute the max projection for each of the 6 normals.
        normals = []
        base_angle = math.radians(outer_angle) + math.pi/6
        for i in range(6):
            theta = base_angle + i * math.pi/3
            normals.append((math.cos(theta), math.sin(theta)))
            
        # Generate all vertices of all inner hexagons
        all_vertices = []
        for c in centers:
            v = get_hex_vertices(c[0], c[1], 1.0, c[2])
            all_vertices.extend(v)
            
        for nx, ny in normals:
            for vx, vy in all_vertices:
                proj = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if proj > max_proj:
                    max_proj = proj
                    if proj < 0: # Just in case, but max_proj should be positive
                        pass
        
        # S * sqrt(3)/2 >= max_proj
        # S >= max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3.0)

    def objective(params):
        """Params: [cx1, cy1, a1, cx2, cy2, a2, ..., cx12, cy12, a12, ox, oy, o_angle]
        Total 12*3 + 3 = 39 params.
        """
        # Check time
        if time.time() > deadline:
            return 1e9
        
        centers = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i+1]
            a = params[3*i+2]
            centers.append((cx, cy, a))
        
        outer_center = (params[36], params[37])
        outer_angle = params[38]
        
        # Check disjointness
        # This is expensive. O(N^2).
        # We can add penalties for overlaps.
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Quick distance check
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Min distance between centers of disjoint hexagons (aligned) is 2*apothem = sqrt(3) ~ 1.732
                # If dist < sqrt(3) - 1e-6, they definitely overlap if aligned.
                # With rotation, min dist can be smaller? No, min distance between two convex bodies is at least the distance between centers minus sum of inradii?
                # Actually, if they overlap, dist < 2 * circumradius = 2.
                # Let's just use the boolean check if dist < 2.0
                if dist < 2.0:
                    if not hexagons_disjoint(centers[i], centers[j], centers[i][2], centers[j][2]):
                        penalty += 100.0 + (2.0 - dist) * 10.0
        
        if penalty > 0:
            # Add penalty to objective
            side = outer_side_length(centers, outer_center, outer_angle)
            return side + penalty
        
        side = outer_side_length(centers, outer_center, outer_angle)
        return side

    # Initial configuration: Hexagonal lattice
    # 1 center, 6 neighbors, 5 from next ring.
    # Lattice vectors for hexagons of side 1:
    # Distance between centers is sqrt(3).
    # Neighbors at distance sqrt(3) at angles 0, 60, 120...
    # Let's set up initial centers.
    
    # Center at (0,0)
    init_centers = []
    init_centers.append((0.0, 0.0, 0.0))
    
    r1 = math.sqrt(3)
    for i in range(6):
        angle = i * math.pi / 3
        x = r1 * math.cos(angle)
        y = r1 * math.sin(angle)
        init_centers.append((x, y, 0.0))
    
    # Next ring: distance 2 * sqrt(3)
    r2 = 2 * math.sqrt(3)
    # Add 5 points from the second ring.
    # The second ring vertices in the lattice are at angles 0, 60... with radius 2*sqrt(3)?
    # No, the neighbors of the first ring neighbors.
    # (sqrt(3), 0) has neighbors at (sqrt(3)+sqrt(3), 0) = (2sqrt(3), 0), (sqrt(3)+0.5sqrt(3), 1.5sqrt(3)/2? No.
    # Lattice points: (i*a + j*b_x, j*b_y)
    # Let's just pick 5 points at radius 2*sqrt(3) at angles 0, 60, 120, 180, 240.
    for i in range(5):
        angle = i * math.pi / 3
        x = r2 * math.cos(angle)
        y = r2 * math.sin(angle)
        init_centers.append((x, y, 0.0))
    
    # Flatten to params
    init_params = []
    for c in init_centers:
        init_params.extend([c[0], c[1], c[2]])
    
    # Outer center and angle
    # Centroid of init_centers is roughly (0,0)
    ox, oy = 0.0, 0.0
    o_angle = 0.0
    init_params.extend([ox, oy, o_angle])
    
    # Optimize
    best_params = init_params
    best_score = objective(init_params)
    
    # Use Nelder-Mead or L-BFGS-B?
    # L-BFGS-B is better for smooth-ish landscapes, but penalty is non-differentiable.
    # Nelder-Mead is robust to non-differentiability.
    
    try:
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        if time.time() < deadline - 5:
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
    except Exception as e:
        pass
    
    # If we have time, try a few random restarts or local refinements
    # But Nelder-Mead should be decent.
    
    # Extract best solution
    final_inner = []
    for i in range(N):
        cx = best_params[3*i]
        cy = best_params[3*i+1]
        a = best_params[3*i+2]
        final_inner.append([cx, cy, a])
        
    final_outer_center = (best_params[36], best_params[37])
    final_outer_angle = best_params[38]
    
    # Calculate final side length
    final_side = outer_side_length([(p[0], p[1], p[2]) for p in final_inner], final_outer_center, final_outer_angle)
    
    # Add small epsilon to side to ensure containment
    final_side += 1e-6
    
    return final_inner, final_outer_center, final_side, final_outer_angle
# EVOLVE-BLOCK-END
