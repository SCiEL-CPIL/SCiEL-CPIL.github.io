"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # We want to pack 12 unit regular hexagons inside a regular hexagon.
    # The seed program uses a linear chain, which requires a very large outer hexagon.
    # A much more compact arrangement is a hexagonal cluster.
    
    # A hexagonal cluster of 12 hexagons consists of:
    # - 1 hexagon at the center (ring 0)
    # - 6 hexagons in the first ring (ring 1)
    # - 5 hexagons in the second ring (ring 2) [one gap in the 6 positions]
    
    # Let's place them with side length 1, angle 0 (flat top/bottom or pointy top?).
    # Standard orientation for a regular hexagon with vertices at (1,0), (0.5, sqrt(3)/2), etc.
    # The distance between centers of adjacent touching hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    
    # Wait, the distance between centers of two adjacent unit hexagons is 2 * (sqrt(3)/2) = sqrt(3).
    # Let's verify: Apothem = sqrt(3)/2. So center-to-center distance is 2 * apothem = sqrt(3).
    
    # Let's define the centers in a hexagonal grid.
    # The neighbors of (0,0) in a hexagonal grid with side length 1 are at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300 degrees.
    # Let's use axial coordinates for simplicity, then convert to Cartesian.
    # Axial coordinates (q, r) -> Cartesian (x, y):
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # This is for pointy-top hexagons. Let's use flat-top.
    # Flat-top:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    # Let's just generate the Cartesian coordinates directly.
    # Center at (0,0).
    centers = [(0.0, 0.0)]
    
    # Ring 1: 6 neighbors at distance sqrt(3)
    d = math.sqrt(3)
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))
        
    # Ring 2: We need 5 more hexagons.
    # The ring 2 positions are at distance 2*sqrt(3) from the center, but only in the 6 directions?
    # No, the next ring in a hexagonal packing has 12 positions?
    # Let's visualize.
    #   . . .
    #  . X X .
    # . X O X .
    #  . X X .
    #   . . .
    # O is center. X are ring 1.
    # The next ring (ring 2) surrounds the 6 ring 1 hexagons.
    # There are 12 positions in ring 2.
    # We need to pick 5 of them such that the bounding hexagon is minimized.
    # Due to symmetry, we might want to leave one position empty to make the shape more circular?
    # Or fill 5 consecutive positions?
    # Let's try to fill positions that balance the shape.
    
    # Let's generate the 12 ring 2 positions.
    # They are at distance 2*sqrt(3) from the center?
    # No, the distance from center to ring 2 centers is not simply 2*sqrt(3).
    # In a hexagonal grid, the neighbors of a ring 1 hexagon are:
    # 1. The center (ring 0)
    # 2. 4 other ring 1 hexagons? No, ring 1 hexagons don't touch each other?
    # Wait, in a hexagonal packing, ring 1 hexagons DO touch each other.
    # Center (0,0). Neighbors at distance sqrt(3).
    # Distance between two neighbors (e.g., angle 0 and 60):
    # (sqrt(3), 0) and (sqrt(3)/2, 3/2).
    # dx = sqrt(3)/2, dy = 3/2.
    # dist^2 = 3/4 + 9/4 = 12/4 = 3. dist = sqrt(3).
    # So ring 1 hexagons touch each other.
    
    # The ring 2 positions are the neighbors of the ring 1 hexagons that are not in ring 0 or 1.
    # Let's list them.
    # For the hexagon at (sqrt(3), 0) [angle 0]:
    # Its neighbors are:
    # - (0,0) [ring 0]
    # - (sqrt(3)/2, 3/2) [ring 1, angle 60]
    # - (sqrt(3)/2, -3/2) [ring 1, angle 300]
    # - (3*sqrt(3)/2, 3/2) [ring 2]
    # - (3*sqrt(3)/2, -3/2) [ring 2]
    # - (sqrt(3), 3) [ring 2]? No, let's calculate properly.
    
    # Let's use the axial coordinate system to be sure.
    # Flat-top hexagons, side length 1.
    # Center at (0,0) -> (q,r) = (0,0).
    # Neighbors in axial coords (q,r): (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1).
    # Cartesian:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    # (0,0) -> (0, 0)
    # (1,0) -> (1.5, sqrt(3)/2)
    # (1,-1) -> (1.5, sqrt(3)*(-1+0.5)) = (1.5, -sqrt(3)/2)
    # (0,-1) -> (0, -sqrt(3))
    # (-1,0) -> (-1.5, -sqrt(3)/2)
    # (-1,1) -> (-1.5, sqrt(3)*(1-0.5)) = (-1.5, sqrt(3)/2)
    # (0,1) -> (0, sqrt(3))
    
    # These are the 6 ring 1 centers.
    # Now find ring 2 centers.
    # They are at distance 2 in axial coords?
    # The neighbors of (1,0) are:
    # (2,0), (2,-1), (1,-1)[ring 1], (0,-1)[ring 1], (0,1)[ring 1], (1,1)
    # So (2,0), (2,-1), (1,1) are new.
    # Similarly for other ring 1 centers.
    
    # Let's collect all ring 2 centers.
    ring2_coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Check if it's in ring 2
            # Distance in axial coords?
            # Max(|q|, |r|, |q+r|) ? No.
            # Let's just check if it's not in ring 0 or 1.
            pass
            
    # Easier: generate all points with max distance 2 from origin in axial grid.
    # Axial distance is max(|q|, |r|, |q+r|) / something?
    # Actually, let's just list the neighbors of ring 1 that aren't ring 0 or 1.
    
    ring1 = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    ring0 = [(0,0)]
    neighbors = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    
    ring2 = set()
    for q, r in ring1:
        for dq, dr in neighbors:
            nq, nr = q + dq, r + dr
            if (nq, nr) not in ring0 and (nq, nr) not in ring1:
                ring2.add((nq, nr))
                
    # ring2 should have 12 elements.
    # Let's convert to Cartesian.
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = math.sqrt(3) * (r + q/2)
        return (x, y)
        
    centers = [axial_to_cartesian(0,0)]
    for q, r in ring1:
        centers.append(axial_to_cartesian(q, r))
        
    # Now pick 5 from ring2.
    # To minimize the bounding hexagon, we should pick a symmetric subset.
    # The 12 ring 2 centers form a hexagon with a "hole" in the middle? No, they form the next ring.
    # If we pick 5, we leave 1 empty.
    # The empty one will be at a specific angle.
    # Let's pick 5 consecutive ones to keep the shape compact?
    # Or spread them out?
    # If we leave one empty, the bounding box might be slightly larger in that direction, but smaller on average?
    # Actually, if we leave one empty, the shape is no longer fully symmetric.
    # Maybe it's better to just put all 12 in ring 2? But we only need 5.
    # So we pick 5.
    # Let's pick the 5 that are "closest" to the center? They are all equidistant.
    # Let's pick 5 that form a symmetric pattern.
    # Since 5 is odd, we can't have full 6-fold symmetry.
    # We can have 1-fold symmetry (if we pick 5 consecutive) or something else.
    # Let's try picking 5 consecutive.
    
    ring2_list = list(ring2)
    # Sort by angle
    ring2_list.sort(key=lambda p: math.atan2(p[1], p[0]))
    
    # Pick 5 consecutive
    selected_ring2 = ring2_list[:5]
    
    for q, r in selected_ring2:
        centers.append(axial_to_cartesian(q, r))
        
    # Now we have 12 centers.
    # We need to find the minimum outer hexagon side length that contains all of them.
    # The outer hexagon should be centered at the centroid of the inner hexagons?
    # Or maybe at (0,0)?
    # Let's assume the outer hexagon is centered at (0,0) and has angle 0.
    # We need to find the minimum side length S such that all inner hexagons are inside.
    # An inner hexagon at (cx, cy) with angle 0 is inside the outer hexagon (center 0,0, side S, angle 0)
    # if the distance from (cx, cy) to the boundary is at least the apothem of the inner hexagon (sqrt(3)/2).
    # Actually, the condition is that the inner hexagon (which is a regular polygon) is inside the outer one.
    # For two concentric regular hexagons with the same orientation, the inner one is inside if its vertex is inside.
    # But they are not concentric.
    # The safe condition is: distance from center of inner hex to the boundary of outer hex >= apothem of inner hex.
    # This is sufficient but not necessary.
    # A better condition: the maximum distance from the center of the outer hex to any vertex of the inner hex must be <= S.
    # Wait, the outer hexagon is defined by S. The vertices of the outer hex are at distance S from the center.
    # The boundary is closer to the center than the vertices (at apothem distance).
    # So the condition is: for every vertex of every inner hexagon, it must be inside the outer hexagon.
    
    # Let's implement a function to check if a point is inside a regular hexagon.
    def point_in_hexagon(px, py, cx, cy, S, angle_deg):
        # Transform point to hexagon's local frame
        angle_rad = math.radians(angle_deg)
        dx = px - cx
        dy = py - cy
        # Rotate
        x = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        y = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        
        # Hexagon with side S, center (0,0), flat-top (angle 0)
        # Vertices: (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
        # Inequalities:
        # |x| <= S
        # |y| <= S*sqrt(3)/2
        # |x| + |y|/sqrt(3) <= S
        
        # Actually, the inequalities for a flat-top hexagon centered at origin with side S:
        # -S <= x <= S
        # -S*sqrt(3)/2 <= y <= S*sqrt(3)/2
        # -S*sqrt(3) <= sqrt(3)*x - y <= S*sqrt(3)  ? No.
        
        # Let's use the apothem distance.
        # The hexagon is the intersection of 6 half-planes.
        # The normals are at angles 0, 60, 120, 180, 240, 300 degrees.
        # The distance from origin to each side is apothem = S * sqrt(3) / 2.
        # So for a point (x, y) in the rotated frame:
        # dot((x,y), (cos(theta), sin(theta))) <= S * sqrt(3) / 2 for all 6 theta.
        
        apothem = S * math.sqrt(3) / 2
        for i in range(6):
            theta = i * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            if x * nx + y * ny > apothem + 1e-9:
                return False
        return True
        
    # We need to find the minimum S.
    # Let's binary search on S.
    # Lower bound: max distance from origin to any center + apothem of inner hex?
    # Actually, the lower bound is the maximum over all inner hexagons of the minimum S needed to contain that hexagon.
    
    def min_S_for_hexagon(cx, cy, angle_deg):
        # The minimum S such that the hexagon at (cx, cy) with angle angle_deg is inside a hexagon at (0,0) with angle 0.
        # This is the maximum distance from (0,0) to any vertex of the inner hexagon.
        # Wait, no. The outer hexagon is defined by its side length S.
        # The vertices of the outer hexagon are at distance S.
        # The sides are at distance S*sqrt(3)/2.
        # The inner hexagon must be inside the outer one.
        # The critical constraint is the sides, not the vertices, usually.
        # Actually, the inner hexagon is inside if all its vertices are inside.
        # Let's compute the max "support function" value of the inner hexagon in the 6 directions of the outer hexagon.
        
        # Inner hexagon vertices (relative to its center):
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        # Rotated by angle_deg, then translated by (cx, cy).
        
        # We need: for all 6 outer normals (angles 0, 60, ...),
        # max_{v in inner_vertices} dot(v, n_outer) <= S * sqrt(3) / 2
        
        # So S >= (2/sqrt(3)) * max_{i, j} dot(inner_vertex_j_rotated, n_outer_i) + (cx, cy) dot n_outer_i?
        # No, the inner hexagon is at (cx, cy).
        # The vertices are at (cx + vx, cy + vy).
        # dot((cx+vx, cy+vy), n) = cx*nx + cy*ny + vx*nx + vy*ny.
        
        max_val = 0
        for i in range(6):
            theta = i * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            
            # Max over inner vertices
            max_v = -1e9
            for j in range(6):
                vtheta = j * math.pi / 3 + math.radians(angle_deg)
                vx = math.cos(vtheta)
                vy = math.sin(vtheta)
                val = (cx + vx) * nx + (cy + vy) * ny
                max_v = max(max_v, val)
                
            max_val = max(max_val, max_v)
            
        S = max_val * 2 / math.sqrt(3)
        return S
        
    min_S = 0
    for cx, cy in centers:
        min_S = max(min_S, min_S_for_hexagon(cx, cy, 0))
        
    # Return the result
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    outer_center = (0.0, 0.0)
    outer_side = min_S
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
