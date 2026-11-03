import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # 150s limit, keep 5s margin
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon."""
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices
    
    def hexagon_radius(cx, cy, side, angle_deg, pt):
        """Compute distance from point to hexagon boundary (negative if inside)."""
        # Transform point to hexagon local coordinates
        dx = pt[0] - cx
        dy = pt[1] - cy
        angle = math.radians(angle_deg)
        ca = math.cos(angle)
        sa = math.sin(angle)
        # Rotate point into hexagon frame
        x = dx * ca + dy * sa
        y = -dx * sa + dy * ca
        
        # For a flat-top hexagon with side length s, the apothem is s*sqrt(3)/2
        # The hexagon can be defined as |x| <= s, |y| <= s*sqrt(3)/2, and 
        # appropriate inequalities for the slanted sides
        s = side
        apothem = s * math.sqrt(3) / 2
        
        # The hexagon vertices in local frame (flat-top) are at angles 0, 60, 120, 180, 240, 300
        # Vertex 0: (s, 0)
        # Vertex 1: (s/2, apothem)
        # Vertex 2: (-s/2, apothem)
        # Vertex 3: (-s, 0)
        # Vertex 4: (-s/2, -apothem)
        # Vertex 5: (s/2, -apothem)
        
        # The boundary is defined by 6 half-planes
        # We compute the minimum distance to any edge
        # Each edge is a line segment
        # For simplicity, we use the fact that for a regular hexagon, the distance from center
        # to boundary in direction theta is: r(theta) = apothem / cos((theta mod 60deg) - 30deg)
        
        # But we want signed distance: positive outside, negative inside
        # The radius in direction (x,y) is:
        theta = math.atan2(y, x)
        # Normalize theta to [0, 60)
        theta_mod = theta % (math.pi / 3)
        # The angle relative to the closest vertex direction
        # Vertex directions are at 0, 60, 120, ...
        # The apothem direction is at 30, 90, 150, ...
        # For theta in [0, 60), the closest apothem direction is at 30 degrees
        # The distance to boundary is: apothem / cos(theta_mod - pi/6)
        # But we need to handle the sign correctly
        
        # Actually, let's use a simpler approach: for each edge, compute distance
        # Edge 0: from vertex 0 to vertex 1
        # This is getting complex. Let's use a different approach.
        
        # For a regular hexagon, we can use the formula for the distance from a point
        # to a regular hexagon centered at origin.
        
        # The hexagon is the intersection of 6 half-planes.
        # Half-plane i: n_i . p <= apothem
        # where n_i are unit normals at angles 0, 60, 120, 180, 240, 300 degrees
        
        max_violation = 0
        for i in range(6):
            normal_angle = i * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            # Distance from point to the boundary plane
            dist = nx * x + ny * y - apothem
            if dist > max_violation:
                max_violation = dist
        
        return max_violation
    
    def hexagons_disjoint(inner):
        """Check if all inner hexagons are disjoint (non-overlapping)."""
        n = len(inner)
        for i in range(n):
            for j in range(i+1, n):
                cx1, cy1, a1 = inner[i]
                cx2, cy2, a2 = inner[j]
                # Check if two hexagons overlap
                # A simple sufficient condition: if distance between centers is less than 
                # the sum of circumradii, they might overlap
                # But this is not exact. Let's use a more precise check.
                
                # For two regular hexagons of side 1, they are disjoint if the distance
                # between centers is at least 2*side = 2? No, that's for circles.
                # For hexagons, the minimum distance for disjointness depends on orientation.
                # The closest points on two hexagons can be at distance less than 2 if they
                # are offset.
                
                # A better approach: check if any vertex of one is inside the other,
                # and check if any edge intersects.
                # But this is expensive. For our lattice-based approach, we know the
                # hexagons are on a honeycomb lattice with distance sqrt(3) between centers
                # for touching hexagons. So if distance >= sqrt(3) - epsilon, they are disjoint.
                
                d = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                # For two unit hexagons on a honeycomb lattice, touching means d = 2*side = 2? 
                # No, the distance between centers of two touching hexagons in a honeycomb
                # is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732
                # Wait, let me reconsider.
                # In a regular hexagonal tiling, the distance from center to center of adjacent
                # hexagons is 2 * apothem = 2 * (sqrt(3)/2) * side = sqrt(3) * side.
                # For side=1, this is sqrt(3) ≈ 1.732.
                # So if d < sqrt(3) - small_epsilon, they overlap.
                
                if d < math.sqrt(3) - 1e-9:
                    return False
        return True
    
    def compute_outer_side(inner, outer_center, outer_angle):
        """Compute the minimum outer hexagon side length to contain all inner hexagons."""
        cx, cy = outer_center
        
        # For each inner hexagon, compute the maximum distance from outer_center
        # to any point on the inner hexagon, projected onto each of the 6 directions
        # of the outer hexagon.
        
        # The outer hexagon is defined by 6 half-planes:
        # n_i . (p - c) <= side * apothem_factor
        # Actually, for a regular hexagon with side length L, the apothem is L*sqrt(3)/2.
        # The constraint is: for each direction i (0, 60, ..., 300 degrees),
        # max over all points p in inner hexagons of (n_i . (p - c)) <= L * apothem
        
        # So L >= max_i (max_p n_i . (p - c)) / apothem_factor
        # where apothem_factor = sqrt(3)/2
        
        max_proj = 0
        for i in range(6):
            normal_angle = math.radians(outer_angle) + i * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            max_for_direction = 0
            for j in range(len(inner)):
                icx, icy, iangle = inner[j]
                # Get vertices of inner hexagon
                verts = get_hex_vertices(icx, icy, 1.0, iangle)
                for vx, vy in verts:
                    proj = nx * (vx - cx) + ny * (vy - cy)
                    if proj > max_for_direction:
                        max_for_direction = proj
            
            if max_for_direction > max_proj:
                max_proj = max_for_direction
        
        apothem_factor = math.sqrt(3) / 2
        side = max_proj / apothem_factor
        return side
    
    # Initial configuration: honeycomb lattice
    # The 12 closest points to origin on a hexagonal lattice with spacing sqrt(3)
    # Ring 0: (0, 0)
    # Ring 1: 6 points at distance sqrt(3)
    # Ring 2: 12 points at distance sqrt(3)*sqrt(3) = 3? No.
    
    # Let me construct the lattice properly.
    # Hexagonal lattice basis:
    # a1 = (sqrt(3), 0)
    # a2 = (sqrt(3)/2, 3/2)
    # Wait, for unit hexagons, the lattice vectors are:
    # The centers form a triangular lattice with spacing 2*apothem = sqrt(3).
    # Basis vectors: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
    # |a2| = sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3). Good.
    
    a1 = (math.sqrt(3), 0)
    a2 = (math.sqrt(3)/2, 3/2)
    
    # Generate lattice points
    def lattice_point(i, j):
        x = i * a1[0] + j * a2[0]
        y = i * a1[1] + j * a2[1]
        return (x, y)
    
    # Collect points by distance from origin
    points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x, y = lattice_point(i, j)
            d = math.sqrt(x*x + y*y)
            points.append((d, i, j, x, y))
    
    points.sort(key=lambda p: p[0])
    
    # Take the 12 closest points
    selected = points[:N]
    inner = [[x, y, 0.0] for (_, _, _, x, y) in selected]
    
    # Verify disjointness
    if not hexagons_disjoint(inner):
        # Fallback: use a linear arrangement with correct spacing
        inner = [[math.sqrt(3) * k, 0.0, 0.0] for k in range(N)]
    
    # Compute initial outer side
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    best_side = compute_outer_side(inner, outer_center, outer_angle)
    
    # Try to optimize outer angle
    for trial in range(60):
        if time.time() > deadline:
            break
        angle = trial * math.pi / 60  # 3 degree steps
        side = compute_outer_side(inner, outer_center, angle)
        if side < best_side:
            best_side = side
            outer_angle = angle
    
    # Try some local optimization of outer center
    for dx in [-0.1, 0.0, 0.1]:
        for dy in [-0.1, 0.0, 0.1]:
            if time.time() > deadline:
                break
            oc = (dx, dy)
            side = compute_outer_side(inner, oc, outer_angle)
            if side < best_side:
                best_side = side
                outer_center = oc
    
    # Final check
    return inner, outer_center, best_side, math.degrees(outer_angle)
# EVOLVE-BLOCK-END
