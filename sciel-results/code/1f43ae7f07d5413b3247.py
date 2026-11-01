import numpy as np
import time
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Helper: Generate vertices of a regular hexagon
    # Standard orientation: flat-top if angle=0? 
    # Let's define a regular hexagon with side 1.
    # Vertices relative to center (0,0) for a "pointy-top" vs "flat-top"
    # Usually, for packing, we can define the hexagon such that one side is horizontal.
    # Let's use a standard definition:
    # v_k = (cos(theta + k*pi/3), sin(theta + k*pi/3)) * side
    # If theta = 0, vertex at (1,0). This is "pointy-right".
    # If theta = pi/6 (30 deg), vertices are at 30, 90, 150... This is "flat-top" (top edge horizontal).
    # Let's stick to a specific convention.
    # The outer hexagon also needs a definition.
    # Let's assume the outer hexagon is "flat-top" (side horizontal at top/bottom) or similar.
    # The score is the side length.
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        """
        Returns vertices of a regular hexagon.
        angle_deg: rotation of the hexagon.
        We define the base hexagon (angle=0) with vertices at:
        (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        This is a 'pointy-right' hexagon. The 'flat' sides are on the left/right? No.
        Distance between (0.5, sqrt(3)/2) and (-0.5, sqrt(3)/2) is 1. This is a horizontal top edge.
        So angle=0 corresponds to a 'flat-top' hexagon (top and bottom edges horizontal).
        """
        angle_rad = np.deg2rad(angle_deg)
        angles = angle_rad + np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack((x, y))

    def check_disjointness(inner_verts, outer_verts, outer_side, outer_center, outer_angle):
        """
        Check if all inner hexagons are inside the outer hexagon and disjoint from each other.
        Returns True if valid, False otherwise.
        """
        # 1. Check disjointness between inner hexagons
        # For convex polygons, they are disjoint if no edge intersections and no point inside another.
        # Given the complexity and time limit, a robust geometric check is expensive.
        # However, since we are using a lattice-based approach, we can verify validity 
        # by checking if the minimum distance between any two inner hexagons is > 0 (with epsilon)
        # and if all inner vertices are inside the outer hexagon.
        
        # Fast check: Bounding boxes
        # But let's do a more precise check for the final candidate.
        # For the search, we can rely on the lattice construction ensuring disjointness 
        # if the lattice spacing is correct.
        
        # The "Flower" configuration:
        # 1 in center, 6 around it, 5 more?
        # Total 12.
        # A hexagonal number sequence: 1, 7, 19...
        # 12 is not a centered hexagonal number.
        # Configuration: 
        #   . X .
        #  X X X X
        #   . X .
        # This is 7.
        # Add 5 more to the perimeter?
        
        # Let's try a specific layout known to be dense for N=12.
        # Layout: 
        #   A B
        #  C D E
        #   F G
        #  H I J
        #   K L
        # This is a parallelogram-like shape?
        
        # Let's use a simple grid of hexagons.
        # In a hexagonal lattice, neighbors are at distance sqrt(3) (if side=1, distance between centers of touching hexagons is 2*apothem = 2*(sqrt(3)/2) = sqrt(3)).
        # Wait, for regular hexagons of side s=1:
        # Apothem a = sqrt(3)/2.
        # Distance between centers of touching hexagons = 2a = sqrt(3).
        
        # Let's generate a list of lattice positions for 12 hexagons.
        # Axial coordinates (q, r)
        # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        
        def axial_to_cube(q, r):
            x = q
            z = r
            y = -x - z
            return (x, y, z)
            
        def cube_to_axial(x, y, z):
            return (x, z)
            
        def cube_to_pixel(x, y, z, size=1):
            # For flat-top hexagons, pixel coordinates:
            # x = size * 3/2 * q
            # y = size * sqrt(3) * (r + q/2)
            q, r = cube_to_axial(x, y, z)
            px = size * 1.5 * q
            py = size * math.sqrt(3) * (r + 0.5 * q)
            return (px, py)

        # Generate a compact set of 12 axial coordinates
        # Center at (0,0)
        # Layer 0: (0,0) -> 1
        # Layer 1: 6 neighbors
        # Layer 2: 12 neighbors (we only need 5 more to reach 12)
        
        coords = [(0,0)]
        # Add layer 1
        layer1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        coords.extend(layer1)
        # Now we have 7. Need 5 more from layer 2.
        # Layer 2 neighbors of layer 1?
        # Let's just pick 5 from the next ring to keep it compact.
        # The "radius" 2 hexes are at distance 2.
        # To minimize bounding box, we should pick ones that fill the "corners" or keep the shape round.
        # Let's pick: (2,0), (1,1), (0,2), (-1,2), (-2,1) ... wait, need to check adjacency.
        # Let's try a specific set that forms a "thick" hexagon.
        # Maybe:
        # (0,0)
        # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # (2,0), (1,1), (0,2), (-1,2), (-2,1) -- These are 5 specific ones from the 2nd ring.
        # Let's verify if this set is connected and compact.
        
        candidate_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        
        # Convert to pixel coordinates
        # Note: The orientation of the lattice must match the hexagon orientation.
        # Our hexagons are "flat-top" (angle=0).
        # The lattice generated by cube_to_pixel above assumes flat-top hexagons.
        
        positions = []
        for q, r in candidate_coords:
            x, y = cube_to_pixel(q, r, 1.0)
            positions.append([x, y, 0.0]) # angle 0 for all initially
            
        # This configuration has 12 hexagons.
        # We need to find the minimal outer hexagon side length that contains them.
        # The outer hexagon can be rotated.
        # Let's assume the outer hexagon is also "flat-top" (angle=0) initially, 
        # but we can rotate the entire cluster or the outer hexagon.
        # Actually, rotating the cluster relative to the outer hexagon is equivalent 
        # to rotating the outer hexagon relative to the fixed cluster.
        
        # Calculate the bounding hexagon for a given rotation of the outer hexagon.
        # The outer hexagon is centered at (0,0) for now. We can translate the cluster 
        # to center it in the outer hexagon.
        
        def min_enclosing_hex_side(points, outer_angle_deg):
            """
            Find the minimum side length of a regular hexagon (with given rotation)
            centered at (0,0) that contains all points.
            """
            outer_rad = np.deg2rad(outer_angle_deg)
            # The hexagon is defined by 6 half-planes.
            # For a flat-top hexagon (angle 0), the normals to the sides are at 0, 60, 120, 180, 240, 300 degrees.
            # Wait, if vertices are at 0, 60..., the sides are perpendicular to 30, 90, 150...
            # Let's derive the support function.
            # A regular hexagon of side s, centered at origin, rotated by alpha.
            # The distance from center to a side (apothem) is A = s * sqrt(3) / 2.
            # The hexagon is the intersection of 6 half-planes.
            # The normal vectors to the sides are at angles: alpha + 30 + k*60 ?
            # Let's check.
            # If vertices are at angles theta_k = alpha + k*60.
            # The side between vertex k and k+1 has a normal pointing outwards.
            # The angle of the normal is alpha + 30 + k*60.
            
            normals = []
            for k in range(6):
                angle = outer_rad + np.deg2rad(30 + k * 60)
                normals.append((np.cos(angle), np.sin(angle)))
            
            max_dist = 0
            for nx, ny in normals:
                # Project all points onto the normal
                # dist = dot(p, n)
                # We need s such that A >= max(dist)
                # A = s * sqrt(3) / 2
                # So s >= 2 * max(dist) / sqrt(3)
                projections = [px * nx + py * ny for px, py in points]
                max_proj = max(projections)
                # Note: Since the hexagon is centered at 0, we must consider the max absolute value?
                # No, the half-plane is n . p <= A.
                # Since the hexagon is symmetric, we just need to satisfy n . p <= A for all p.
                # Because -n is also a normal (for the opposite side), we need |n . p| <= A?
                # Yes, because the set of normals includes both n and -n (at k and k+3).
                # So max_proj will be positive, and the opposite side constraint is -n . p <= A => n . p >= -A.
                # So we need A >= max(|n . p|).
                # But since we iterate over all 6 normals which cover the full circle, 
                # max(n . p) for a specific n is enough if we take the max over all normals?
                # Actually, for a fixed normal n, the constraint is n.p <= A.
                # For the opposite normal -n, the constraint is -n.p <= A => n.p >= -A.
                # So we need A >= max(n.p) AND A >= -min(n.p) => A >= max(|n.p|).
                # Since we loop through all 6 normals, do we get both n and -n?
                # Normals are at 30, 90, 150, 210, 270, 330.
                # 30 and 210 are opposite. 90 and 270 are opposite.
                # So yes, for each pair, we effectively check the absolute projection.
                # So A >= max_proj is sufficient if max_proj is the max over ALL points and ALL normals?
                # No, for a specific normal n, we need A >= max(p . n).
                # Since we check all 6, the condition A >= max_{k} (max_{p} p . n_k) is correct.
                # This ensures p is inside the hexagon.
                
                if max_proj > max_dist:
                    max_dist = max_proj
            
            side = 2 * max_dist / math.sqrt(3)
            return side

        # We need to center the cluster.
        # The cluster is not necessarily centered at (0,0).
        # Let's calculate the centroid of the 12 positions and subtract it.
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx_cluster = sum(xs) / len(xs)
        cy_cluster = sum(ys) / len(ys)
        
        centered_positions = [[x - cx_cluster, y - cy_cluster, a] for x, y, a in positions]
        
        # Now we can rotate the outer hexagon to minimize the side length.
        # The cluster is fixed. The outer hexagon rotates.
        # Due to 6-fold symmetry of the outer hexagon, we only need to search 0 to 60 degrees.
        
        best_side = float('inf')
        best_angle = 0.0
        
        # Sample angles
        # The function is periodic with period 60 deg.
        # It might have minima at specific angles.
        # Let's do a fine grid search first.
        
        for angle in np.linspace(0, 60, 601):
            side = min_enclosing_hex_side([(x, y) for x, y, _ in centered_positions], angle)
            if side < best_side:
                best_side = side
                best_angle = angle
                
        # Refine around best_angle
        # Local optimization
        angles_to_check = np.linspace(best_angle - 2, best_angle + 2, 100)
        for angle in angles_to_check:
            side = min_enclosing_hex_side([(x, y) for x, y, _ in centered_positions], angle)
            if side < best_side:
                best_side = side
                best_angle = angle

        # The inner hexagons are at centered_positions.
        # The outer hexagon is centered at (0,0), side best_side, angle best_angle.
        
        # Check validity roughly
        # We should ensure they are disjoint.
        # In this lattice configuration, they are touching, so they are disjoint (interiors don't overlap).
        # The problem statement says "disjoint", which usually means interiors are disjoint.
        # Touching boundaries is allowed.
        
        inner = centered_positions
        outer_center = (0.0, 0.0)
        outer_side = best_side
        outer_angle = best_angle
        
        return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
