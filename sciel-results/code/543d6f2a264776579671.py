# EVOLVE-BLOCK-START
import numpy as np
import time


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    t0 = time.time()
    deadline = 620.0  # safety margin before 630s

    # Hexagonal packing of 11 unit hexagons:
    # A "flower" pattern: 1 center + 6 ring + 4 more arranged optimally
    # Using axial coordinates for hex grid (pointy-top, side=1)
    # Distance between adjacent hex centers = sqrt(3) for pointy-top
    # 
    # Let's use a compact arrangement:
    # Row 0 (top):    3 hexagons
    # Row 1 (middle): 5 hexagons  
    # Row 2 (bottom): 3 hexagons
    # Total: 11
    #
    # For pointy-top hexagons with side s=1:
    # - horizontal spacing between columns: sqrt(3)
    # - vertical spacing between rows: 1.5
    # - odd rows offset by sqrt(3)/2
    #
    # Actually, let's use flat-top hexagons (angle=0 means vertex at angle 0)
    # For flat-top: vertices at 0, 60, 120, 180, 240, 300 degrees
    # Adjacent centers: horizontal = 1.5*s, vertical = sqrt(3)*s
    # 
    # Let me use a standard hex grid layout with flat-top orientation.
    # Center spacing: dx = 1.5, dy = sqrt(3)
    # For row r, column c:
    #   x = 1.5 * c
    #   y = sqrt(3) * (r + 0.5 * (c & 1))
    #
    # Let's place 11 hexagons in a compact cluster:
    # Using axial coordinates (q, r) where:
    #   x = 1.5 * q
    #   y = sqrt(3) * (r + 0.5 * q)
    #
    # 11 hexagons in a hexagonal shape of radius 2 would be 1+6+12=19 (too many)
    # Radius 1: 1+6 = 7 hexagons
    # So we need 7 + 4 more.
    
    # Let me try a different approach: 3-4-3-1 or similar compact shape
    # Or use the known optimal packing for 11 hexagons.
    
    # Compact arrangement: center + 6 surrounding + 4 in gaps
    # In hex grid with flat-top (side=1):
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3) from center (for flat-top, neighbors are at angles 0,60,...)
    # Actually for flat-top hexagons, the 6 neighbors are at:
    # (1.5, sqrt(3)/2), (0, sqrt(3)), (-1.5, sqrt(3)/2), (-1.5, -sqrt(3)/2), (0, -sqrt(3)), (1.5, -sqrt(3)/2)
    
    sqrt3 = np.sqrt(3.0)
    
    # 7 hexagons in hexagonal cluster (center + 6 neighbors)
    centers = np.array([
        [0.0, 0.0],
        [1.5, sqrt3/2],
        [0.0, sqrt3],
        [-1.5, sqrt3/2],
        [-1.5, -sqrt3/2],
        [0.0, -sqrt3],
        [1.5, -sqrt3/2],
    ])
    
    # Add 4 more hexagons. Place them at the "second ring" positions that are most compact.
    # Second ring positions (distance 2*sqrt(3) from center in various directions):
    # (3, 0), (1.5, 3*sqrt3/2), (-1.5, 3*sqrt3/2), (-3, 0), (-1.5, -3*sqrt3/2), (1.5, -3*sqrt3/2)
    # Pick 4 of these that minimize the bounding hexagon.
    
    # Let's add: (3, 0), (-3, 0), (1.5, 3*sqrt3/2), (-1.5, -3*sqrt3/2)
    # This gives a somewhat elongated shape. Let me try a more symmetric choice.
    
    # Try: (3, 0), (-3, 0), (0, 2*sqrt3), (0, -2*sqrt3) -- but (0, 2*sqrt3) is already close to existing
    # (0, sqrt3) is in ring 1, so (0, 2*sqrt3) would be at distance sqrt(3) from it - touching.
    
    # Let me just compute the minimal enclosing hexagon for the 7+4 configuration.
    # For now, use 4 additional at: (3,0), (-3,0), (1.5, 3*sqrt3/2), (-1.5, -3*sqrt3/2)
    
    extra = np.array([
        [3.0, 0.0],
        [-3.0, 0.0],
        [1.5, 3*sqrt3/2],
        [-1.5, -3*sqrt3/2],
    ])
    
    all_centers = np.vstack([centers, extra])
    
    # All hexagons have angle 0 (flat-top)
    inner_hex_data = np.column_stack([all_centers[:, 0], all_centers[:, 1], np.zeros(11)])
    
    # Compute the minimal regular hexagon (with same orientation as inner hexes, i.e., flat-top)
    # that contains all vertices of all inner hexagons.
    # 
    # A regular hexagon with center (cx, cy), side length s, angle a has vertices:
    #   (cx + s*cos(a + 60*i), cy + s*sin(a + 60*i)) for i=0..5
    # 
    # For a hexagon to contain a point p, p must be on the correct side of each edge.
    # The 6 edges correspond to 6 half-planes. For a regular hexagon, the constraint is:
    #   For each direction theta_k = a + 60*k (k=0..5), the projection of (p - center) onto 
    #   the outward normal of edge k must be <= s * cos(30 deg) = s * sqrt(3)/2
    #
    # Actually, for a regular hexagon with circumradius s (side length = circumradius for regular hex),
    # the inradius (apothem) is s * sqrt(3)/2.
    #
    # The 6 outward normals point in directions a + 30 + 60*k for k=0..5 (perpendicular to edges).
    # Wait, let me think more carefully.
    #
    # Vertices at angles a, a+60, ..., a+300.
    # Edge from vertex i to vertex i+1. The outward normal to this edge points in direction a + 30 + 60*i.
    # The distance from center to each edge (apothem) is s * sqrt(3)/2.
    #
    # So for a point p to be inside: for each k=0..5,
    #   dot(p - center, normal_k) <= s * sqrt(3)/2
    # where normal_k = (cos(a + 30 + 60*k), sin(a + 30 + 60*k))
    #
    # To find minimal s for a given center and angle:
    #   s = max over all points p and all k of [dot(p - center, normal_k) / (sqrt(3)/2)]
    #
    # We can optimize over center (cx, cy) and angle a.
    # 
    # For simplicity, let's fix angle = 0 (same as inner hexes) and optimize center.
    # Then try a few angles and pick the best.
    
    def compute_min_side(centers_arr, angle_deg):
        """Compute minimal side length of regular hexagon (given angle) containing all vertices of unit hexes at given centers."""
        a = np.radians(angle_deg)
        # 6 outward normal directions
        normals = np.array([
            [np.cos(a + np.radians(30 + 60*k)), np.sin(a + np.radians(30 + 60*k))]
            for k in range(6)
        ])  # shape (6, 2)
        
        # Generate all vertices of all inner hexagons
        # Each inner hex has 6 vertices: center + (cos(a_i + 60*j), sin(a_i + 60*j)) for j=0..5
        # All inner hexes have angle 0 in our setup
        all_vertices = []
        for cx, cy in centers_arr:
            for j in range(6):
                theta = np.radians(60 * j)
                all_vertices.append([cx + np.cos(theta), cy + np.sin(theta)])
        all_vertices = np.array(all_vertices)  # shape (66, 2)
        
        # For each vertex and each normal, compute dot product
        # dot = all_vertices @ normals.T  # shape (66, 6)
        # s_k = dot[:, k] / (sqrt(3)/2)
        # s = max over all
        
        dots = all_vertices @ normals.T  # shape (66, 6)
        s_values = dots / (sqrt3 / 2.0)
        return np.max(s_values)
    
    # Try several angles and centers to find the best packing
    best_side = float('inf')
    best_center = np.array([0.0, 0.0])
    best_angle = 0.0
    
    # Optimize over angle (0 to 30 degrees, since hexagon has 60-degree symmetry)
    # and center (small range around centroid of inner hexes)
    
    centroid = np.mean(all_centers, axis=0)
    
    # Try a grid of angles and centers
    angles_to_try = np.linspace(0, 30, 16)
    center_offsets = np.array([
        [0, 0],
        [0.5, 0], [-0.5, 0],
        [0, 0.5], [0, -0.5],
        [0.5, 0.5], [-0.5, -0.5],
        [0.5, -0.5], [-0.5, 0.5],
    ])
    
    for ang in angles_to_try:
        if time.time() > deadline:
            break
        for off in center_offsets:
            if time.time() > deadline:
                break
            # The center of the outer hexagon
            outer_center = centroid + off
            # Compute min side with this outer center and angle
            # We need to recompute: s = max over vertices of dot(v - outer_center, normal) / (sqrt(3)/2)
            a = np.radians(ang)
            normals = np.array([
                [np.cos(a + np.radians(30 + 60*k)), np.sin(a + np.radians(30 + 60*k))]
                for k in range(6)
            ])
            
            all_vertices = []
            for cx, cy in all_centers:
                for j in range(6):
                    theta = np.radians(60 * j)
                    all_vertices.append([cx + np.cos(theta), cy + np.sin(theta)])
            all_vertices = np.array(all_vertices)
            
            shifted = all_vertices - outer_center
            dots = shifted @ normals.T
            s_val = np.max(dots) / (sqrt3 / 2.0)
            
            if s_val < best_side:
                best_side = s_val
                best_center = outer_center.copy()
                best_angle = ang
    
    # Use the best configuration found
    outer_hex_data = np.array([best_center[0], best_center[1], best_angle])
    outer_hex_side_length = best_side
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
