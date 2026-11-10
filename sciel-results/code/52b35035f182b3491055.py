# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    using a hexagonal lattice arrangement and optimizing rotation to minimize the outer hexagon.
    """
    # Hexagonal lattice coordinates for 12 unit hexagons
    # Using axial coordinates (q, r) for hex grid, converted to pixel coordinates
    # For unit hexagons (side=1), the distance between centers is sqrt(3)
    # Axial to pixel: x = sqrt(3) * (q + r/2), y = 3/2 * r
    
    # 12 hexagons in a compact arrangement:
    # Center: (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3)
    # Ring 2: 5 hexagons (partial ring to make 12 total)
    
    # Let's use a known compact arrangement
    # Actually, let's think of it as a hexagon of hexagons
    # A hexagon of hexagons with "radius" 2 has 1+6+12 = 19, too many
    # With "radius" 1 has 1+6 = 7
    # So we need a different arrangement
    
    # Let's use: center + 6 in first ring + 5 in second ring (alternating positions)
    # Or better: a 3x4 rectangular-ish block in hex grid
    
    # Axial coordinates for a compact 12-hexagon shape
    # Let me use a hexagonal number pattern:
    # Positions in axial (q, r):
    coords_axial = [
        (0, 0),      # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (1, 1), (-1, 2), (-2, 1), (-2, 0),  # partial ring 2
    ]
    
    # Convert axial to Cartesian (for unit hexagons, side=1)
    # x = sqrt(3) * (q + r/2), y = 3/2 * r
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in coords_axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append([x, y])
    centers = np.array(centers)
    
    # Center the configuration
    centers -= centers.mean(axis=0)
    
    # All inner hexagons have angle 0 (flat-top or pointy-top? Let's use angle 0)
    # With angle 0, vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    
    def compute_outer_hex_side(angle_deg):
        """Compute the minimum side length of a regular hexagon (at given angle)
        that contains all 12 unit hexagons."""
        angle_rad = np.radians(angle_deg)
        
        # Rotate all centers by angle_rad
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_centers = centers @ rot_mat.T
        
        # For each inner hexagon, compute all 6 vertices
        # Vertex i of a unit hexagon at center c with angle a:
        # (cx + cos(a + 60i*pi/180), cy + sin(a + 60i*pi/180))
        # But we also need to rotate the hexagon shape itself
        
        # Actually, the hexagon orientation is fixed at angle_deg (all same)
        # So each inner hexagon has vertices at:
        # center + (cos(angle_deg + 60i), sin(angle_deg + 60i)) for i=0..5
        
        # Wait, the problem says the inner hexagons have their own angles.
        # In our lattice, all inner hexagons are axis-aligned (angle=0).
        # But we're rotating the whole configuration, so effectively all inner
        # hexagons get rotated by angle_rad.
        
        # Let me reconsider: the inner hexagons have angle 0 in the lattice frame.
        # When we rotate the entire configuration by angle_rad, each inner hexagon
        # is also rotated by angle_rad. So each inner hexagon's vertices are:
        # rotated_center + (cos(angle_rad + 60i*pi/180), sin(angle_rad + 60i*pi/180))
        
        # For a regular hexagon centered at origin with side s and angle theta,
        # a point p is inside if for each of the 6 edges, the point is on the
        # correct side.
        
        # The 6 vertices of the outer hexagon (centered at origin, side s, angle theta):
        # v_i = s * (cos(theta + 60i*pi/180), sin(theta + 60i*pi/180))
        
        # The 6 edge normals (pointing outward) are at angles theta + 60i*pi/180 + 30 degrees
        # Actually, for a regular hexagon, the edge normals are at angles theta + 30 + 60i degrees
        
        # Let me compute the required s such that all vertices of all inner hexagons
        # are inside the outer hexagon.
        
        # For a regular hexagon with center at origin, side s, and angle theta,
        # the distance from center to each edge (apothem) is s * cos(pi/6) = s * sqrt(3)/2
        # The edge normals are at angles theta + 30 + 60i degrees for i=0..5
        
        # A point p is inside if for all i, dot(p, n_i) <= s * sqrt(3)/2
        # where n_i = (cos(theta + 30 + 60i*pi/180), sin(theta + 30 + 60i*pi/180))
        
        # So we need s >= max over all points p and all i of dot(p, n_i) / (sqrt(3)/2)
        # = max over all points p and all i of dot(p, n_i) * 2/sqrt(3)
        
        # Collect all vertices of all inner hexagons
        all_vertices = []
        for c in rotated_centers:
            for i in range(6):
                vx = c[0] + np.cos(angle_rad + i * np.pi / 3)
                vy = c[1] + np.sin(angle_rad + i * np.pi / 3)
                all_vertices.append([vx, vy])
        all_vertices = np.array(all_vertices)
        
        # Compute required s
        max_proj = 0
        for i in range(6):
            normal_angle = angle_rad + np.pi / 6 + i * np.pi / 3  # theta + 30 + 60i
            n = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            projections = all_vertices @ n
            max_proj = max(max_proj, np.max(projections))
        
        # s * sqrt(3)/2 >= max_proj => s >= max_proj * 2/sqrt(3)
        s = max_proj * 2.0 / np.sqrt(3)
        return s
    
    # Optimize over angle
    result = minimize_scalar(compute_outer_hex_side, bounds=(0, 60), method='bounded')
    optimal_angle = result.x
    min_s = result.fun
    
    # Final configuration
    angle_rad = np.radians(optimal_angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    final_centers = centers @ rot_mat.T
    
    # Inner hexagons: (x, y, angle_deg)
    inner_hex_data = np.zeros((12, 3))
    for i, c in enumerate(final_centers):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = optimal_angle  # all same angle
    
    # Outer hexagon: centered at origin, angle = optimal_angle, side = min_s
    outer_hex_data = np.array([0.0, 0.0, optimal_angle])
    outer_hex_side_length = min_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
