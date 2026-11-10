# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    # Hexagonal lattice arrangement: center + 6 neighbors + 4 on outer ring
    # Using pointy-top hexagons (angle=0 means vertex at angle 0)
    # For unit hexagons, the lattice vectors are:
    #   v1 = (sqrt(3), 0)  - horizontal spacing
    #   v2 = (sqrt(3)/2, 1.5) - 60-degree direction
    
    sqrt3 = np.sqrt(3)
    
    # Center at origin, 6 neighbors, then 4 more on outer ring
    # Lattice positions (in units of hex spacing)
    # Hex i at lattice position (col, row) where:
    # x = sqrt(3) * (col + row/2)
    # y = 1.5 * row
    
    positions = []
    # Center
    positions.append((0, 0))
    # 6 neighbors
    for i in range(6):
        angle = np.radians(60 * i)
        positions.append((sqrt3 * np.cos(angle), 1.5 * np.sin(angle) * 2 / sqrt3 * sqrt3 / 1.5))
    
    # Let me use a cleaner approach with axial coordinates
    # Axial coords (q, r) -> cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    
    axial_positions = [
        (0, 0),      # center
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # 6 neighbors
        (2, 0), (2, -1), (1, -2), (0, 2),  # 4 more on outer ring
    ]
    
    inner_positions = []
    for q, r in axial_positions:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        inner_positions.append((x, y))
    
    inner_hex_data = np.array([[x, y, 0.0] for x, y in inner_positions])
    
    # Now find minimal enclosing regular hexagon
    # A regular hexagon with center (cx, cy), side s, angle a has vertices at:
    # (cx + s*cos(a + 60i), cy + s*sin(a + 60i)) for i=0..5
    # 
    # For a point to be inside, it must satisfy for each edge (from vertex i to i+1):
    # the cross product test
    #
    # Strategy: try various orientations and centers, minimize the required side length
    
    def get_hex_vertices(cx, cy, s, a_deg):
        a = np.radians(a_deg)
        angles = a + np.radians(60 * np.arange(6))
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    def is_point_in_hex(px, py, cx, cy, s, a_deg):
        """Check if point (px,py) is inside hexagon with center (cx,cy), side s, angle a_deg"""
        verts = get_hex_vertices(cx, cy, s, a_deg)
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            pt = np.array([px, py]) - p1
            cross = edge[0] * pt[1] - edge[1] * pt[0]
            if cross < -1e-9:
                return False
        return True
    
    def required_side_for_orientation(cx, cy, a_deg):
        """Find minimum side length s such that all inner hex vertices are inside the outer hex"""
        # Get all vertices of all inner hexagons
        all_verts = []
        for x, y, ang in inner_hex_data:
            a = np.radians(ang)
            angles = a + np.radians(60 * np.arange(6))
            verts = np.column_stack([x + np.cos(angles), y + np.sin(angles)])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # For a given orientation a_deg and center (cx, cy), find min s
        # such that all points are inside
        # For each point, the required s depends on the maximum "radius" in the hex metric
        # 
        # A point (px, py) is inside hex with center (cx,cy), side s, angle a iff:
        # for each of 6 edge normals n_i, the projection of (point - center) onto n_i <= s * cos(30deg) = s * sqrt(3)/2
        # Actually, for a regular hexagon with side s, the inradius is s*sqrt(3)/2
        # and the apothem (distance from center to edge) is s*sqrt(3)/2
        
        a = np.radians(a_deg)
        # Edge normals point inward
        # For hex with vertices at angles a+60i, edges are between consecutive vertices
        # The normal to edge i (from vertex i to i+1) points toward center
        
        # The distance from center to each edge is s * sqrt(3)/2
        # For a point to be inside, its distance to each edge (measured along the inward normal) must be <= s*sqrt(3)/2
        
        # Edge i is between vertices at angles a+60i and a+60(i+1)
        # The midpoint of edge i is at angle a+60i+30 from center, at distance s*sqrt(3)/2
        # The inward normal direction is from edge midpoint toward center, i.e., angle a+60i+30+180
        
        max_required_s = 0
        for px, py in all_verts:
            # Relative to center
            dx = px - cx
            dy = py - cy
            for i in range(6):
                # Inward normal angle for edge i
                normal_angle = a + np.radians(60 * i + 30 + 180)
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                # Projection of point onto inward normal
                proj = dx * nx + dy * ny
                # For point to be inside: proj <= s * sqrt(3)/2
                # So s >= proj / (sqrt(3)/2) = 2*proj/sqrt(3)
                if proj > max_required_s * sqrt3 / 2:
                    max_required_s = proj * 2 / sqrt3
        return max_required_s
    
    # Optimize over center and angle
    def objective(params):
        cx, cy, a_deg = params
        return required_side_for_orientation(cx, cy, a_deg)
    
    # Initial guess: center at centroid of inner hexagons, angle 0
    centroid_x = np.mean(inner_hex_data[:, 0])
    centroid_y = np.mean(inner_hex_data[:, 1])
    initial = [centroid_x, centroid_y, 0.0]
    
    result = minimize(objective, initial, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    cx, cy, a_deg = result.x
    side = required_side_for_orientation(cx, cy, a_deg)
    
    # Ensure side is at least sqrt(11)
    side = max(side, np.sqrt(11) + 1e-6)
    
    outer_hex_data = np.array([cx, cy, a_deg])
    
    return inner_hex_data, outer_hex_data, side


# EVOLVE-BLOCK-END
