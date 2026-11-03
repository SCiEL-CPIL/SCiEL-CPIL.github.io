import numpy as np
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Places N=11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    The inner hexagons are arranged in a 3x3 block plus 2 additional hexagons to form a compact shape.
    We search for the optimal rotation of the inner configuration to minimize the containing hexagon side length.
    """
    # Define the positions of the 11 unit hexagons in a compact cluster.
    # A 3x3 block of hexagons in a hexagonal grid has 9 hexagons.
    # We add 2 more to make it 11.
    # Let's use a hexagonal coordinate system (axial) for the inner hexagons.
    # The 3x3 block can be represented by axial coordinates (q, r) such that |q|<=1, |r|<=1, |q+r|<=1? No, that's a hex of radius 1 (7 hexes).
    # A 3x3 block is a parallelogram or a hexagon of radius 2 minus corners?
    # Let's just list the centers for a compact 11-hex shape.
    # Consider a hexagon of radius 2 (19 hexes). We remove 8 to leave 11.
    # Alternatively, take a 7-hex center and add 4 around it.
    
    # Let's use the axial coordinates for a "thick plus" shape or similar.
    # Centers of 7-hex cluster (radius 1):
    # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Add 4 more: (2,0), (0,2), (-2,2), (2,-2)? This might be too spread out.
    # Let's try a 3x3 grid in axial coordinates: q in [-1, 1], r in [-1, 1]. That's 9 hexes.
    # Add (2,0) and (-2,0)?
    
    # Let's define the 11 centers explicitly in Cartesian coordinates for a unit hexagon (side 1, distance between centers is sqrt(3)).
    # Wait, if side length is 1, the distance between adjacent centers is sqrt(3).
    # Let d = np.sqrt(3).
    
    d = np.sqrt(3.0)
    
    # 3x3 block of hexagons:
    # Row 0 (y=0): x = -d, 0, d
    # Row 1 (y=d*sqrt(3)/2): x = -d/2, d/2
    # Row 2 (y=d*sqrt(3)): x = -d, 0, d
    # That's 8 hexes.
    # Let's use a more standard hexagonal grid layout.
    # Axial coordinates (q, r) to Cartesian:
    # x = d * (q + r/2)
    # y = d * (sqrt(3)/2) * r
    
    # Let's pick 11 axial coordinates that form a compact shape.
    # A hexagon of radius 1 has 7 hexes.
    # Let's take the 7-hex cluster and add 4 more to extend it slightly.
    # Centers:
    # (0,0)
    # (1,0), (-1,0)
    # (0,1), (0,-1)
    # (1,-1), (-1,1)
    # This is the 7-hex cluster.
    # Add (2,0), (-2,0), (0,2), (0,-2)?
    # Let's try adding (2,0) and (-2,0).
    
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, 0), (-2, 0)
    ]
    
    # Convert to Cartesian
    centers = []
    for q, r in axial_coords:
        x = d * (q + r / 2.0)
        y = d * (np.sqrt(3.0) / 2.0) * r
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # The vertices of a unit regular hexagon centered at (cx, cy) with rotation theta:
    # The distance from center to vertex is 1 (since side length is 1, circumradius is 1).
    # Wait, for a regular hexagon, circumradius = side length. So R = 1.
    
    def get_vertices(theta):
        # Base angles for a hexagon with a vertex on the x-axis
        base_angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]) + theta
        verts = np.stack([np.cos(base_angles), np.sin(base_angles)], axis=1)
        return verts
    
    def hexagon_side_length(vertices, outer_center=(0,0), outer_rotation=0.0):
        # Find the minimum side length S of a regular hexagon centered at outer_center
        # with rotation outer_rotation that contains all vertices.
        # The constraint for a point (x,y) to be inside a regular hexagon of side S, center C, rotation phi:
        # For each of the 6 faces, the dot product of (point - C) with the face normal must be <= S * sqrt(3)/2 ?
        # Actually, the apothem of a hexagon with side S is S * sqrt(3)/2.
        # So for each face normal n_i, we need (point - C) . n_i <= S * sqrt(3)/2.
        # Thus S >= max_i [ (point - C) . n_i ] / (sqrt(3)/2).
        
        n_points = vertices.shape[0]
        # 6 normals for the hexagon
        base_normals = np.array([np.cos(np.pi/6 + i*np.pi/3), np.sin(np.pi/6 + i*np.pi/3)] for i in range(6)) + outer_rotation
        # Wait, the normals are perpendicular to the sides.
        # If vertices are at angles 0, pi/3, ..., then sides are between vertices.
        # The normal to the side between angle 0 and pi/3 is at angle pi/6.
        # Yes, base_normals above is correct.
        
        # Compute (point - C) . n_i for all points and all normals
        # vertices is (N*6, 2), base_normals is (6, 2)
        # We want max over all points and all normals of (point - C) . n_i
        # Then S = max_val / (sqrt(3)/2)
        
        # Shift vertices by outer_center
        shifted = vertices - np.array(outer_center)
        
        # Compute dot products: (N*6, 6)
        dots = shifted @ base_normals.T
        
        max_dot = np.max(dots)
        
        S = max_dot / (np.sqrt(3.0) / 2.0)
        return S
    
    best_S = float('inf')
    best_inner = None
    
    # Search for optimal rotation of the inner cluster
    # We can also search for the optimal rotation of the outer hexagon, but due to symmetry,
    # we can fix the outer hexagon rotation to 0 and rotate the inner cluster.
    
    # Let's try rotations from 0 to 60 degrees in steps of 1 degree
    for angle_deg in range(0, 60, 2):
        angle = np.radians(angle_deg)
        
        # Rotate the centers
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_centers = centers @ rotation_matrix.T
        
        # Generate all vertices for all hexagons
        base_verts = get_vertices(0.0) # 6x2
        all_verts = []
        for i in range(N):
            verts = base_verts + rotated_centers[i]
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Calculate the side length of the containing hexagon
        S = hexagon_side_length(all_verts)
        
        if S < best_S:
            best_S = S
            best_inner = [[float(x), float(y), angle_deg] for (x, y) in rotated_centers]
    
    # We can also try rotating the individual hexagons? No, the shape is determined by centers and the hexagon orientation.
    # Since the hexagons are regular, rotating them individually doesn't change the set of vertices if we also rotate the cluster.
    # But wait, the problem says "each inner hexagon may translate and rotate".
    # So we can rotate each hexagon independently.
    # However, for a cluster of touching hexagons, the relative orientations are often fixed to avoid overlap.
    # In our cluster, the hexagons are not touching, they are separated.
    # Actually, if we place them with centers at the axial coordinates, they are touching.
    # If they are touching, their orientations must be aligned to avoid overlap.
    # So rotating the entire cluster is the only degree of freedom.
    
    # If the hexagons are not touching, we could rotate them individually.
    # But our cluster has touching hexagons, so we must keep them aligned.
    
    return best_inner, (0.0, 0.0), float(best_S), 0.0
# EVOLVE-BLOCK-END
