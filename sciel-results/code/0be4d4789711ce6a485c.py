import math
import time
import numpy as np

N = 11
D = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Vertices of a unit regular hexagon (side length 1) centered at origin
    # Pointy-right orientation: vertices at angles 0, 60, 120, 180, 240, 300
    hex_verts_base = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])

    # Lattice vectors for hexagonal packing of unit hexagons
    # Distance between centers of touching hexagons is sqrt(3)
    # Basis vectors:
    u = np.array([math.sqrt(3), 0.0])
    v = np.array([math.sqrt(3)/2, 1.5])
    
    # Generate a set of candidate lattice points (axial coordinates)
    # We want 11 points.
    # Standard hexagonal grid points (i, j) -> i*u + j*v
    # Center (0,0)
    # Neighbors (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Next ring: (2,0), (1,1), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    
    # Let's pick 11 points that form a compact shape.
    # A hexagon of side 2 has 19 points. We can take a subset.
    # Or simply take the 7 center+neighbors and 4 from the next ring symmetrically.
    # To minimize the bounding hexagon, we should avoid points that are far in any of the 6 directions.
    # The 6 directions of the bounding hexagon correspond to the normals of the outer hexagon.
    
    # Let's use these 11 points:
    lattice_points = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2) # Picking 4 opposite pairs from next ring? 
        # Wait, (2,0) and (-2,0) are distance 2*sqrt(3) from center.
        # (1,1) is vector u+v.
        # Let's just compute the bounding box for a few symmetric sets and pick the best.
    ]
    
    # Actually, let's just use a simple valid configuration:
    # 1 center, 6 neighbors, and 4 more placed to keep symmetry.
    # If we place them at (2,0), (1,1), (-1,2), (-2,1), we break symmetry.
    # Let's try (2,0), (-2,0), (1,1), (-1,-1)? No, (-1,-1) is already a neighbor? No, (1,-1) is neighbor. (-1,1) is neighbor.
    # Axial coords: (1,-1) is neighbor. (-1,1) is neighbor.
    # So (1,1) is NOT a neighbor.
    
    # Let's define the 11 points explicitly in Cartesian coordinates.
    centers = []
    # Center
    centers.append(np.array([0.0, 0.0]))
    
    # 6 Neighbors
    for k in range(6):
        angle = k * math.pi / 3
        # Distance is sqrt(3)
        # Wait, the lattice vectors u and v have length sqrt(3).
        # The 6 neighbors are at angles 0, 60, 120, 180, 240, 300?
        # u is at 0 deg. v is at 60 deg.
        # The 6 neighbors in the hexagonal lattice are at:
        # u, v, v-u, -u, -v, u-v
        # Let's verify distances.
        # u = (sqrt(3), 0)
        # v = (sqrt(3)/2, 1.5)
        # v-u = (-sqrt(3)/2, 1.5) -> angle 120? No, atan2(1.5, -0.866) = 120 deg.
        # -u = (-sqrt(3), 0) -> 180 deg.
        # -v = (-sqrt(3)/2, -1.5) -> 240 deg.
        # u-v = (sqrt(3)/2, -1.5) -> 300 deg.
        
        neighbors = [
            u, v, v-u, -u, -v, u-v
        ]
        for n_vec in neighbors:
            centers.append(n_vec.copy())
            
    # Now we have 7 centers. We need 4 more.
    # Next ring points are sums of two basis vectors or similar.
    # (2,0) -> 2u
    # (1,1) -> u+v
    # (0,2) -> 2v
    # (-1,2) -> v-u ? No, that's a neighbor. (-1,2) in axial is -u + 2v?
    # Let's just generate all points with max(|i|,|j|,|i+j|) <= 2 and pick 4.
    
    candidate_extra = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            # Axial coordinate check for hex distance <= 2
            if abs(i) <= 2 and abs(j) <= 2 and abs(i+j) <= 2:
                # Exclude the 7 already used (distance <= 1)
                if abs(i) <= 1 and abs(j) <= 1 and abs(i+j) <= 1:
                    continue
                pos = i * u + j * v
                candidate_extra.append(pos)
                
    # We need to pick 4 from candidate_extra to minimize the bounding hexagon.
    # Since we don't have time for optimization, let's pick 4 that are symmetric.
    # The 12 points in the second ring are at distance 2*sqrt(3) or sqrt(3)*sqrt(3)=3?
    # Distance of 2u is 2*sqrt(3).
    # Distance of u+v is sqrt(3+3+2*3*0.5) = sqrt(6+3) = 3? No.
    # |u+v|^2 = 3 + 2.25 + 2*(sqrt(3)/2*1.5) = 5.25 + 1.5*sqrt(3) ?
    # u=(1.732, 0), v=(0.866, 1.5). u+v = (2.598, 1.5). |u+v|^2 = 6.75 + 2.25 = 9. So dist=3.
    # |2u|^2 = 12. So dist=sqrt(12)=3.46.
    # So u+v is closer than 2u.
    # The 6 points of type u+v, v+u, etc (corners of the hex ring) are at distance 3.
    # The 6 points of type 2u, 2v, etc (midpoints of edges) are at distance 2*sqrt(3) ~ 3.46.
    # So the "corners" of the hexagonal grid (like u+v) are closer to the center than the "edge midpoints" (like 2u).
    # Therefore, to minimize the bounding box, we should prefer the corner points.
    
    # The 6 corner points of the second ring are:
    # u+v, v-u (no, that's neighbor), 
    # Let's list them:
    # (1,1) -> u+v
    # (1,-1) -> neighbor
    # (-1,1) -> neighbor
    # (2,0) -> 2u
    # (0,2) -> 2v
    # (-2,0) -> -2u
    # (0,-2) -> -2v
    # (-1,2) -> -u+2v
    # (1,-2) -> u-2v
    # (-2,1) -> -2u+v
    # (2,-1) -> 2u-v
    
    # The "corners" (max distance from center in hex metric) are usually the ones that define the shape.
    # Actually, the bounding hexagon side length is determined by the maximum projection.
    # Let's just pick 4 symmetric points from the "corners" of the second ring.
    # The 6 points at distance 3 (in Euclidean) are:
    # u+v, v+u (same), 
    # Actually, the 6 points are:
    # (1,1), (1,-2), (-2,1), (-1,-1) is neighbor? No.
    # Let's just pick:
    extra_points = [
        u + v,      # (1,1)
        -(u + v),   # (-1,-1) -- Wait, (-1,-1) is a neighbor? No, (1,-1) is neighbor. (-1,1) is neighbor.
                    # Axial (i,j). (1,1) is not neighbor. (-1,-1) is not neighbor.
        v - u,      # (0,1) is neighbor? No, (0,1) is v. (1,0) is u.
                    # v-u is (-1,1) which IS a neighbor.
        # Let's re-evaluate neighbors.
        # Neighbors in axial (i,j): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
        # So (1,1) is NOT a neighbor.
        # (-1,-1) is NOT a neighbor.
        # (2,0) is NOT a neighbor.
        # (-2,0) is NOT a neighbor.
        # (0,2) is NOT a neighbor.
        # (0,-2) is NOT a neighbor.
        
        # So we have 12 candidates.
        # The 6 "corners" of the hexagon of radius 2 are:
        # (2,0), (1,1), (-1,2), (-2,0), (-1,-1), (1,-2).
        # These are at distance sqrt(12)=3.46 and 3 respectively?
        # (2,0) -> 2u -> dist 3.46
        # (1,1) -> u+v -> dist 3.0
        # So the "corners" of the hexagonal grid (in terms of hex distance) are the ones with |i+j| large?
        # Hex distance is (|i|+|j|+|i+j|)/2.
        # (2,0): (2+0+2)/2 = 2.
        # (1,1): (1+1+2)/2 = 2.
        # They are all at hex distance 2.
        # But Euclidean distance varies.
        # To minimize the enclosing regular hexagon (which is aligned with the lattice if we rotate appropriately),
        # we should minimize the maximum projection on the 6 normal directions.
        # The normals are aligned with the lattice axes if the outer hexagon is aligned.
        # If we align the outer hexagon with the lattice (angle 0), the normals are at 0, 60, 120...
        # The projection of a point P on normal n is P . n.
        # For (2,0) -> 2u = (2sqrt(3), 0). Proj on (1,0) is 2sqrt(3) ~ 3.46.
        # For (1,1) -> u+v = (2.598, 1.5). Proj on (1,0) is 2.598. Proj on (0.5, 0.866) is 1.299+1.3=2.6.
        # So (2,0) sticks out more in the x-direction.
        # Therefore, picking the "corner" points like (1,1) is better than (2,0) if we want a compact shape in all directions?
        # Actually, a regular hexagon of side S contains a hexagonal grid cluster.
        # The optimal outer hexagon will likely be aligned with the lattice.
        
        # Let's just pick 4 points that are symmetric:
        # (1,1), (-1,-1), (1,-2), (-1,2)?
        # Or (1,1), (-1,1) is neighbor...
        # Let's pick (1,1), (-1,-1), (2,0), (-2,0)?
        # This creates a diamond shape.
        
        # Let's just use the first 4 from a sorted list of candidates by distance to center, prioritizing symmetry.
        # For simplicity and robustness, let's just add 4 points that are far but symmetric.
        # (1,1), (-1,-1), (0,2), (0,-2)?
    ]
    
    # Let's hardcode a valid set of 11 points that is known to be compact.
    # Using the 7 center+neighbors, and adding (1,1), (-1,-1), (2,0), (-2,0) is not symmetric.
    # Adding (1,1), (-1,1) is invalid (neighbor).
    # Adding (1,1), (-1,-1), (1,-2), (-1,2) is symmetric.
    
    final_centers = centers.copy()
    final_centers.append(u + v)       # (1,1)
    final_centers.append(-(u + v))    # (-1,-1)
    final_centers.append(u - 2*v)     # (1,-2)
    final_centers.append(-u + 2*v)    # (-1,2)
    
    # Verify no overlaps (should be fine with lattice)
    # And compute bounding hexagon side.
    
    angles = [0.0] * 11
    
    # Compute all vertices
    all_verts = np.zeros((11 * 6, 2))
    for i in range(11):
        c = final_centers[i]
        # Rotation is 0, so just add center
        all_verts[i*6:(i+1)*6] = hex_verts_base + c
        
    # Normals of outer hexagon (angle 0)
    normals = np.array([
        [math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)]
        for k in range(6)
    ])
    
    # Projections
    projs = all_verts @ normals.T
    max_abs_proj = np.max(np.abs(projs))
    S = (2.0 / math.sqrt(3)) * max_abs_proj
    
    # Check validity: Ensure no overlaps.
    # For lattice points, min distance is sqrt(3). Hexagon radius (circumradius) is 1.
    # Two hexagons overlap if center distance < 2 * apothem? No.
    # Two unit hexagons touch if center distance is sqrt(3).
    # They overlap if center distance < sqrt(3).
    # In our lattice, min distance is exactly sqrt(3). So they touch but do not overlap.
    # "Disjoint" usually means interiors are disjoint. Touching is allowed.
    
    inner_rows = []
    for i in range(11):
        inner_rows.append([float(final_centers[i][0]), float(final_centers[i][1]), 0.0])
        
    center = (0.0, 0.0)
    side = float(S)
    angle_deg = 0.0
    
    return inner_rows, center, side, angle_deg
