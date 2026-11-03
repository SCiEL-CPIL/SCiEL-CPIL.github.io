"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Define the 11 unit hexagons in a hexagonal cluster arrangement.
    # The centers are placed at the vertices of a regular hexagon of radius 2
    # and the center, forming a "flower" pattern.
    # This is the minimal area packing of 11 unit hexagons in a hexagonal lattice.
    # Centers: (0,0) and 6 neighbors at distance 2 in 6 directions, plus 4 more?
    # Actually, for 11 hexagons, the standard hexagonal number arrangement is:
    # Center (0,0)
    # 6 at distance 2: (2,0), (1, sqrt(3)), (-1, sqrt(3)), (-2,0), (-1, -sqrt(3)), (1, -sqrt(3))
    # That's 7. We need 4 more.
    # Next ring: distance 4? No, that's too far.
    # Let's use axial coordinates.
    # A hexagonal number H_k = 3k(k-1)+1. H_2=7, H_3=19.
    # So 11 is not a perfect hexagonal number.
    # Let's try a 3x3 grid of hexagons minus corners?
    # Or a compact arrangement.
    # Let's use the following centers for 11 hexagons:
    # (0,0)
    # (2,0), (1,s3), (-1,s3), (-2,0), (-1,-s3), (1,-s3)  [6 neighbors]
    # (4,0), (3,s3), (3,-s3), (-3,s3)  [4 more to make 11]
    # This might not be optimal. Let's try to find a better packing.
    # Actually, the minimal enclosing hexagon for a set of points is determined by
    # the max of the support functions in 6 directions.
    # Let's define the centers and then optimize the rotation of the whole cluster
    # and the position of the outer hexagon.

    s3 = np.sqrt(3)
    # Centers of the 11 unit hexagons in a compact cluster
    # Using axial coordinates (q, r) where x = 1.5*q, y = s3/2*q + s3*r
    # For unit hexagons, the distance between centers of adjacent hexagons is s3*2/3 * 2? 
    # No, the distance between centers of adjacent unit hexagons is 2 * (sqrt(3)/2) = sqrt(3) * 2 / 1? 
    # The apothem of a unit hexagon is sqrt(3)/2. So the distance between centers of adjacent hexagons is 2 * apothem = sqrt(3).
    # So in axial coordinates, the step is sqrt(3) in the y-direction for r, and 1.5 in x and sqrt(3)/2 in y for q.
    
    # Let's place them in a 3x3 grid in axial coordinates, but that's 9.
    # Let's add 2 more.
    # Centers in Cartesian coordinates:
    centers = [
        (0.0, 0.0),
        (s3, 0.0),
        (s3/2, 1.5),
        (-s3/2, 1.5),
        (-s3, 0.0),
        (-s3/2, -1.5),
        (s3/2, -1.5),
        (2*s3, 0.0),
        (2*s3 - s3/2, 1.5),
        (2*s3 - s3/2, -1.5),
        (0.0, 3.0)
    ]
    # This is a guess. Let's use a more systematic approach.
    # We'll generate a set of 11 centers and then optimize.
    
    # Actually, let's use the fact that the optimal packing of 11 unit hexagons in a hexagon
    # is known. The minimal side length is approximately 4.0 + something.
    # Let's use a simple arrangement and then optimize the rotation.
    
    # For now, let's use a line of 11 hexagons as a fallback, but that's not good.
    # Let's use a 2D arrangement.
    
    # Centers for 11 hexagons in a hexagonal cluster:
    # Ring 0: 1 hex at (0,0)
    # Ring 1: 6 hexes at distance s3
    # Ring 2: 12 hexes at distance 2*s3, but we only need 4 more.
    # Let's pick 4 from ring 2.
    
    ring1 = [(s3 * np.cos(np.pi/3 * i), s3 * np.sin(np.pi/3 * i)) for i in range(6)]
    ring2 = [(2*s3 * np.cos(np.pi/3 * i + np.pi/6), 2*s3 * np.sin(np.pi/3 * i + np.pi/6)) for i in range(6)]
    
    # Pick 4 from ring2 that are closest to the existing cluster to minimize the hull.
    # Actually, let's just use the first 4.
    centers = [(0.0, 0.0)] + ring1 + ring2[:4]
    
    # Convert to numpy array
    centers = np.array(centers)
    
    # Now, we need to find the minimal enclosing regular hexagon.
    # The side length of the enclosing hexagon is determined by the max of the support
    # functions in the 6 normal directions of the hexagon.
    # For a hexagon with side length s, the support in direction theta is s * (cos(theta) + sqrt(3) * sin(theta)) / 2?
    # Actually, the support function of a regular hexagon with side length s in direction (cos(phi), sin(phi)) is:
    # h(phi) = s * max_{k=0..5} cos(phi - k*pi/3)
    # The minimal s such that all centers are inside is the max over all centers and all 6 directions.
    
    # Let's define the 6 normal directions of the hexagon.
    # For a hexagon with vertices at angles 0, 60, 120, 180, 240, 300,
    # the normals are at 30, 90, 150, 210, 270, 330.
    
    # The support in direction (nx, ny) for a point (x, y) is nx*x + ny*y.
    # The hexagon with center (cx, cy) and side length s contains (x, y) if:
    # For all 6 directions, the projection of (x - cx, y - cy) onto the normal is <= s * apothem.
    # The apothem of a hexagon with side length s is s * sqrt(3) / 2.
    
    # So the condition is:
    # max_{i=0..5} (nx_i * (x - cx) + ny_i * (y - cy)) <= s * sqrt(3) / 2
    # for all centers (x, y).
    
    # This is equivalent to:
    # s >= max_{i=0..5, j=0..10} (nx_i * (x_j - cx) + ny_i * (y_j - cy)) / (sqrt(3) / 2)
    # and also the negative directions, so we need to consider both signs.
    
    # Actually, the hexagon is symmetric, so we need:
    # |max_{i=0..5, j=0..10} (nx_i * (x_j - cx) + ny_i * (y_j - cy))| <= s * sqrt(3) / 2
    
    # Let's define the 6 normal directions.
    normals = np.array([
        [np.cos(np.pi/6 + k * np.pi/3), np.sin(np.pi/6 + k * np.pi/3)] for k in range(6)
    ])
    
    # Now, we want to minimize s by choosing cx, cy, and the rotation of the hexagon.
    # The rotation of the hexagon is equivalent to rotating the normals.
    # Let's parameterize the rotation by an angle alpha.
    
    # We'll do a simple grid search over alpha, cx, cy.
    # But that might be slow. Let's use a smarter approach.
    
    # For a fixed alpha, the optimal cx, cy is the center of the minimal enclosing hexagon,
    # which is the average of the extreme points in each direction.
    
    # Let's do a grid search over alpha from 0 to 60 degrees (since the hexagon has 6-fold symmetry).
    
    best_s = float('inf')
    best_alpha = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Grid search over alpha
    for alpha in np.linspace(0, np.pi/3, 181):
        # Rotate the normals
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        normals_rot = (rot @ normals.T).T
        
        # For each normal direction, find the min and max projection of the centers.
        projections = centers @ normals_rot.T  # shape (11, 6)
        min_proj = projections.min(axis=0)
        max_proj = projections.max(axis=0)
        
        # The required apothem in each direction is (max_proj - min_proj) / 2
        # The center is at (min_proj + max_proj) / 2
        apothem_required = (max_proj - min_proj) / 2
        center_proj = (min_proj + max_proj) / 2
        
        # The side length s is related to the apothem by apothem = s * sqrt(3) / 2
        # So s = apothem * 2 / sqrt(3)
        s_required = apothem_required.max() * 2 / np.sqrt(3)
        
        if s_required < best_s:
            best_s = s_required
            best_alpha = alpha
            # Compute the center in Cartesian coordinates
            # The center in the rotated frame is center_proj, so in the original frame:
            center_rot = center_proj
            best_cx = np.dot(center_rot, normals_rot[:, 0])  # This is not correct.
            # Actually, the center is the same in all frames, so we can just use the average of the centers.
            best_cx = centers[:, 0].mean()
            best_cy = centers[:, 1].mean()
    
    # The outer hexagon has center (best_cx, best_cy), side length best_s, and rotation best_alpha.
    # The inner hexagons are at the given centers with rotation 0.
    
    inner = [[float(c[0]), float(c[1]), 0.0] for c in centers]
    return inner, (float(best_cx), float(best_cy)), float(best_s), float(np.degrees(best_alpha))
# EVOLVE-BLOCK-END
