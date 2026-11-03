import numpy as np
import itertools

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define lattice vectors for hexagonal packing
    # Unit hexagon side length 1.
    # Distance between centers in honeycomb is sqrt(3).
    # Orientation: Flat top hexagons.
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 1.5)
    
    sqrt3 = np.sqrt(3)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3 / 2.0, 1.5])
    
    # Generate candidate centers
    # Center
    centers = [np.array([0.0, 0.0])]
    
    # Ring 1 (6 points)
    ring1_indices = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for i, j in ring1_indices:
        centers.append(i * v1 + j * v2)
        
    # Ring 2 candidates (6 points at distance 3)
    # These are (1,1), (-1,2), (-2,1), (-1,-1), (1,-2), (2,-1) in (i,j) coords?
    # Let's verify distances.
    # (1,1) -> v1+v2 = (1.5*sqrt3, 1.5). Dist 3.
    # (-1,2) -> -v1+2v2 = (0, 3). Dist 3.
    # (-2,1) -> -2v1+v2 = (-1.5*sqrt3, 1.5). Dist 3.
    # (-1,-1) -> -v1-v2 = (-1.5*sqrt3, -1.5). Dist 3.
    # (1,-2) -> v1-2v2 = (0, -3). Dist 3.
    # (2,-1) -> 2v1-v2 = (1.5*sqrt3, -1.5). Dist 3.
    
    ring2_candidates = []
    for i, j in [(1,1), (-1,2), (-2,1), (-1,-1), (1,-2), (2,-1)]:
        ring2_candidates.append(i * v1 + j * v2)
        
    # We need 4 more centers to reach N=11.
    # We try all combinations of 4 from the 6 candidates.
    
    best_S = float('inf')
    best_centers = None
    
    # Function to get vertices of a unit hexagon at center c, angle 0
    def get_hex_vertices(c):
        # Angle 0: vertex at (1, 0)
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        vertices = []
        for a in angles:
            v = np.array([np.cos(a), np.sin(a)])
            vertices.append(c + v)
        return vertices

    # Normals for the outer hexagon (aligned with inner lattice symmetry)
    # Outer hexagon flat-top.
    # Normals point to the midpoints of the sides.
    # n1: (0, 1)
    # n2: (sqrt(3)/2, 1/2)
    # n3: (-sqrt(3)/2, 1/2)
    normals = [
        np.array([0.0, 1.0]),
        np.array([sqrt3 / 2.0, 0.5]),
        np.array([-sqrt3 / 2.0, 0.5])
    ]
    
    def calc_min_side(centers_list):
        # Get all vertices
        all_vertices = []
        for c in centers_list:
            all_vertices.extend(get_hex_vertices(c))
        
        max_proj = 0.0
        for n in normals:
            for v in all_vertices:
                proj = np.dot(v, n)
                if abs(proj) > max_proj:
                    max_proj = abs(proj)
        
        # S * sqrt(3) / 2 = max_proj
        S = 2.0 * max_proj / sqrt3
        return S

    # Iterate over combinations of 4 points from ring2_candidates
    for combo in itertools.combinations(range(6), 4):
        selected_ring2 = [ring2_candidates[i] for i in combo]
        current_centers = centers + selected_ring2
        
        S = calc_min_side(current_centers)
        
        if S < best_S:
            best_S = S
            best_centers = current_centers
            
    # Prepare output
    # inner: list of [x, y, angle]
    inner = []
    for c in best_centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
        
    # Outer: center (0,0), side best_S, angle 0
    return inner, (0.0, 0.0), float(best_S), 0.0
# EVOLVE-BLOCK-END
