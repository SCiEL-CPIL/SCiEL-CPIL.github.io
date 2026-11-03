import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate centers for N unit hexagons in a hexagonal lattice
    # Lattice vectors: v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 3/2)
    # Using axial coordinates (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    centers = []
    # We need to generate enough points
    # Number of points in hexagon of radius k is 3k(k+1) + 1
    # Find k such that 3k(k+1) + 1 >= N
    k = 0
    while 3 * k * (k + 1) + 1 < N:
        k += 1
        
    # Generate all (q, r) with hex distance <= k
    candidates = []
    for q in range(-k, k + 1):
        for r in range(-k, k + 1):
            # Hex distance formula: (|q| + |r| + |q+r|) / 2
            dist_hex = (abs(q) + abs(r) + abs(q + r)) / 2
            if dist_hex <= k:
                x = np.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                # Euclidean distance to origin for sorting
                dist_eucl = x * x + y * y
                candidates.append((dist_eucl, x, y))
                
    # Sort by Euclidean distance to pack tightly around center
    candidates.sort(key=lambda t: t[0])
    
    # Take first N
    selected = candidates[:N]
    inner_centers = [(x, y) for _, x, y in selected]
    
    # Calculate minimum outer side length S
    # Inner hexagons are aligned (angle 0).
    # Condition: Centers must be inside a hexagon of side S-1.
    # Hexagon of side S' centered at origin, aligned with x-axis:
    # |x| <= S'
    # |sqrt(3)*x + y| <= sqrt(3)*S'
    # |-sqrt(3)*x + y| <= sqrt(3)*S'
    
    max_x = 0.0
    max_t1 = 0.0
    max_t2 = 0.0
    
    for x, y in inner_centers:
        max_x = max(max_x, abs(x))
        max_t1 = max(max_t1, abs(np.sqrt(3) * x + y))
        max_t2 = max(max_t2, abs(-np.sqrt(3) * x + y))
        
    # S' is the side length of the inner bounding hexagon
    # S' >= max_x
    # S' >= max_t1 / sqrt(3)
    # S' >= max_t2 / sqrt(3)
    
    S_prime = max(max_x, max_t1 / np.sqrt(3), max_t2 / np.sqrt(3))
    
    # Outer side length S = S_prime + 1 (since inner side is 1)
    S = S_prime + 1.0
    
    # Construct return values
    # inner: list of [x, y, angle_degrees]
    inner_list = [[x, y, 0.0] for x, y in inner_centers]
    
    # outer center
    center = (0.0, 0.0)
    
    # outer side length
    side = S
    
    # outer angle
    angle = 0.0
    
    return inner_list, center, side, angle
# EVOLVE-BLOCK-END
