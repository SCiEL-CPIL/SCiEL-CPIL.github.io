# EVOLVE-BLOCK-START
import numpy as np

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, s).
    Uses a hexagonal lattice subset for optimal packing.
    """
    # Lattice vectors for unit hexagons
    # Distance between centers is sqrt(3)
    # a1 = (sqrt(3), 0)
    # a2 = (sqrt(3)/2, 3/2)
    
    # We select 12 points from the hexagonal lattice.
    # The optimal packing for N=12 is a subset of the lattice cluster of radius 2 (19 points).
    # We pick 12 points such that the maximum hexagonal distance is 2.
    
    # Generate points with hexagonal distance <= 2
    # d(i,j) = max(|i|, |j|, |i+j|)
    points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            d = max(abs(i), abs(j), abs(i+j))
            if d <= 2:
                points.append((i, j, d))
                
    # Sort by distance, then by angle to pick a symmetric set
    # We need 12 points.
    # There are 7 points with d=0 or 1.
    # There are 12 points with d=2.
    # We take all 7 and 5 from d=2.
    
    points_d01 = [p for p in points if p[2] <= 1]
    points_d2 = [p for p in points if p[2] == 2]
    
    # Sort d2 points by angle to pick a contiguous arc or symmetric set
    # Angle of (i,j) in lattice coords
    def lattice_angle(p):
        i, j, _ = p
        x = np.sqrt(3) * (i + j/2)
        y = 1.5 * j
        return np.arctan2(y, x)
        
    points_d2.sort(key=lattice_angle)
    
    # Pick 5 points from d2. To be symmetric, we might pick them evenly.
    # Or just take the first 5.
    # Actually, any 5 will work for the size s.
    # For stability, let's pick 5 that are spread out.
    # Indices 0, 2, 4, 6, 8?
    # There are 12 points.
    # Let's just take the first 5 from the sorted list.
    selected_d2 = points_d2[:5]
    
    selected_points = points_d01 + selected_d2
    
    # Convert to Cartesian coordinates
    inner_centers = []
    for i, j, _ in selected_points:
        x = np.sqrt(3) * (i + j/2)
        y = 1.5 * j
        inner_centers.append([x, y])
        
    inner_centers = np.array(inner_centers)
    
    # Inner hexagons have angle 0
    inner = np.zeros((12, 3))
    inner[:, 0] = inner_centers[:, 0]
    inner[:, 1] = inner_centers[:, 1]
    inner[:, 2] = 0.0
    
    # Outer hexagon
    # Center at (0,0), angle 0
    outer = np.array([0.0, 0.0, 0.0])
    
    # Calculate s
    # s = 2/sqrt(3) * max_proj
    # Normals for angle 0:
    # n0: 30 deg, n1: 90 deg, n2: 150 deg
    n0 = np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))])
    n1 = np.array([np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))])
    n2 = np.array([np.cos(np.deg2rad(150)), np.sin(np.deg2rad(150))])
    
    max_proj = 0
    for c in inner_centers:
        p0 = np.dot(c, n0)
        p1 = np.dot(c, n1)
        p2 = np.dot(c, n2)
        max_proj = max(max_proj, abs(p0), abs(p1), abs(p2))
        
    s = 2 * max_proj / np.sqrt(3)
    
    return inner, outer, s

# EVOLVE-BLOCK-END
