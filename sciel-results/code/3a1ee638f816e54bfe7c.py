import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate hexagonal lattice points
    # Coordinates: x = 1.5 * i, y = sqrt(3) * (j + i/2)
    # Distance squared from origin: 3 * (i^2 + j^2 + i*j)
    
    points = []
    # We need 11 points.
    # k=0: 1 point
    # k=1: 6 points (total 7)
    # k=3: 12 points (total 19)
    # We take all from k=0, k=1, and 4 from k=3.
    
    candidates = []
    # Search range for i, j
    # Max i, j for k=3 is 2 or 3.
    for i in range(-3, 4):
        for j in range(-3, 4):
            k = i*i + j*j + i*j
            if k == 0 or k == 1 or k == 3:
                x = 1.5 * i
                y = np.sqrt(3) * (j + 0.5 * i)
                # Store (k, x, y) to sort by distance
                candidates.append((k, x, y))
                
    # Sort by k (distance squared)
    candidates.sort(key=lambda p: p[0])
    
    # Take first N
    selected = candidates[:N]
    
    inner = []
    max_dist = 0.0
    for _, x, y in selected:
        inner.append([x, y, 0.0])
        # Calculate max distance from origin to any vertex of this hexagon
        # The hexagon has circumradius 1.
        # The max distance from origin to any point in the hexagon is dist(center) + 1
        # This is an upper bound.
        d = np.sqrt(x*x + y*y) + 1.0
        if d > max_dist:
            max_dist = d
            
    # The outer hexagon must have inradius >= max_dist
    # Inradius r = L * sqrt(3) / 2
    # L = 2 * r / sqrt(3)
    side = 2.0 * max_dist / np.sqrt(3)
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
