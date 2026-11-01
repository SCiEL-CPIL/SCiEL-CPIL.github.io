import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    S = 1.0 + 2.0 * sqrt3
    
    # Generate all lattice points in the hexagonal region with radius 2
    points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q) <= 2 and abs(r) <= 2 and abs(q + r) <= 2:
                points.append((q, r))
    
    # Sort by distance from center to pick the most central 11
    points.sort(key=lambda p: (max(abs(p[0]), abs(p[1]), abs(p[0]+p[1])), p[0], p[1]))
    
    selected = points[:N]
    
    inner = []
    for q, r in selected:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
