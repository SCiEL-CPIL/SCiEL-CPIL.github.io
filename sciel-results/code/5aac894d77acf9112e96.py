"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Hexagonal grid coordinates to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    centers = []
    
    # Center
    centers.append((0.0, 0.0))
    
    # 6 neighbors (K=2 cluster)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    neighbors = [
        (1, 0),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1)
    ]
    
    for q, r in neighbors:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    # 4 more from the next shell (K=3)
    # We need 4 points from the 6 available at distance 3.
    # The 6 points are at angles 30, 90, 150, 210, 270, 330.
    # We pick 4 to maintain symmetry or just pick any 4 that fit.
    # Let's pick (1,1), (-1,2), (-2,1), (-1,-1) ?
    # Let's just pick 4 distinct ones.
    extra_points = [
        (1, 1),   # Angle 30
        (-1, 2),  # Angle 90
        (-2, 1),  # Angle 150
        (1, -2)   # Angle 270 (Symmetric-ish)
    ]
    
    for q, r in extra_points:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), 4.5, 0.0
# EVOLVE-BLOCK-END
