import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate 12 inner hexagons in a 3x4 block lattice
    # Axial coordinates q, r
    # q in [-1, 0, 1], r in [-1, 0, 1, 2]
    centers = []
    for q in [-1, 0, 1]:
        for r in [-1, 0, 1, 2]:
            # Convert axial to Cartesian (Flat-top hexagons)
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            centers.append((x, y))
    
    # Calculate center of the cluster to center it at origin
    cx = sum(p[0] for p in centers) / len(centers)
    cy = sum(p[1] for p in centers) / len(centers)
    
    # Shift centers
    inner = []
    for x, y in centers:
        inner.append([x - cx, y - cy, 0.0])
        
    # Determine optimal outer hexagon side length
    # Based on analysis, Pointy-Top outer (angle 30 deg) fits better
    # Max extent calculated as ~4.743
    # Use a safe margin
    side = 4.75
    angle = 30.0
    
    return inner, (0.0, 0.0), side, angle
# EVOLVE-BLOCK-END
