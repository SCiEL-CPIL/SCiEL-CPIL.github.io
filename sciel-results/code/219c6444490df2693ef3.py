import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate hexagonal lattice points for a cluster of radius 2 (13 points)
    # v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 1.5)
    sqrt3 = np.sqrt(3)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3 / 2.0, 1.5])
    
    points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            # Hexagonal coordinate constraint: |i| <= 2, |j| <= 2, |i+j| <= 2
            if abs(i) <= 2 and abs(j) <= 2 and abs(i + j) <= 2:
                p = i * v1 + j * v2
                points.append(p)
                
    # We have 13 points. We need 11.
    # Sort by distance from center to remove the outermost ones?
    # Actually, all 6 corners are at the same distance.
    # Let's just take the first 11.
    # To be deterministic, sort by x then y.
    points.sort(key=lambda p: (p[0], p[1]))
    
    inner_points = points[:N]
    
    # Construct the inner list
    inner = []
    for p in inner_points:
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    # Calculate required outer side length
    # The centers must fit in a hexagon of side L = S - 1
    # The max projection of our points determines L
    # For the 13-point cluster, L = 2*sqrt(3)
    # Since we removed 2 points, L might be smaller or same.
    # Let's calculate the minimal L for the selected points to be safe.
    
    def min_hex_side(points_list):
        min_side = 0.0
        for p in points_list:
            x, y = p
            # Constraints for point (x,y) to be in hexagon of side L centered at 0, aligned
            # 1. |x| <= L
            # 2. |y| <= L * sqrt(3) / 2
            # 3. |x/2 + y*sqrt(3)/2| <= L
            # 4. |-x/2 + y*sqrt(3)/2| <= L
            
            c1 = abs(x)
            c2 = abs(y) * 2.0 / sqrt3
            c3 = abs(x/2.0 + y*sqrt3/2.0)
            c4 = abs(-x/2.0 + y*sqrt3/2.0)
            
            # L must be >= max(c1, c2, c3, c4)
            local_max = max(c1, c2, c3, c4)
            if local_max > min_side:
                min_side = local_max
        return min_side

    L = min_hex_side(inner_points)
    S = 1.0 + L
    
    return inner, (0.0, 0.0), float(S), 0.0
# EVOLVE-BLOCK-END
