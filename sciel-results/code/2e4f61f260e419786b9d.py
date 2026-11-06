import math

def design(problem):
    """
    Returns a 32x32 unit cell with concentric square rings.
    Alternating dielectric and air to create a periodic microstructure.
    """
    size = problem["pixels"]
    cell = [[0] * size for _ in range(size)]
    
    center = (size - 1) / 2.0
    
    for i in range(size):
        for j in range(size):
            dx = abs(i - center)
            dy = abs(j - center)
            # Chebyshev distance (square rings)
            dist = max(dx, dy)
            
            # Define rings. 
            # 0-4: Dielectric
            # 4-8: Air
            # 8-12: Dielectric
            # 12-16: Air
            # This creates a pattern with period 8 in the radial sense.
            
            # Map distance to ring index
            # We want alternating 1 and 0.
            # Let's use: if floor(dist / 4) is even -> 1, else 0.
            # dist 0-3.99 -> idx 0 (even) -> 1
            # dist 4-7.99 -> idx 1 (odd) -> 0
            # dist 8-11.99 -> idx 2 (even) -> 1
            # dist 12-15.99 -> idx 3 (odd) -> 0
            
            ring_idx = int(dist // 4)
            
            if ring_idx % 2 == 0:
                cell[i][j] = 1
            else:
                cell[i][j] = 0
                
    return cell
