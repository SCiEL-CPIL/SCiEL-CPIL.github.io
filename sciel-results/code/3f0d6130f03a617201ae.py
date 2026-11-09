import numpy as np

def squares_in_square10() -> tuple:
    n = 10
    
    # Constructive approach: 3x3 grid of large squares + 1 thin strip square
    # This topology allows sum of sides close to 3.0
    
    eps = 1e-6  # Small thickness for the 10th square
    w = (1.0 - eps) / 3.0  # Width of the 3 main columns
    h = 1.0 / 3.0          # Height of the 3 rows
    
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 9 squares in 3x3 grid
    idx = 0
    for i in range(3):
        for j in range(3):
            cx = (j + 0.5) * w
            cy = (i + 0.5) * h
            centers[idx, 0] = cx
            centers[idx, 1] = cy
            sides[idx] = w
            idx += 1
            
    # 1 square in the remaining strip
    # The strip is at x in [3w, 1], which is [1-eps, 1]
    # Width is eps. Height is 1.
    # Square side is eps.
    cx = 3*w + eps/2  # Which is 1 - eps/2
    cy = 0.5
    centers[idx, 0] = cx
    centers[idx, 1] = cy
    sides[idx] = eps
    
    return centers, sides
