import numpy as np

def construct():
    N = 21
    
    # Calculate optimal diameter for 3x7 hexagonal strip
    # Width W = 3.5 * d
    # Height H = (3 * sqrt(3) + 1) * d
    # Perimeter constraint: 2 * (W + H) <= 4  => W + H <= 2
    # d * (3.5 + 3*sqrt(3) + 1) = 2
    # d * (4.5 + 3*sqrt(3)) = 2
    
    sqrt3 = np.sqrt(3)
    coeff = 4.5 + 3 * sqrt3
    d = 2.0 / coeff
    r = d / 2.0
    
    circles = []
    
    # 7 rows, 3 columns
    # Row i (0 to 6)
    for i in range(7):
        y = i * (sqrt3 / 2.0) * d
        if i % 2 == 0:
            # Even row: x = 0, d, 2d
            for j in range(3):
                x = j * d
                circles.append((x, y, r))
        else:
            # Odd row: x = 0.5d, 1.5d, 2.5d
            for j in range(3):
                x = (j + 0.5) * d
                circles.append((x, y, r))
                
    return np.array(circles)
