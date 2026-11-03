import math

N = 12

def construct():
    # 6x2 grid of unit hexagons
    # Centers:
    # Row i (0 to 5), Col j (0 to 1)
    # x = j * sqrt(3) + (i % 2) * sqrt(3)/2
    # y = i * 1.5
    
    sqrt3 = math.sqrt(3)
    inner = []
    for i in range(6):
        for j in range(2):
            x = j * sqrt3 + (i % 2) * sqrt3 / 2.0
            y = i * 1.5
            # Center the grid
            # Shift x by -sqrt3/2, y by -3.75 (center of 0 to 7.5)
            x -= sqrt3 / 2.0
            y -= 3.75
            inner.append([x, y, 0.0])
            
    # Calculate optimal side length
    # S = 4 * sqrt(3) + 1.5
    side = 4 * sqrt3 + 1.5
    
    return inner, (0.0, 0.0), side, 0.0
