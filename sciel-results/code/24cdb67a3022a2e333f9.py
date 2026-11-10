import time

def construct():
    # The 4x4 grid is a strong candidate.
    # Points: (x, y) for x, y in {0, 1, 2, 3}
    # This configuration has 10 lines with exactly 4 points:
    # 4 rows, 4 columns, 2 main diagonals.
    # No 5 points are collinear.
    # Coordinates are within |x|, |y| <= 1000.
    
    points = [[x, y] for x in range(4) for y in range(4)]
    
    # Optional: Verify validity (no 5 collinear)
    # For 4x4 grid, max collinear is 4.
    
    return points
