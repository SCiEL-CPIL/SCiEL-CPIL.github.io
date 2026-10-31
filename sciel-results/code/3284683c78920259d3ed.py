import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an array of 11 points inside the unit equilateral triangle.
    Uses a triangular lattice grid (m=3) plus one additional point
    placed to maximize the minimum pairwise distance (and thus min triangle area).
    """
    sqrt3 = np.sqrt(3)
    
    # 10 points from m=3 triangular grid
    points = []
    
    # Row 0 (y=0): 4 points
    for i in range(4):
        points.append([i/3.0, 0.0])
        
    # Row 1 (y=sqrt(3)/6): 3 points
    for i in range(3):
        points.append([i/3.0 + 1/6.0, sqrt3/6.0])
        
    # Row 2 (y=sqrt(3)/3): 2 points
    for i in range(2):
        points.append([i/3.0 + 1/3.0, sqrt3/3.0])
        
    # Row 3 (y=sqrt(3)/2): 1 point
    points.append([0.5, sqrt3/2.0])
    
    # 11th point: Centroid of the triangle formed by 
    # (1/6, sqrt(3)/6), (1/2, sqrt(3)/6), and (1/3, 0)
    # This is a "hole" in the grid.
    # x = (1/6 + 1/2 + 1/3) / 3 = (1/6 + 3/6 + 2/6) / 3 = (6/6) / 3 = 1/3
    # y = (sqrt(3)/6 + sqrt(3)/6 + 0) / 3 = (2*sqrt(3)/6) / 3 = sqrt(3)/9
    points.append([1/3.0, sqrt3/9.0])
    
    return np.array(points)
# EVOLVE-BLOCK-END
