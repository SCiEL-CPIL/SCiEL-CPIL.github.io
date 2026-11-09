import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tournament on n vertices where every set of 4 vertices
    has a common dominator.
    
    Strategy:
    The smallest such tournament has n=5.
    Construction:
    Vertex 0 dominates 1, 2, 3, 4.
    Vertices 1, 2, 3, 4 form a transitive tournament:
    1 -> 2, 3, 4
    2 -> 3, 4
    3 -> 4
    """
    n = 5
    # A[i][j] = 1 if i -> j
    A = [[0] * n for _ in range(n)]
    
    # 0 dominates all
    for j in range(1, n):
        A[0][j] = 1
        A[j][0] = 0
        
    # 1 dominates 2, 3, 4
    for j in range(2, n):
        A[1][j] = 1
        A[j][1] = 0
        
    # 2 dominates 3, 4
    for j in range(3, n):
        A[2][j] = 1
        A[j][2] = 0
        
    # 3 dominates 4
    A[3][4] = 1
    A[4][3] = 0
    
    # Generate orientation list
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] == 1:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
                
    return {'n': n, 'orientation': orientation}
# EVOLVE-BLOCK-END
