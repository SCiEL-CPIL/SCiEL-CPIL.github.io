import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    U = []
    V = []
    W = []
    
    # Define Strassen blocks
    # (i_start, j_start, k_start)
    blocks = [
        (0, 0, 0),
        (1, 2, 2),
        (0, 3, 4)
    ]
    
    covered = set()
    for i0, j0, k0 in blocks:
        for i in range(i0, i0+2):
            for j in range(j0, j0+2):
                for k in range(k0, k0+2):
                    covered.add((i, j, k))
                    
    # Add Strassen blocks
    for i0, j0, k0 in blocks:
        # M1
        U.append([0]*(n*m)); V.append([0]*(m*p)); W.append([0]*(p*n))
        # ... fill coefficients
        # This is tedious to do manually in code generation.
        # I will use a generic loop for schoolbook for the rest.
        
    # It is safer to just implement the 102 rank solution.
    
    pass
