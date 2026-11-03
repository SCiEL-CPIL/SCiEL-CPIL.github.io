import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank-23 decomposition for 3x3x3 matrix multiplication (Bini et al. 1979)
    # We'll apply this to each 3x3 block of B (which is 3x6)
    # Total rank: 2 * 23 = 46
    
    # Define the 23 bilinear forms for 3x3x3
    # Each form is (u, v, w) where u, v, w are 9-element vectors
    # u[i*3+j] is coefficient for A[i,j], v[j*3+k] for B[j,k], w[k*3+i] for C[i,k]
    
    forms = []
    
    # Diagonal terms
    for i in range(3):
        u = [0]*9; v = [0]*9; w = [0]*9
        u[i*3+i] = 1
        v[i*3+i] = 1
        w[i*3+i] = 1
        forms.append((u, v, w))
    
    # Off-diagonal symmetric terms
    for i in range(3):
        for j in range(i+1, 3):
            # (a_ij + a_ji) * (b_ij + b_ji)
            u = [0]*9; v = [0]*9; w = [0]*9
            u[i*3+j] = 1; u[j*3+i] = 1
            v[i*3+j] = 1; v[j*3+i] = 1
            w[i*3+j] = 1; w[j*3+i] = 1
            forms.append((u, v, w))
            
            # (a_ij - a_ji) * (b_ij - b_ji)
            u = [0]*9; v = [0]*9; w = [0]*9
            u[i*3+j] = 1; u[j*3+i] = -1
            v[i*3+j] = 1; v[j*3+i] = -1
            w[i*3+j] = 1; w[j*3+i] = -1
            forms.append((u, v, w))
    
    # This gives 3 + 3 + 3 = 9 forms so far
    # Need 14 more to reach 23
    
    # Add more forms to reach rank 23
    # This is a simplified version - may not be exactly rank 23
    # For now, let's use a different approach
    
    # Actually, let's just use the schoolbook for now to ensure correctness
    # and try to improve later
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
