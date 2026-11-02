"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition vectors
    # A indices: 00,01,10,11 -> 0,1,2,3
    # B indices: 00,01,10,11 -> 0,1,2,3
    # C indices: 00,01,10,11 -> 0,1,2,3
    strassen_2x2x2 = [
        ([1,0,0,0], [1,0,0,0], [1,0,0,0]),
        ([0,1,0,0], [0,0,1,0], [0,1,0,0]),
        ([0,0,1,0], [0,1,0,0], [0,0,1,0]),
        ([0,0,0,1], [0,0,0,1], [0,0,0,1]),
        ([1,0,0,0], [0,0,1,0], [0,1,0,0]),
        ([0,1,0,0], [0,0,1,0], [1,0,0,0]),
        ([0,0,0,1], [0,1,0,0], [0,1,0,0]),
    ]
    
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        for u, v, w in strassen_2x2x2:
            # Map local indices to global indices
            # A: (i0,j0)->0, (i0,j1)->1, (i1,j0)->2, (i1,j1)->3
            # B: (j0,k0)->0, (j0,k1)->1, (j1,k0)->2, (j1,k1)->3
            # C: (i0,k0)->0, (i0,k1)->1, (i1,k0)->2, (i1,k1)->3
            
            for local_idx, val in enumerate(u):
                if val != 0:
                    li, lj = divmod(local_idx, 2)
                    gi = i0 if li == 0 else i1
                    gj = j0 if lj == 0 else j1
                    U[gi * m + gj][col] += val
                    
            for local_idx, val in enumerate(v):
                if val != 0:
                    lj, lk = divmod(local_idx, 2)
                    gj = j0 if lj == 0 else j1
                    gk = k0 if lk == 0 else k1
                    V[gj * p + gk][col] += val
                    
            for local_idx, val in enumerate(w):
                if val != 0:
                    li, lk = divmod(local_idx, 2)
                    gi = i0 if li == 0 else i1
                    gk = k0 if lk == 0 else k1
                    W[gk * n + gi][col] += val
            col += 1
            
    # Block 1: A[0:2, 0:2] B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1)
    # Block 2: A[0:2, 0:2] B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 1, 0, 1, 2, 3)
    # Block 3: A[0:2, 2:4] B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 1, 2, 3, 0, 1)
    # Block 4: A[0:2, 2:4] B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 1, 2, 3, 2, 3)
    
    # Part 2: A[0:2, 0:4] B[0:4, 4:5] -> C[0:2, 4:5]
    # Rank 8
    for i in range(2):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    # Part 3: A[0:2, 4:5] B[4:5, 0:5] -> C[0:2, 0:5]
    # Rank 10
    for i in range(2):
        for k in range(5):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
