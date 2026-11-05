import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: Strassen for 4x4x4 (columns 0-3 of B)
    # Rank 49
    # Part 2: Schoolbook for 4x4x3 (columns 4-6 of B)
    # Rank 48
    
    rank1 = 49
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    # U2, V2, W2 are 4x7
    U2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, -1],
        [0, 0, 1, 0, 0, -1, 1],
        [0, 0, 1, 1, -1, 0, 0]
    ])
    V2 = np.array([
        [1, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, -1]
    ])
    W2 = np.array([
        [1, -1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, 0, -1, 0, 1, 0, -1]
    ])
    
    # Build 4x4x4 factors (49 terms)
    col_offset = 0
    for c1 in range(7):
        for c2 in range(7):
            c = col_offset + c1 * 7 + c2
            # U: (i1, j1) and (i2, j2) -> index (i1*2+i2)*4 + (j1*2+j2)?
            # No, U is indexed by i*m+j. m=4.
            # i = i1*2 + i2, j = j1*2 + j2.
            # U2 is indexed by i2*2 + j2.
            # So U[i, c] = U2[i2*2+j2, c1] * U2[i1*2+j1, c2]?
            # Actually, the tensor product of the decompositions.
            # T = sum_{c1,c2} (u_{c1} \otimes u_{c2}) \otimes (v_{c1} \otimes v_{c2}) \otimes (w_{c1} \otimes w_{c2})
            # u_{c1} \otimes u_{c2} is a vector of size 16.
            # Index in U is i*4 + j.
            # i = i1*2 + i2, j = j1*2 + j2.
            # u_{c1}[i1*2+j1] * u_{c2}[i2*2+j2]
            
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            idx_u = i * 4 + j
                            val = U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]
                            U[idx_u][c] = val
                            
            # V: indexed by j*p+k. p=4 for this part.
            # j = j1*2+j2, k = k1*2+k2.
            # V2 is indexed by j2*2+k2.
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            idx_v = j * 4 + k
                            val = V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]
                            V[idx_v][c] = val
                            
            # W: indexed by k*n+i. n=4.
            # k = k1*2+k2, i = i1*2+i2.
            # W2 is indexed by k2*2+i2.
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            idx_w = k * 4 + i
                            val = W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]
                            W[idx_w][c] = val
                            
            col_offset += 1
            
    # Schoolbook for 4x4x3 (columns 4-6 of B)
    # Rank 48
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                k_global = k + 4 # columns 4,5,6
                idx_u = i * 4 + j
                idx_v = j * 7 + k_global
                idx_w = k_global * 4 + i
                U[idx_u][col_offset] = 1
                V[idx_v][col_offset] = 1
                W[idx_w][col_offset] = 1
                col_offset += 1
                
    return {"U": U, "V": V, "W": W}
