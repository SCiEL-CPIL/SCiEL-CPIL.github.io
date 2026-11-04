import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]  # 15 x 66
    V = [[0] * rank for _ in range(m * p)]  # 25 x 66
    W = [[0] * rank for _ in range(p * n)]  # 15 x 66
    
    col = 0
    
    def schoolbook(i0, j0, k0, n_s, m_s, p_s, U, V, W, col):
        for i_s in range(n_s):
            for j_s in range(m_s):
                for k_s in range(p_s):
                    i = i0 + i_s
                    j = j0 + j_s
                    k = k0 + k_s
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col
    
    def strassen_2x2x2(i0, j0, k0, U, V, W, col):
        # Strassen for 2x2x2 block at offset (i0, j0, k0)
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U[(i0+0)*m+(j0+0)][col] = 1; U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+0)*p+(k0+0)][col] = 1; V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = 1; W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        U[(i0+1)*m+(j0+0)][col] = 1; U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+0)*p+(k0+0)][col] = 1
        W[(k0+0)*n+(i0+1)][col] = 1; W[(k0+1)*n+(i0+1)][col] = -1
        col += 1
        
        # M3 = A[0,0] * (B[0,1]-B[1,1])
        U[(i0+0)*m+(j0+0)][col] = 1
        V[(j0+0)*p+(k0+1)][col] = 1; V[(j0+1)*p+(k0+1)][col] = -1
        W[(k0+1)*n+(i0+0)][col] = 1; W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        
        # M4 = A[1,1] * (B[1,0]-B[0,0])
        U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+1)*p+(k0+0)][col] = 1; V[(j0+0)*p+(k0+0)][col] = -1
        W[(k0+0)*n+(i0+0)][col] = 1; W[(k0+0)*n+(i0+1)][col] = 1
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        U[(i0+0)*m+(j0+0)][col] = 1; U[(i0+0)*m+(j0+1)][col] = 1
        V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = -1; W[(k0+1)*n+(i0+0)][col] = 1
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        U[(i0+1)*m+(j0+0)][col] = 1; U[(i0+0)*m+(j0+0)][col] = -1
        V[(j0+0)*p+(k0+0)][col] = 1; V[(j0+0)*p+(k0+1)][col] = 1
        W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        U[(i0+0)*m+(j0+1)][col] = 1; U[(i0+1)*m+(j0+1)][col] = -1
        V[(j0+1)*p+(k0+0)][col] = 1; V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = 1
        col += 1
        
        return col
    
    # Block 1: A[0:2,0:1] × B[0:1,0:2] → C[0:2,0:2], 2x1x2, rank 2
    col = schoolbook(0, 0, 0, 2, 1, 2, U, V, W, col)
    
    # Block 2: A[0:2,1:5] × B[1:5,0:2] → C[0:2,0:2], 2x4x2
    # Split j=1:5 into j=1:3 and j=3:5
    # Block 2a: A[0:2,1:3] × B[1:3,0:2] → C[0:2,0:2], 2x2x2, Strassen rank 7
    col = strassen_2x2x2(0, 1, 0, U, V, W, col)
    # Block 2b: A[0:2,3:5] × B[3:5,0:2] → C[0:2,0:2], 2x2x2, Strassen rank 7
    col = strassen_2x2x2(0, 3, 0, U, V, W, col)
    
    # Block 3: A[0:2,0:1] × B[0:1,2:5] → C[0:2,2:5], 2x1x3, rank 3
    col = schoolbook(0, 0, 2, 2, 1, 3, U, V, W, col)
    
    # Block 4: A[0:2,1:5] × B[1:5,2:5] → C[0:2,2:5], 2x4x3
    # Split k=2:5 into k=2:3 and k=3:5
    # Block 4a: A[0:2,1:5] × B[1:5,2:3] → C[0:2,2:3], 2x4x1, rank 8
    col = schoolbook(0, 1, 2, 2, 4, 1, U, V, W, col)
    # Block 4b: A[0:2,1:5] × B[1:5,3:5] → C[0:2,3:5], 2x4x2
    # Split j=1:5 into j=1:3 and j=3:5
    # Block 4b1: A[0:2,1:3] × B[1:3,3:5] → C[0:2,3:5], 2x2x2, Strassen rank 7
    col = strassen_2x2x2(0, 1, 3, U, V, W, col)
    # Block 4b2: A[0:2,3:5] × B[3:5,3:5] → C[0:2,3:5], 2x2x2, Strassen rank 7
    col = strassen_2x2x2(0, 3, 3, U, V, W, col)
    
    # Block 5: A[2:3,0:1] × B[0:1,0:2] → C[2:3,0:2], 1x1x2, rank 2
    col = schoolbook(2, 0, 0, 1, 1, 2, U, V, W, col)
    
    # Block 6: A[2:3,1:5] × B[1:5,0:2] → C[2:3,0:2], 1x4x2, rank 8
    col = schoolbook(2, 1, 0, 1, 4, 2, U, V, W, col)
    
    # Block 7: A[2:3,0:1] × B[0:1,2:5] → C[2:3,2:5], 1x1x3, rank 3
    col = schoolbook(2, 0, 2, 1, 1, 3, U, V, W, col)
    
    # Block 8: A[2:3,1:5] × B[1:5,2:5] → C[2:3,2:5], 1x4x3, rank 12
    col = schoolbook(2, 1, 2, 1, 4, 3, U, V, W, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
