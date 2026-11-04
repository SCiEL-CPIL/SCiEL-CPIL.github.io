import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = 74
    
    U = [[0] * rank for _ in range(n * m)]  # 15 x 74
    V = [[0] * rank for _ in range(m * p)]  # 25 x 74
    W = [[0] * rank for _ in range(p * n)]  # 15 x 74
    
    col = 0
    
    # Helper for schoolbook sub-block
    # A_sub is n_s x m_s, B_sub is m_s x p_s
    # Global indices: A rows i0..i0+n_s-1, B cols k0..k0+p_s-1, inner j0..j0+m_s-1
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
    
    # Block 1: A11 (2x2) x B11 (2x2) -> C11 (2x2), Strassen rank 7
    # i0=0, j0=0, k0=0, n_s=2, m_s=2, p_s=2
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U[0*m+0][col] = 1; U[1*m+1][col] = 1
    V[0*p+0][col] = 1; V[1*p+1][col] = 1
    W[0*n+0][col] = 1; W[1*n+1][col] = 1
    col += 1
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    U[1*m+0][col] = 1; U[1*m+1][col] = 1
    V[0*p+0][col] = 1
    W[0*n+1][col] = 1; W[1*n+1][col] = -1
    col += 1
    
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    U[0*m+0][col] = 1
    V[0*p+1][col] = 1; V[1*p+1][col] = -1
    W[1*n+0][col] = 1; W[1*n+1][col] = 1
    col += 1
    
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    U[1*m+1][col] = 1
    V[1*p+0][col] = 1; V[0*p+0][col] = -1
    W[0*n+0][col] = 1; W[0*n+1][col] = 1
    col += 1
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    U[0*m+0][col] = 1; U[0*m+1][col] = 1
    V[1*p+1][col] = 1
    W[0*n+0][col] = -1; W[1*n+0][col] = 1
    col += 1
    
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[1*m+0][col] = 1; U[0*m+0][col] = -1
    V[0*p+0][col] = 1; V[0*p+1][col] = 1
    W[1*n+1][col] = 1
    col += 1
    
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[0*m+1][col] = 1; U[1*m+1][col] = -1
    V[1*p+0][col] = 1; V[1*p+1][col] = 1
    W[0*n+0][col] = 1
    col += 1
    
    # Block 2: A12 (2x3) x B21 (3x2) -> C11 (2x2), schoolbook rank 12
    # i0=0, j0=2, k0=0, n_s=2, m_s=3, p_s=2
    col = schoolbook(0, 2, 0, 2, 3, 2, U, V, W, col)
    
    # Block 3: A11 (2x2) x B12 (2x3) -> C12 (2x3), schoolbook rank 12
    # i0=0, j0=0, k0=2, n_s=2, m_s=2, p_s=3
    col = schoolbook(0, 0, 2, 2, 2, 3, U, V, W, col)
    
    # Block 4: A12 (2x3) x B22 (3x3) -> C12 (2x3), schoolbook rank 18
    # i0=0, j0=2, k0=2, n_s=2, m_s=3, p_s=3
    col = schoolbook(0, 2, 2, 2, 3, 3, U, V, W, col)
    
    # Block 5: A21 (1x2) x B11 (2x2) -> C21 (1x2), schoolbook rank 4
    # i0=2, j0=0, k0=0, n_s=1, m_s=2, p_s=2
    col = schoolbook(2, 0, 0, 1, 2, 2, U, V, W, col)
    
    # Block 6: A22 (1x3) x B21 (3x2) -> C21 (1x2), schoolbook rank 6
    # i0=2, j0=2, k0=0, n_s=1, m_s=3, p_s=2
    col = schoolbook(2, 2, 0, 1, 3, 2, U, V, W, col)
    
    # Block 7: A21 (1x2) x B12 (2x3) -> C22 (1x3), schoolbook rank 6
    # i0=2, j0=0, k0=2, n_s=1, m_s=2, p_s=3
    col = schoolbook(2, 0, 2, 1, 2, 3, U, V, W, col)
    
    # Block 8: A22 (1x3) x B22 (3x3) -> C22 (1x3), schoolbook rank 9
    # i0=2, j0=2, k0=2, n_s=1, m_s=3, p_s=3
    col = schoolbook(2, 2, 2, 1, 3, 3, U, V, W, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
