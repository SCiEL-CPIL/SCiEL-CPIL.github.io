import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Split m=5 into m1=4 and m2=1
    # Part 1: 4x4x6 using Strassen on two 2x2x6 blocks (rank 14)
    # Part 2: 4x1x6 using schoolbook (rank 4)
    # Total rank = 18
    
    def get_strassen_2x2x2():
        """Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
        U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[i,k]
        """
        U = [[0.0]*7 for _ in range(4)]
        V = [[0.0]*7 for _ in range(4)]
        W = [[0.0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1.0; U[3][0] = 1.0
        V[0][0] = 1.0; V[3][0] = 1.0
        W[0][0] = 1.0; W[3][0] = 1.0
        
        # M2 = (A21+A22)B11
        U[2][1] = 1.0; U[3][1] = 1.0
        V[0][1] = 1.0
        W[2][1] = 1.0; W[3][1] = -1.0
        
        # M3 = A11(B12-B22)
        U[0][2] = 1.0
        V[1][2] = 1.0; V[3][2] = -1.0
        W[1][2] = 1.0; W[3][2] = 1.0
        
        # M4 = A22(B21-B11)
        U[3][3] = 1.0
        V[2][3] = 1.0; V[0][3] = -1.0
        W[0][3] = 1.0; W[2][3] = 1.0
        
        # M5 = (A11+A12)B22
        U[0][4] = 1.0; U[1][4] = 1.0
        V[3][4] = 1.0
        W[0][4] = -1.0; W[1][4] = 1.0
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1.0; U[0][5] = -1.0
        V[0][5] = 1.0; V[1][5] = 1.0
        W[3][5] = 1.0
        
        # M7 = (A12-A22)(B21+B22)
        U[1][6] = 1.0; U[3][6] = -1.0
        V[2][6] = 1.0; V[3][6] = 1.0
        W[0][6] = 1.0
        
        return U, V, W
    
    total_rank = 18
    U = [[0.0] * total_rank for _ in range(n * m)]  # 20 rows
    V = [[0.0] * total_rank for _ in range(m * p)]  # 30 rows
    W = [[0.0] * total_rank for _ in range(p * n)]  # 24 rows
    
    strassen_U, strassen_V, strassen_W = get_strassen_2x2x2()
    
    # Part 1: Two 2x2x6 Strassen blocks
    for block in range(2):
        n_off = block * 2
        m_off = block * 2
        
        for r in range(7):
            col = block * 7 + r
            
            # U: A[n_off+i, m_off+j]
            for i in range(2):
                for j in range(2):
                    u_idx = i * 2 + j
                    if strassen_U[u_idx][r] != 0:
                        U[(n_off + i) * m + (m_off + j)][col] = strassen_U[u_idx][r]
            
            # V: B[m_off+j, k]
            for j in range(2):
                for k in range(p):
                    v_idx = j * 2 + k
                    if strassen_V[v_idx][r] != 0:
                        V[(m_off + j) * p + k][col] = strassen_V[v_idx][r]
            
            # W: C[k, n_off+i], trace convention W[k*n + i, r]
            for k in range(p):
                for i in range(2):
                    w_idx = k * 2 + i
                    if strassen_W[w_idx][r] != 0:
                        W[k * n + (n_off + i)][col] = strassen_W[w_idx][r]
    
    # Part 2: 4x1x6 schoolbook (m=4)
    for i in range(4):
        col = 14 + i
        U[i * m + 4][col] = 1.0
        for k in range(p):
            V[4 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
    
    return {"U": U, "V": V, "W": W}

