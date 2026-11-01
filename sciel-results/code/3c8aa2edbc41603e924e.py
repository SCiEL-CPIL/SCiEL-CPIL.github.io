import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def check_decomposition(U, V, W):
        """Check if U,V,W form a valid decomposition."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # M_{2,3,2} decomposition (rank 8)
    def m_2x3x2_decomposition():
        """Return U (6x8), V (6x8), W (4x8) for M_{2,3,2}."""
        # U: 6x8, V: 6x8, W: 4x8
        U = [[0] * 8 for _ in range(6)]
        V = [[0] * 8 for _ in range(6)]
        W = [[0] * 8 for _ in range(4)]
        
        # Strassen part (rank 7) for the 2x2x2 core
        # A is 2x2, B is 2x2 (rows 0,1 of the 3x2 B)
        # U rows: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # V rows: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
        # W rows: p11=C[0,0], p12=C[0,1], p21=C[1,0], p22=C[1,1]
        
        U_s = [
            [1, 0, 1, 0, 1, 0, 1],   # a
            [0, 0, 0, 0, 1, 0, -1],  # b
            [0, 1, 0, 0, 0, 1, 0],   # c
            [1, 1, 0, 1, 0, -1, 0],  # d
        ]
        V_s = [
            [1, 1, 0, 0, 0, 0, 1],   # e
            [0, 0, 1, 0, 0, 0, 1],   # f
            [0, 0, -1, 0, 0, 1, 0],  # g
            [1, 0, 0, 1, 1, 1, 0],   # h
        ]
        W_s = [
            [1, 0, 0, 0, -1, 0, 1],   # p11
            [0, 0, 1, 0, 1, 0, 0],    # p12
            [0, 1, 0, 1, 0, 0, 0],    # p21
            [1, -1, 1, 0, 0, 1, 0],   # p22
        ]
        
        # Fill Strassen part
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    U[i*3+j][c] = U_s[i*2+j][c]
        
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    V[j*2+k][c] = V_s[j*2+k][c]
        
        for k in range(2):
            for i in range(2):
                for c in range(7):
                    W[k*2+i][c] = W_s[k*2+i][c]
        
        # Add 1 rank-1 term for j=2 (3rd row of B)
        U[0*3+2][7] = 1
        U[1*3+2][7] = 1
        V[2*2+0][7] = 1
        V[2*2+1][7] = 1
        W[0*2+0][7] = 1
        W[1*2+0][7] = 1
        W[0*2+1][7] = 1
        W[1*2+1][7] = 1
        
        return U, V, W
    
    # M_{2,3,1} decomposition (rank 3)
    def m_2x3x1_decomposition():
        """Return U (6x3), V (3x3), W (2x3) for M_{2,3,1}."""
        # A is 2x3, B is 3x1, C is 2x1
        # C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        
        U = [[0] * 3 for _ in range(6)]
        V = [[0] * 3 for _ in range(3)]
        W = [[0] * 3 for _ in range(2)]
        
        # Term 0: A[*,0] * B[0,0] -> C[*,0]
        U[0*3+0][0] = 1
        U[1*3+0][0] = 1
        V[0*1+0][0] = 1
        W[0*2+0][0] = 1
        W[0*2+1][0] = 1
        
        # Term 1: A[*,1] * B[1,0] -> C[*,0]
        U[0*3+1][1] = 1
        U[1*3+1][1] = 1
        V[1*1+0][1] = 1
        W[1*2+0][1] = 1
        W[1*2+1][1] = 1
        
        # Term 2: A[*,2] * B[2,0] -> C[*,0]
        U[0*3+2][2] = 1
        U[1*3+2][2] = 1
        V[2*1+0][2] = 1
        W[2*2+0][2] = 1
        W[2*2+1][2] = 1
        
        return U, V, W
    
    # M_{2,3,3} decomposition (rank 11)
    def m_2x3x3_decomposition():
        """Return U (6x11), V (9x11), W (6x11) for M_{2,3,3}."""
        U_232, V_232, W_232 = m_2x3x2_decomposition()
        U_231, V_231, W_231 = m_2x3x1_decomposition()
        
        rank = 11
        U = [[0] * rank for _ in range(6)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(6)]
        
        # M_{2,3,2} part (columns 0-6): k=0,1
        for i in range(6):
            for c in range(7):
                U[i][c] = U_232[i][c]
        
        for j in range(3):
            for k in range(2):
                for c in range(7):
                    V[j*3+k][c] = V_232[j*2+k][c]
        
        for k in range(2):
            for i in range(2):
                for c in range(7):
                    W[k*2+i][c] = W_232[k*2+i][c]
        
        # M_{2,3,1} part (columns 7-10): k=2
        for i in range(6):
            for c in range(3):
                U[i][7+c] = U_231[i][c]
        
        for j in range(3):
            for c in range(3):
                V[j*3+2][7+c] = V_231[j][c]
        
        for i in range(2):
            for c in range(3):
                W[2*2+i][7+c] = W_231[i][c]
        
        return U, V, W
    
    # Assemble M_{2,3,8} as M_{2,3,3} + M_{2,3,3} + M_{2,3,2}
    # Total rank: 11 + 11 + 8 = 30
    
    total_rank = 30
    U_final = [[0] * total_rank for _ in range(6)]
    V_final = [[0] * total_rank for _ in range(24)]
    W_final = [[0] * total_rank for _ in range(16)]
    
    # Block 0: M_{2,3,3} for k=0,1,2
    U_233, V_233, W_233 = m_2x3x3_decomposition()
    for i in range(6):
        for c in range(11):
            U_final[i][c] = U_233[i][c]
    for j in range(3):
        for k in range(3):
            for c in range(11):
                V_final[j*8+k][c] = V_233[j*3+k][c]
    for k in range(3):
        for i in range(2):
            for c in range(11):
                W_final[k*2+i][c] = W_233[k*2+i][c]
    
    # Block 1: M_{2,3,3} for k=3,4,5
    for i in range(6):
        for c in range(11):
            U_final[i][11+c] = U_233[i][c]
    for j in range(3):
        for k in range(3):
            for c in range(11):
                V_final[j*8+(3+k)][11+c] = V_233[j*3+k][c]
    for k in range(3):
        for i in range(2):
            for c in range(11):
                W_final[(3+k)*2+i][11+c] = W_233[k*2+i][c]
    
    # Block 2: M_{2,3,2} for k=6,7
    U_232, V_232, W_232 = m_2x3x2_decomposition()
    for i in range(6):
        for c in range(8):
            U_final[i][22+c] = U_232[i][c]
    for j in range(3):
        for k in range(2):
            for c in range(8):
                V_final[j*8+(6+k)][22+c] = V_232[j*2+k][c]
    for k in range(2):
        for i in range(2):
            for c in range(8):
                W_final[(6+k)*2+i][22+c] = W_232[k*2+i][c]
    
    if check_decomposition(U_final, V_final, W_final):
        return {"U": U_final, "V": V_final, "W": W_final}
    
    return best_result
# EVOLVE-BLOCK-END
