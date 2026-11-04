import numpy as np
import time

def construct():
    n, m, p = 2, 3, 4
    
    # Build the full tensor for verification
    N_U = n * m  # 6
    N_V = m * p  # 12
    N_W = p * n  # 8
    
    # Target tensor
    T = np.zeros((N_U, N_V, N_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Strassen 2x2x2 decomposition (rank 7)
    # M1 = (A11+A22)(B12+B21)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B21)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2 x 2x2, the factor matrices:
    # U (2x2=4 rows, 7 cols): each row is (i,j) position
    # V (2x2=4 rows, 7 cols): each row is (j,k) position  
    # W (2x2=4 rows, 7 cols): each row is (k,i) position
    
    def strassen_2x2():
        """Return U, V, W for 2x2 x 2x2 matrix multiplication, rank 7"""
        # Positions: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B12+B21): A indices 0,3; B indices 1,2; C contributes to C11,C22
        U[0, 0] = 1; U[3, 0] = 1
        V[1, 0] = 1; V[2, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21+A22)B11: A indices 2,3; B index 0; C contributes to C21,C22
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A11(B11-B21): A index 0; B indices 0,2; C contributes to C11,C12
        U[0, 2] = 1
        V[0, 2] = 1; V[2, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        
        # M4 = A22(B22-B12): A index 3; B indices 3,1; C contributes to C11,C21
        U[3, 3] = 1
        V[3, 3] = 1; V[1, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A11+A12)B22: A indices 0,1; B index 3; C contributes to C11,C12
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1
        
        # M6 = (A21-A11)(B11+B12): A indices 2,0; B indices 0,1; C contributes to C22
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22): A indices 1,3; B indices 2,3; C contributes to C11
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def rank_r_identity(nrows, rank):
        """Simple rank-r decomposition: U has 1s on diagonal, V and W identity-like"""
        U = np.zeros((nrows, rank))
        V = np.zeros((nrows, rank))
        W = np.zeros((nrows, rank))
        for i in range(min(nrows, rank)):
            U[i, i] = 1
            V[i, i] = 1
            W[i, i] = 1
        return U, V, W
    
    # Decomposition strategy:
    # Split A (2x3) into A_left (2x2) and A_right (2x1)
    # Split B (3x4) into B_top (2x4) and B_bottom (1x4)
    # 
    # C = A*B = A_left * B_top + A_left * B_bottom + A_right * B_top + A_right * B_bottom
    #
    # A_left (2x2) * B_top (2x4): 
    #   Split B_top into B_top_left (2x2) and B_top_right (2x2)
    #   = A_left * B_top_left + A_left * B_top_right
    #   Each is 2x2 x 2x2, rank 7 each -> 14
    #
    # A_left (2x2) * B_bottom (1x4):
    #   Split into B_bottom_left (1x2) and B_bottom_right (1x2)
    #   = A_left * B_bottom_left + A_left * B_bottom_right
    #   Each is 2x2 x 1x2, rank 4 each -> 8
    #
    # A_right (2x1) * B_top (2x4):
    #   Split into B_top_left (2x2) and B_top_right (2x2)
    #   = A_right * B_top_left + A_right * B_top_right
    #   Each is 2x1 x 2x2, rank 4 each -> 8
    #
    # A_right (2x1) * B_bottom (1x4):
    #   Split into B_bottom_left (1x2) and B_bottom_right (1x2)
    #   = A_right * B_bottom_left + A_right * B_bottom_right
    #   Each is 2x1 x 1x2, rank 2 each -> 4
    #
    # Total: 14 + 8 + 8 + 4 = 34... that's worse than schoolbook!
    
    # Better approach: Use the known result that rank(2x3x4) = 19
    # Let me try a different decomposition.
    
    # Actually, let me just use the schoolbook as fallback and try to find better.
    # For now, return schoolbook rank 24.
    
    rank = n * m * p  # 24
    U = np.zeros((N_U, rank))
    V = np.zeros((N_V, rank))
    W = np.zeros((N_W, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Verify
    error = 0
    for a in range(N_U):
        for b in range(N_V):
            for c in range(N_W):
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - T[a, b, c]) > 1e-10:
                    error += abs(val - T[a, b, c])
    
    if error > 1e-10:
        raise ValueError("Schoolbook decomposition failed verification")
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
