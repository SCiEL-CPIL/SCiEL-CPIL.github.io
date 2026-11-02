"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    
    U = [[0] * 54 for _ in range(n * m)]  # 10 x 54
    V = [[0] * 54 for _ in range(m * p)]  # 30 x 54
    W = [[0] * 54 for _ in range(p * n)]  # 12 x 54
    
    col = 0
    
    # Strassen decomposition for 2x2x2 blocks
    def strassen_2x2x2(base_U, base_V, base_W, col, 
                       U_idx, V_idx, W_idx):
        """
        Add Strassen terms for a 2x2x2 block.
        U_idx: list of 2 indices for U (rows)
        V_idx: list of 2 indices for V (rows)
        W_idx: list of 2 indices for W (rows)
        """
        nonlocal col
        
        # M1 = A[0,0] * (B[0,0] + B[0,1])
        # U: U_idx[0], V: V_idx[0] and V_idx[0]+1, W: W_idx[0]
        U[U_idx[0]][col] = 1
        V[V_idx[0]][col] = 1
        V[V_idx[0]+1][col] = 1
        W[W_idx[0]][col] = 1
        col += 1
        
        # M2 = (A[0,0] + A[1,0]) * B[1,0]
        # U: U_idx[0] and U_idx[0]+1, V: V_idx[1], W: W_idx[0]
        U[U_idx[0]][col] = 1
        U[U_idx[1]][col] = 1
        V[V_idx[1]][col] = 1
        W[W_idx[0]][col] = 1
        col += 1
        
        # M3 = A[1,0] * (B[1,0] + B[1,1])
        # U: U_idx[1], V: V_idx[1] and V_idx[1]+1, W: W_idx[0]
        U[U_idx[1]][col] = 1
        V[V_idx[1]][col] = 1
        V[V_idx[1]+1][col] = 1
        W[W_idx[0]][col] = 1
        col += 1
        
        # M4 = (A[0,1] - A[1,1]) * B[1,1]
        # U: U_idx[0]+1 and U_idx[1]+1, V: V_idx[1]+1, W: W_idx[1]
        U[U_idx[0]+1][col] = 1
        U[U_idx[1]+1][col] = -1
        V[V_idx[1]+1][col] = 1
        W[W_idx[1]][col] = 1
        col += 1
        
        # M5 = (A[0,0] + A[0,1]) * B[0,1]
        # U: U_idx[0] and U_idx[0]+1, V: V_idx[0]+1, W: W_idx[1]
        U[U_idx[0]][col] = 1
        U[U_idx[0]+1][col] = 1
        V[V_idx[0]+1][col] = 1
        W[W_idx[1]][col] = 1
        col += 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        # U: U_idx[1] and U_idx[0], V: V_idx[0] and V_idx[0]+1, W: W_idx[0]
        U[U_idx[1]][col] = 1
        U[U_idx[0]][col] = -1
        V[V_idx[0]][col] = 1
        V[V_idx[0]+1][col] = 1
        W[W_idx[0]][col] = 1
        col += 1
        
        # M7 = (A[1,1] - A[0,0]) * (B[0,0] + B[1,0])
        # U: U_idx[1]+1 and U_idx[0], V: V_idx[0] and V_idx[1], W: W_idx[1]
        U[U_idx[1]+1][col] = 1
        U[U_idx[0]][col] = -1
        V[V_idx[0]][col] = 1
        V[V_idx[1]][col] = 1
        W[W_idx[1]][col] = 1
        col += 1
        
        return col
    
    # Split m=5 into m1=2 and m2=3
    # Split p=6 into three blocks of p=2 each
    # Block 1: i in [0,1], j in [0,1], k in [0,1] -> 2x2x2, rank 7
    # Block 2: i in [0,1], j in [0,1], k in [2,3] -> 2x2x2, rank 7
    # Block 3: i in [0,1], j in [0,1], k in [4,5] -> 2x2x2, rank 7
    # Block 4: i in [0,1], j in [2], k in [0,1] -> 2x1x2, rank 2
    # Block 5: i in [0,1], j in [2], k in [2,3] -> 2x1x2, rank 2
    # Block 6: i in [0,1], j in [2], k in [4,5] -> 2x1x2, rank 2
    # Block 7: i in [0,1], j in [3,4], k in [0,1] -> 2x2x2, rank 7
    # Block 8: i in [0,1], j in [3,4], k in [2,3] -> 2x2x2, rank 7
    # Block 9: i in [0,1], j in [3,4], k in [4,5] -> 2x2x2, rank 7
    
    # Total rank = 7*6 + 2*3 = 42 + 6 = 48? 
    # Wait, m=5 split into 2+3. 
    # j in [0,1] is m1=2.
    # j in [2,3,4] is m2=3.
    # m2 can be split into 1+2: j in [2] (m=1) and j in [3,4] (m=2).
    # So:
    # j in [0,1] (m=2): 3 blocks of 2x2x2 -> 3*7 = 21
    # j in [2] (m=1): 3 blocks of 2x1x2 -> 3*2 = 6
    # j in [3,4] (m=2): 3 blocks of 2x2x2 -> 3*7 = 21
    # Total = 21 + 6 + 21 = 48
    
    # Let's re-verify the indices.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Block 1: i=[0,1], j=[0,1], k=[0,1]
    # U_idx = [0*5+0, 1*5+0] = [0, 5]
    # V_idx = [0*6+0, 1*6+0] = [0, 6]
    # W_idx = [0*2+0, 1*2+0] = [0, 2]
    col = strassen_2x2x2(U, V, W, col, [0, 5], [0, 6], [0, 2])
    
    # Block 2: i=[0,1], j=[0,1], k=[2,3]
    # U_idx = [0, 5]
    # V_idx = [0*6+2, 1*6+2] = [2, 8]
    # W_idx = [2*2+0, 3*2+0] = [4, 6]
    col = strassen_2x2x2(U, V, W, col, [0, 5], [2, 8], [4, 6])
    
    # Block 3: i=[0,1], j=[0,1], k=[4,5]
    # U_idx = [0, 5]
    # V_idx = [0*6+4, 1*6+4] = [4, 10]
    # W_idx = [4*2+0, 5*2+0] = [8, 10]
    col = strassen_2x2x2(U, V, W, col, [0, 5], [4, 10], [8, 10])
    
    # Block 4: i=[0,1], j=[2], k=[0,1]
    # This is 2x1x2. Rank 2.
    # C[0,0] = A[0,2]*B[2,0], C[0,1] = A[0,2]*B[2,1]
    # C[1,0] = A[1,2]*B[2,0], C[1,1] = A[1,2]*B[2,1]
    # Term 1: A[0,2]*B[2,0] -> W[0,0]
    U[0*5+2][col] = 1
    V[2*6+0][col] = 1
    W[0*2+0][col] = 1
    col += 1
    # Term 2: A[0,2]*B[2,1] -> W[1,0]
    U[0*5+2][col] = 1
    V[2*6+1][col] = 1
    W[1*2+0][col] = 1
    col += 1
    # Term 3: A[1,2]*B[2,0] -> W[0,1]
    U[1*5+2][col] = 1
    V[2*6+0][col] = 1
    W[0*2+1][col] = 1
    col += 1
    # Term 4: A[1,2]*B[2,1] -> W[1,1]
    U[1*5+2][col] = 1
    V[2*6+1][col] = 1
    W[1*2+1][col] = 1
    col += 1
    
    # Block 5: i=[0,1], j=[2], k=[2,3]
    U[0*5+2][col] = 1
    V[2*6+2][col] = 1
    W[2*2+0][col] = 1
    col += 1
    U[0*5+2][col] = 1
    V[2*6+3][col] = 1
    W[3*2+0][col] = 1
    col += 1
    U[1*5+2][col] = 1
    V[2*6+2][col] = 1
    W[2*2+1][col] = 1
    col += 1
    U[1*5+2][col] = 1
    V[2*6+3][col] = 1
    W[3*2+1][col] = 1
    col += 1
    
    # Block 6: i=[0,1], j=[2], k=[4,5]
    U[0*5+2][col] = 1
    V[2*6+4][col] = 1
    W[4*2+0][col] = 1
    col += 1
    U[0*5+2][col] = 1
    V[2*6+5][col] = 1
    W[5*2+0][col] = 1
    col += 1
    U[1*5+2][col] = 1
    V[2*6+4][col] = 1
    W[4*2+1][col] = 1
    col += 1
    U[1*5+2][col] = 1
    V[2*6+5][col] = 1
    W[5*2+1][col] = 1
    col += 1
    
    # Block 7: i=[0,1], j=[3,4], k=[0,1]
    # U_idx = [0*5+3, 1*5+3] = [3, 8]
    # V_idx = [3*6+0, 4*6+0] = [18, 24]
    # W_idx = [0*2+0, 1*2+0] = [0, 2]
    col = strassen_2x2x2(U, V, W, col, [3, 8], [18, 24], [0, 2])
    
    # Block 8: i=[0,1], j=[3,4], k=[2,3]
    # U_idx = [3, 8]
    # V_idx = [3*6+2, 4*6+2] = [20, 26]
    # W_idx = [2*2+0, 3*2+0] = [4, 6]
    col = strassen_2x2x2(U, V, W, col, [3, 8], [20, 26], [4, 6])
    
    # Block 9: i=[0,1], j=[3,4], k=[4,5]
    # U_idx = [3, 8]
    # V_idx = [3*6+4, 4*6+4] = [22, 28]
    # W_idx = [4*2+0, 5*2+0] = [8, 10]
    col = strassen_2x2x2(U, V, W, col, [3, 8], [22, 28], [8, 10])
    
    # Total rank should be 48.
    # Let's check if col == 48.
    # 7*3 + 2*3 + 7*3 = 21 + 6 + 21 = 48.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
