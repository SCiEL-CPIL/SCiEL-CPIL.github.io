"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    # We'll use 4 Strassen 2x2x2 blocks, each of rank 7, total rank 28
    rank = 28
    
    # U has shape (n*m, rank) = (8, 28)
    # V has shape (m*p, rank) = (16, 28)
    # W has shape (p*n, rank) = (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A is 2x4, index: A[i,j] -> U index i*m+j = i*4+j
    # B is 4x4, index: B[j,k] -> V index j*p+k = j*4+k
    # C is 2x4, index: C[i,k] -> W index k*n+i = k*2+i
    
    # Split A into A_left (cols 0,1) and A_right (cols 2,3)
    # A_left[i,j] for i in 0,1; j in 0,1 -> U indices: i*4+j for j in 0,1
    # A_right[i,j] for i in 0,1; j in 0,1 -> U indices: i*4+(j+2)
    
    # Split B into 4 2x2 blocks:
    # B00: rows 0,1 cols 0,1 -> V indices: 0,1,4,5 (j*4+k)
    # B01: rows 0,1 cols 2,3 -> V indices: 2,3,6,7
    # B10: rows 2,3 cols 0,1 -> V indices: 8,9,12,13
    # B11: rows 2,3 cols 2,3 -> V indices: 10,11,14,15
    
    # C_left: cols 0,1 -> W indices: k*2+i for k in 0,1
    #   C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3]
    # C_right: cols 2,3 -> W indices: k*2+i for k in 2,3
    #   C[0,2]=W[4], C[1,2]=W[5], C[0,3]=W[6], C[1,3]=W[7]
    
    # Strassen 2x2x2 coefficients
    # For X (2x2) * Y (2x2) = Z (2x2):
    # p0 = (X00+X11)*(Y00+Y11)
    # p1 = (X10+X11)*Y00
    # p2 = X00*(Y01-Y11)
    # p3 = X11*(Y10-Y00)
    # p4 = (X00+X01)*Y11
    # p5 = (X10-X00)*(Y00+Y01)
    # p6 = (X01-X11)*(Y10+Y11)
    # Z00 = p0 + p3 - p4 + p6
    # Z01 = p2 + p4
    # Z10 = p1 + p5
    # Z11 = p0 - p1 + p2 + p3
    
    def add_strassen_block(u_idx, v_idx, w_idx, col_start):
        """
        u_idx: list of 4 U indices for X00,X01,X10,X11 (row-major: [X00, X01, X10, X11])
        v_idx: list of 4 V indices for Y00,Y01,Y10,Y11 (row-major: [Y00, Y01, Y10, Y11])
        w_idx: list of 4 W indices for Z00,Z01,Z10,Z11 (row-major: [Z00, Z01, Z10, Z11])
        col_start: starting column
        """
        # p0 = (X00+X11)*(Y00+Y11)
        U[u_idx[0]][col_start + 0] += 1
        U[u_idx[3]][col_start + 0] += 1
        V[v_idx[0]][col_start + 0] += 1
        V[v_idx[3]][col_start + 0] += 1
        # Z00 += p0
        W[w_idx[0]][col_start + 0] += 1
        # Z11 += p0
        W[w_idx[3]][col_start + 0] += 1
        
        # p1 = (X10+X11)*Y00
        U[u_idx[2]][col_start + 1] += 1
        U[u_idx[3]][col_start + 1] += 1
        V[v_idx[0]][col_start + 1] += 1
        # Z10 += p1
        W[w_idx[2]][col_start + 1] += 1
        # Z11 -= p1
        W[w_idx[3]][col_start + 1] += -1
        
        # p2 = X00*(Y01-Y11)
        U[u_idx[0]][col_start + 2] += 1
        V[v_idx[1]][col_start + 2] += 1
        V[v_idx[3]][col_start + 2] += -1
        # Z01 += p2
        W[w_idx[1]][col_start + 2] += 1
        # Z11 += p2
        W[w_idx[3]][col_start + 2] += 1
        
        # p3 = X11*(Y10-Y00)
        U[u_idx[3]][col_start + 3] += 1
        V[v_idx[2]][col_start + 3] += 1
        V[v_idx[0]][col_start + 3] += -1
        # Z00 += p3
        W[w_idx[0]][col_start + 3] += 1
        # Z11 += p3
        W[w_idx[3]][col_start + 3] += 1
        
        # p4 = (X00+X01)*Y11
        U[u_idx[0]][col_start + 4] += 1
        U[u_idx[1]][col_start + 4] += 1
        V[v_idx[3]][col_start + 4] += 1
        # Z01 += p4
        W[w_idx[1]][col_start + 4] += 1
        # Z00 -= p4
        W[w_idx[0]][col_start + 4] += -1
        
        # p5 = (X10-X00)*(Y00+Y01)
        U[u_idx[2]][col_start + 5] += 1
        U[u_idx[0]][col_start + 5] += -1
        V[v_idx[0]][col_start + 5] += 1
        V[v_idx[1]][col_start + 5] += 1
        # Z10 += p5
        W[w_idx[2]][col_start + 5] += 1
        
        # p6 = (X01-X11)*(Y10+Y11)
        U[u_idx[1]][col_start + 6] += 1
        U[u_idx[3]][col_start + 6] += -1
        V[v_idx[2]][col_start + 6] += 1
        V[v_idx[3]][col_start + 6] += 1
        # Z00 += p6
        W[w_idx[0]][col_start + 6] += 1
    
    # Block 0: A_left * B00 -> C_left
    # A_left: X00=U[0], X01=U[1], X10=U[4], X11=U[5]
    # B00: Y00=V[0], Y01=V[1], Y10=V[4], Y11=V[5]
    # C_left: Z00=W[0], Z01=W[2], Z10=W[1], Z11=W[3]
    u_block0 = [0, 1, 4, 5]  # X00, X01, X10, X11
    v_block0 = [0, 1, 4, 5]  # Y00, Y01, Y10, Y11
    w_block0 = [0, 2, 1, 3]  # Z00, Z01, Z10, Z11
    add_strassen_block(u_block0, v_block0, w_block0, 0)
    
    # Block 1: A_right * B10 -> C_left
    # A_right: X00=U[2], X01=U[3], X10=U[6], X11=U[7]
    # B10: Y00=V[8], Y01=V[9], Y10=V[12], Y11=V[13]
    # C_left: Z00=W[0], Z01=W[2], Z10=W[1], Z11=W[3]
    u_block1 = [2, 3, 6, 7]
    v_block1 = [8, 9, 12, 13]
    w_block1 = [0, 2, 1, 3]
    add_strassen_block(u_block1, v_block1, 7)
    
    # Block 2: A_left * B01 -> C_right
    # A_left: X00=U[0], X01=U[1], X10=U[4], X11=U[5]
    # B01: Y00=V[2], Y01=V[3], Y10=V[6], Y11=V[7]
    # C_right: Z00=W[4], Z01=W[6], Z10=W[5], Z11=W[7]
    u_block2 = [0, 1, 4, 5]
    v_block2 = [2, 3, 6, 7]
    w_block2 = [4, 6, 5, 7]
    add_strassen_block(u_block2, v_block2, 14)
    
    # Block 3: A_right * B11 -> C_right
    # A_right: X00=U[2], X01=U[3], X10=U[6], X11=U[7]
    # B11: Y00=V[10], Y01=V[11], Y10=V[14], Y11=V[15]
    # C_right: Z00=W[4], Z01=W[6], Z10=W[5], Z11=W[7]
    u_block3 = [2, 3, 6, 7]
    v_block3 = [10, 11, 14, 15]
    w_block3 = [4, 6, 5, 7]
    add_strassen_block(u_block3, v_block3, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
