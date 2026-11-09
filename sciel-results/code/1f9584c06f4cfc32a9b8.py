import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2_2x2():
    """Return U, V, W for 2x2 x 2x2 Strassen decomposition (rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    U indexed by (i,j) -> i*2+j
    V indexed by (j,k) -> j*2+k
    W indexed by (k,i) -> k*2+i (trace convention)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    # a=U[0], b=U[1], c=U[2], d=U[3]
    # e=V[0], f=V[1], g=V[2], h=V[3]
    # C11=W[0], C21=W[1], C12=W[2], C22=W[3]
    U[0][0] = 1; U[3][0] = 1  # a+d
    V[0][0] = 1; V[1][0] = 1; V[2][0] = 1  # e+h... wait
    # e=V[0*2+0]=V[0], f=V[0*2+1]=V[1], g=V[1*2+0]=V[2], h=V[1*2+1]=V[3]
    # e+h = V[0] + V[3]
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C11 + C22
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1  # c+d
    V[0][1] = 1  # e
    W[1][1] = 1; W[3][1] = -1  # C21 - C22
    
    # M3 = a(f-h)
    U[0][2] = 1  # a
    V[1][2] = 1; V[3][2] = -1  # f-h
    W[2][2] = 1; W[3][2] = 1  # C12 + C22
    
    # M4 = d(g-e)
    U[3][3] = 1  # d
    V[2][3] = 1; V[0][3] = -1  # g-e
    W[0][3] = 1; W[1][3] = 1  # C11 + C21
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1  # a+b
    V[3][4] = 1  # h
    W[0][4] = -1; W[2][4] = 1  # -C11 + C12
    
    # M6 = (c-a)(e+f)
    U[0][5] = -1; U[2][5] = 1  # c-a
    V[0][5] = 1; V[1][5] = 1  # e+f
    W[3][5] = 1  # C22
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1  # b-d
    V[2][6] = 1; V[3][6] = 1  # g+h
    W[0][6] = 1  # C11
    
    return U, V, W

def _strassen_4x4_4x4():
    """Return U, V, W for 4x4 x 4x4 Strassen decomposition (rank 49).
    U: (16, 49), V: (16, 49), W: (16, 49)
    U indexed by (i,j) -> i*4+j
    V indexed by (j,k) -> j*4+k
    W indexed by (k,i) -> k*4+i (trace convention)
    """
    U2, V2, W2 = _strassen_2x2_2x2()
    
    # Block-level Strassen coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A blocks: A11=(0:2,0:2), A12=(0:2,2:4), A21=(2:4,0:2), A22=(2:4,2:4)
    # B blocks: B11=(0:2,0:2), B12=(0:2,2:4), B21=(2:4,0:2), B22=(2:4,2:4)
    # C blocks: C11=(0:2,0:2), C12=(0:2,2:4), C21=(2:4,0:2), C22=(2:4,2:4)
    
    # For each block-level term t (0..6), we have:
    # A_t = sum of A blocks with coefficients
    # B_t = sum of B blocks with coefficients
    # C_t = sum of C blocks with coefficients
    # Then A_t * B_t contributes to C with coefficients
    
    # Define block-level coefficients
    # C_A[t][block_i][block_j] where block_i, block_j in {0,1}
    # Block 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right
    # Actually, let's use 2x2 block indices: (row_blk, col_blk)
    
    C_A = [
        [[1,0],[0,1]],  # M1: A11 + A22
        [[0,0],[1,1]],  # M2: A21 + A22
        [[1,0],[0,0]],  # M3: A11
        [[0,0],[0,1]],  # M4: A22
        [[1,1],[0,0]],  # M5: A11 + A12
        [[1,0],[0,1]],  # M6: A21 - A11... wait
        # M6 = (A21-A11)(B11+B12)
        # So A6 = A21 - A11
        # A21 is block (1,0), A11 is block (0,0)
        [[-1,0],[1,0]],  # M6: -A11 + A21
        [[0,1],[0,-1]]   # M7: A12 - A22
    ]
    
    C_B = [
        [[1,0],[0,1]],  # M1: B11 + B22
        [[1,0],[0,0]],  # M2: B11
        [[0,1],[0,-1]], # M3: B12 - B22
        [[0,-1],[1,0]], # M4: -B11 + B21
        [[0,0],[0,1]],  # M5: B22
        [[1,1],[0,0]],  # M6: B11 + B12
        [[0,1],[1,0]]   # M7: B12 + B21... wait
        # M7 = (A12-A22)(B21+B22)
        # So B7 = B21 + B22
        # B21 is block (1,0), B22 is block (1,1)
    ]
    # Fix C_B[6]:
    C_B[6] = [[0,0],[1,1]]  # B21 + B22
    
    # C targets: for each t, list of (block_i, block_k, coeff)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    Targets = [
        [(0,0,1), (1,1,1)],    # M1 -> C11, C22
        [(1,0,1), (1,1,-1)],   # M2 -> C21, C22
        [(0,1,1), (1,1,1)],    # M3 -> C12, C22
        [(0,0,1), (1,0,1)],    # M4 -> C11, C21
        [(0,0,-1), (0,1,1)],   # M5 -> -C11, C12
        [(1,1,1)],             # M6 -> C22
        [(0,0,1)]              # M7 -> C11
    ]
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    col = 0
    for t in range(7):
        for i_blk in range(2):
            for j_blk in range(2):
                coeff_a = C_A[t][i_blk][j_blk]
                if coeff_a == 0:
                    continue
                # For this A block, apply the 2x2x2x2 Strassen
                for i_loc in range(2):
                    for j_loc in range(2):
                        # Global A index: (i_blk*2 + i_loc, j_blk*2 + j_loc)
                        I = i_blk * 2 + i_loc
                        J = j_blk * 2 + j_loc
                        a_idx = I * 4 + J
                        
                        # For each 2x2x2x2 Strassen term
                        for r in range(7):
                            # U2[r] gives coefficient for A entry (i_loc, j_loc) in the 2x2 block
                            # Actually, U2 is indexed by (i,j) -> i*2+j
                            u_coeff = U2[i_loc*2 + j_loc][r]
                            if u_coeff != 0:
                                U[a_idx][col] = coeff_a * u_coeff
                            
                            # V: for this A block (i_blk, j_blk), the corresponding B block is...
                            # Wait, I need to think about this more carefully.
                            # For a given block-level term t and A block (i_blk, j_blk),
                            # the B block is determined by t.
                            # Actually, no. The block-level Strassen term t defines:
                            # A_t = sum_{i_blk,j_blk} C_A[t][i_blk][j_blk] A_{i_blk,j_blk}
                            # B_t = sum_{j_blk,k_blk} C_B[t][j_blk][k_blk] B_{j_blk,k_blk}
                            # Then A_t * B_t is a 2x2 matrix, and we compute it using 2x2x2x2 Strassen.
                            # So for each t, we have 7 scalar-level Strassen terms.
                            # For each scalar-level term r, we have:
                            # U_r: linear combination of A_t entries
                            # V_r: linear combination of B_t entries
                            # W_r: contribution to C_t entries
                            
                            # A_t entries: for each (i_blk, j_blk), the 2x2 block A_{i_blk,j_blk}
                            # The 2x2x2x2 Strassen U2 gives coefficients for the 4 entries of the 2x2 matrix A_t.
                            # But A_t is a sum of 2x2 blocks, so we need to expand.
                            
                            # For a fixed t and r, the U part is:
                            # sum_{i_blk,j_blk} C_A[t][i_blk][j_blk] * (sum_{i_loc,j_loc} U2[i_loc,j_loc,r] * A_{i_blk,j_blk}[i_loc,j_loc])
                            # = sum_{i_blk,j_blk,i_loc,j_loc} C_A[t][i_blk][j_blk] * U2[i_loc,j_loc,r] * A_{i_blk,j_blk}[i_loc,j_loc]
                            
                            # So for global A index (I, J) = (i_blk*2+i_loc, j_blk*2+j_loc):
                            # U[I*4+J, col] = C_A[t][i_blk][j_blk] * U2[i_loc*2+j_loc, r]
                            
                            # Similarly for V:
                            # For global B index (J, K) = (j_blk*2+j_loc, k_blk*2+k_loc):
                            # V[J*4+K, col] = C_B[t][j_blk][k_blk] * V2[j_loc*2+k_loc, r]
                            
                            # And for W:
                            # For global C index (K, I) = (k_blk*2+k_loc, i_blk*2+i_loc):
                            # W[K*4+I, col] = C_C[t][i_blk][k_blk] * W2[k_loc*2+i_loc, r]
                            
                            # But I'm iterating over i_blk, j_blk, i_loc, j_loc, r, which is not the right order.
                            # Let me restructure.
    
    # I made a mistake in the loop structure. Let me redo this.
    # For each t (0..6) and r (0..6), we have one rank-1 term.
    # For each (t, r), we fill U, V, W.
    
    col = 0
    for t in range(7):
        for r in range(7):
            # Fill U
            for i_blk in range(2):
                for j_blk in range(2):
                    coeff_a = C_A[t][i_blk][j_blk]
                    if coeff_a == 0:
                        continue
                    for i_loc in range(2):
                        for j_loc in range(2):
                            u_coeff = U2[i_loc*2 + j_loc][r]
                            if u_coeff != 0:
                                I = i_blk * 2 + i_loc
                                J = j_blk * 2 + j_loc
                                U[I*4 + J][col] = coeff_a * u_coeff
            
            # Fill V
            for j_blk in range(2):
                for k_blk in range(2):
                    coeff_b = C_B[t][j_blk][k_blk]
                    if coeff_b == 0:
                        continue
                    for j_loc in range(2):
                        for k_loc in range(2):
                            v_coeff = V2[j_loc*2 + k_loc][r]
                            if v_coeff != 0:
                                J = j_blk * 2 + j_loc
                                K = k_blk * 2 + k_loc
                                V[J*4 + K][col] = coeff_b * v_coeff
            
            # Fill W
            for i_t, k_t, coeff_c in Targets[t]:
                for i_loc in range(2):
                    for k_loc in range(2):
                        w_coeff = W2[k_loc*2 + i_loc][r]
                        if w_coeff != 0:
                            I = i_t * 2 + i_loc
                            K = k_t * 2 + k_loc
                            W[K*4 + I][col] = coeff_c * w_coeff
            
            col += 1
    
    return U, V, W

def _schoolbook_4x4_4x2():
    """Return U, V, W for 4x4 x 4x2 schoolbook decomposition (rank 32).
    U: (16, 32), V: (8, 32), W: (8, 32)
    U indexed by (i,j) -> i*4+j
    V indexed by (j,k) -> j*2+k (for 4x2 B)
    W indexed by (k,i) -> k*4+i (for 4x2 C, trace convention)
    """
    rank = 32
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i*4 + j][col] = 1
                V[j*2 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Get the 4x4 x 4x4 Strassen decomposition
    U_s, V_s, W_s = _strassen_4x4_4x4()
    
    # Get the 4x4 x 4x2 schoolbook decomposition
    U_b, V_b, W_b = _schoolbook_4x4_4x2()
    
    rank_s = len(U_s[0])  # 49
    rank_b = len(U_b[0])  # 32
    rank = rank_s + rank_b  # 81
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(24)]  # 4x6 B
    W = [[0]*rank for _ in range(24)]  # 4x6 C (trace convention: k*4+i)
    
    # Copy Strassen part
    for i in range(16):
        for r in range(rank_s):
            U[i][r] = U_s[i][r]
    
    for j in range(4):
        for k in range(4):
            for r in range(rank_s):
                V[j*6 + k][r] = V_s[j*4 + k][r]
    
    for k in range(4):
        for i in range(4):
            for r in range(rank_s):
                W[k*4 + i][r] = W_s[k*4 + i][r]
    
    # Copy schoolbook part (for B columns 4..5, C columns 4..5)
    for i in range(16):
        for r in range(rank_b):
            U[i][rank_s + r] = U_b[i][r]
    
    for j in range(4):
        for k in range(2):
            for r in range(rank_b):
                V[j*6 + (4 + k)][rank_s + r] = V_b[j*2 + k][r]
    
    for k in range(2):
        for i in range(4):
            for r in range(rank_b):
                W[(4 + k)*4 + i][rank_s + r] = W_b[k*4 + i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
