import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total rank 78
    total_rank = 78
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    offset = 0
    
    # Helper: Add a block decomposition
    def add_block(U_block, V_block, W_block, u_offset, v_offset, w_offset):
        nonlocal offset
        r = U_block.shape[1]
        U[u_offset:u_offset+U_block.shape[0], offset:offset+r] = U_block
        V[v_offset:v_offset+V_block.shape[0], offset:offset+r] = V_block
        W[w_offset:w_offset+W_block.shape[0], offset:offset+r] = W_block
        offset += r

    # 1. Part 1: Row 2 of A (1x4) * B (4x7)
    # U part: A[2, :] -> indices 2*4 + j = 8+j. Shape (4, 28)
    # V part: B[:, :] -> indices j*7 + k. Shape (28, 28)
    # W part: C[:, 2] -> indices k*3 + 2. Shape (7, 28)
    
    # Schoolbook 1x4x7
    U_1 = np.zeros((4, 28))
    V_1 = np.zeros((28, 28))
    W_1 = np.zeros((7, 28))
    for j in range(4):
        for k in range(7):
            r = j*7 + k
            U_1[j, r] = 1
            V_1[j*7+k, r] = 1
            W_1[k, r] = 1
            
    # Map to global
    # U rows: 8..11
    # V rows: 0..27
    # W rows: 2, 5, 8, 11, 14, 17, 20 (k*3+2)
    
    u_start = 8
    v_start = 0
    # W is sparse, need to map k to k*3+2
    W_1_global = np.zeros((21, 28))
    for k in range(7):
        W_1_global[k*3+2, :] = W_1[k, :]
        
    add_block(U_1, V_1, W_1_global, u_start, v_start, 0)
    # Note: W offset in add_block is relative to start of W matrix.
    # But W_1_global is already full size (21, 28).
    # So we should just assign directly or use a helper that handles full size.
    # Let's refactor add_block to take full size blocks and offsets.
    
    # Reset U, V, W
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    offset = 0
    
    def add_block_full(U_b, V_b, W_b, u_off, v_off, w_off):
        nonlocal offset
        r = U_b.shape[1]
        U[u_off:u_off+U_b.shape[0], offset:offset+r] = U_b
        V[v_off:v_off+V_b.shape[0], offset:offset+r] = V_b
        W[w_off:w_off+W_b.shape[0], offset:offset+r] = W_b
        offset += r

    # Part 1
    U_1 = np.zeros((4, 28))
    V_1 = np.zeros((28, 28))
    W_1 = np.zeros((21, 28))
    for j in range(4):
        for k in range(7):
            r = j*7 + k
            U_1[j, r] = 1
            V_1[j*7+k, r] = 1
            W_1[k*3+2, r] = 1
            
    add_block_full(U_1, V_1, W_1, 8, 0, 0)
    
    # Part 2a: A[0:2, 0:2] * B[0:2, :]
    # A rows 0,1. Cols 0,1.
    # B rows 0,1. Cols 0..6.
    # C rows 0,1. Cols 0..6.
    
    # Sub 2a1: B[0:2, 0:2] (2x2x2 Strassen)
    # U: A[0:2, 0:2] -> rows 0,1, 4,5.
    # V: B[0:2, 0:2] -> rows 0,1, 7,8.
    # W: C[0:2, 0:2] -> rows 0,1, 3,4.
    
    # We need a Strassen 2x2x2 generator
    def strassen_222():
        # Returns U (4,7), V (4,7), W (4,7)
        # Standard Strassen
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices: A00=0, A01=1, A10=2, A11=3
        # B00=0, B01=1, B10=2, B11=3
        # C00=0, C01=1, C10=2, C11=3
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[0, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[1, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[3, 6] = 1; W[2, 6] = 1
        
        return U, V, W

    def schoolbook_221():
        # 2x2x1
        # C00 = A00B00 + A01B10
        # C10 = A10B00 + A11B10
        # Rank 4
        U = np.zeros((4, 4))
        V = np.zeros((4, 4))
        W = np.zeros((4, 4))
        
        # Term 1: A00 B00
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A01 B10
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        # Term 3: A10 B00
        U[2, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
        # Term 4: A11 B10
        U[3, 3] = 1; V[2, 3] = 1; W[2, 3] = 1
        
        return U, V, W

    def rank11_223():
        # 2x2x3
        # Split B into B1 (2x2) and B2 (2x1)
        # C = A B1 + A B2
        # U, V, W for B1 part (Strassen)
        U1, V1, W1 = strassen_222()
        # U, V, W for B2 part (Schoolbook 2x2x1)
        U2, V2, W2 = schoolbook_221()
        
        # Combine
        U = np.hstack([U1, U2])
        # V: B1 is cols 0,1. B2 is col 2.
        # V1 is (4,7). V2 is (4,4).
        # V2 corresponds to B[0,2], B[1,2], B[2,2]? No.
        # In 2x2x1, B is 2x1.
        # In 2x2x3, B is 2x3.
        # V indices for B:
        # B1: (0,0), (0,1), (1,0), (1,1) -> Global 0,1,2,3
        # B2: (0,2), (1,2) -> Global 4,5
        # Wait, V shape is (m*p, R).
        # For 2x2x3, m=2, p=3. V is (6, R).
        # B1 uses B[0:2, 0:2]. Indices 0,1,2,3.
        # B2 uses B[0:2, 2:3]. Indices 4,5.
        
        # V1 (Strassen) expects B indices 0,1,2,3.
        # V2 (Schoolbook) expects B indices 0,1 for its local 2x1.
        # We need to map V2's local indices 0,1 to global 4,5.
        
        V = np.zeros((6, 11))
        V[0:4, 0:7] = V1
        # V2 is (4,4).
        # V2[0, :] corresponds to B[0, 0] local. Global B[0, 2] -> index 4.
        # V2[1, :] corresponds to B[0, 1] local? No, Schoolbook 2x2x1:
        # B is 2x1. Rows 0,1.
        # V2[0, :] is B[0,0]. V2[2, :] is B[1,0].
        # So V2[0, :] -> Global 4. V2[2, :] -> Global 5.
        V[4, 7:11] = V2[0, :]
        V[5, 7:11] = V2[2, :]
        
        # W is (4, 11)
        W = np.hstack([W1, W2])
        
        return U, V, W

    # Now assemble Part 2a
    # 2a1: Strassen 2x2x2
    U_s, V_s, W_s = strassen_222()
    # Map to global
    # U: A[0:2, 0:2] -> rows 0,1, 4,5
    U_2a1 = np.zeros((12, 7))
    U_2a1[0:4, :] = U_s # Rows 0,1,2,3? No.
    # Strassen U is (4,7). Indices 0,1,2,3 correspond to A00,A01,A10,A11.
    # Global A[0:2, 0:2]:
    # A00 -> 0
    # A01 -> 1
    # A10 -> 4
    # A11 -> 5
    # So U_s[0] -> U_2a1[0]
    # U_s[1] -> U_2a1[1]
    # U_s[2] -> U_2a1[4]
    # U_s[3] -> U_2a1[5]
    
    U_2a1[0, :] = U_s[0, :]
    U_2a1[1, :] = U_s[1, :]
    U_2a1[4, :] = U_s[2, :]
    U_2a1[5, :] = U_s[3, :]
    
    # V: B[0:2, 0:2] -> rows 0,1, 7,8
    V_2a1 = np.zeros((28, 7))
    # V_s indices 0,1,2,3 -> B00,B01,B10,B11
    # Global B[0:2, 0:2]:
    # B00 -> 0
    # B01 -> 1
    # B10 -> 7
    # B11 -> 8
    V_2a1[0, :] = V_s[0, :]
    V_2a1[1, :] = V_s[1, :]
    V_2a1[7, :] = V_s[2, :]
    V_2a1[8, :] = V_s[3, :]
    
    # W: C[0:2, 0:2] -> rows 0,1, 3,4
    W_2a1 = np.zeros((21, 7))
    # W_s indices 0,1,2,3 -> C00,C01,C10,C11
    # Global C[0:2, 0:2]:
    # C00 -> 0
    # C01 -> 1
    # C10 -> 3
    # C11 -> 4
    W_2a1[0, :] = W_s[0, :]
    W_2a1[1, :] = W_s[1, :]
    W_2a1[3, :] = W_s[2, :]
    W_2a1[4, :] = W_s[3, :]
    
    add_block_full(U_2a1, V_2a1, W_2a1, 0, 0, 0)
    
    # 2a2: Strassen 2x2x2 (B cols 2,3)
    U_2a2 = np.zeros((12, 7))
    U_2a2[0, :] = U_s[0, :]
    U_2a2[1, :] = U_s[1, :]
    U_2a2[4, :] = U_s[2, :]
    U_2a2[5, :] = U_s[3, :]
    
    V_2a2 = np.zeros((28, 7))
    # B[0:2, 2:4]
    # B[0,2] -> 2
    # B[0,3] -> 3
    # B[1,2] -> 9
    # B[1,3] -> 10
    V_2a2[2, :] = V_s[0, :]
    V_2a2[3, :] = V_s[1, :]
    V_2a2[9, :] = V_s[2, :]
    V_2a2[10, :] = V_s[3, :]
    
    W_2a2 = np.zeros((21, 7))
    # C[0:2, 2:4]
    # C[0,2] -> 2
    # C[0,3] -> 3
    # C[1,2] -> 5
    # C[1,3] -> 6
    W_2a2[2, :] = W_s[0, :]
    W_2a2[3, :] = W_s[1, :]
    W_2a2[5, :] = W_s[2, :]
    W_2a2[6, :] = W_s[3, :]
    
    add_block_full(U_2a2, V_2a2, W_2a2, 0, 0, 0)
    
    # 2a3: Rank 11 2x2x3 (B cols 4,5,6)
    U_r, V_r, W_r = rank11_223()
    U_2a3 = np.zeros((12, 11))
    U_2a3[0, :] = U_r[0, :]
    U_2a3[1, :] = U_r[1, :]
    U_2a3[4, :] = U_r[2, :]
    U_2a3[5, :] = U_r[3, :]
    
    V_2a3 = np.zeros((28, 11))
    # B[0:2, 4:7]
    # B[0,4] -> 4
    # B[0,5] -> 5
    # B[0,6] -> 6
    # B[1,4] -> 11
    # B[1,5] -> 12
    # B[1,6] -> 13
    V_2a3[4, :] = V_r[0, :]
    V_2a3[5, :] = V_r[1, :]
    V_2a3[6, :] = V_r[2, :]
    V_2a3[11, :] = V_r[3, :]
    V_2a3[12, :] = V_r[4, :]
    V_2a3[13, :] = V_r[5, :]
    
    W_2a3 = np.zeros((21, 11))
    # C[0:2, 4:7]
    # C[0,4] -> 4
    # C[0,5] -> 5
    # C[0,6] -> 6
    # C[1,4] -> 7
    # C[1,5] -> 8
    # C[1,6] -> 9
    W_2a3[4, :] = W_r[0, :]
    W_2a3[5, :] = W_r[1, :]
    W_2a3[6, :] = W_r[2, :]
    W_2a3[7, :] = W_r[3, :]
    W_2a3[8, :] = W_r[4, :]
    W_2a3[9, :] = W_r[5, :]
    
    add_block_full(U_2a3, V_2a3, W_2a3, 0, 0, 0)
    
    # Part 2b: A[0:2, 2:4] * B[2:4, :]
    # Same as 2a, but A cols 2,3 and B rows 2,3.
    
    # 2b1: Strassen (B[2:4, 0:2])
    U_2b1 = np.zeros((12, 7))
    # A[0:2, 2:4]
    # A02 -> 2
    # A03 -> 3
    # A12 -> 6
    # A13 -> 7
    U_2b1[2, :] = U_s[0, :]
    U_2b1[3, :] = U_s[1, :]
    U_2b1[6, :] = U_s[2, :]
    U_2b1[7, :] = U_s[3, :]
    
    V_2b1 = np.zeros((28, 7))
    # B[2:4, 0:2]
    # B[2,0] -> 14
    # B[2,1] -> 15
    # B[3,0] -> 21
    # B[3,1] -> 22
    V_2b1[14, :] = V_s[0, :]
    V_2b1[15, :] = V_s[1, :]
    V_2b1[21, :] = V_s[2, :]
    V_2b1[22, :] = V_s[3, :]
    
    W_2b1 = np.zeros((21, 7))
    # C[0:2, 0:2]
    W_2b1[0, :] = W_s[0, :]
    W_2b1[1, :] = W_s[1, :]
    W_2b1[3, :] = W_s[2, :]
    W_2b1[4, :] = W_s[3, :]
    
    add_block_full(U_2b1, V_2b1, W_2b1, 0, 0, 0)
    
    # 2b2: Strassen (B[2:4, 2:4])
    U_2b2 = np.zeros((12, 7))
    U_2b2[2, :] = U_s[0, :]
    U_2b2[3, :] = U_s[1, :]
    U_2b2[6, :] = U_s[2, :]
    U_2b2[7, :] = U_s[3, :]
    
    V_2b2 = np.zeros((28, 7))
    # B[2:4, 2:4]
    # B[2,2] -> 16
    # B[2,3] -> 17
    # B[3,2] -> 23
    # B[3,3] -> 24
    V_2b2[16, :] = V_s[0, :]
    V_2b2[17, :] = V_s[1, :]
    V_2b2[23, :] = V_s[2, :]
    V_2b2[24, :] = V_s[3, :]
    
    W_2b2 = np.zeros((21, 7))
    # C[0:2, 2:4]
    W_2b2[2, :] = W_s[0, :]
    W_2b2[3, :] = W_s[1, :]
    W_2b2[5, :] = W_s[2, :]
    W_2b2[6, :] = W_s[3, :]
    
    add_block_full(U_2b2, V_2b2, W_2b2, 0, 0, 0)
    
    # 2b3: Rank 11 (B[2:4, 4:7])
    U_2b3 = np.zeros((12, 11))
    U_2b3[2, :] = U_r[0, :]
    U_2b3[3, :] = U_r[1, :]
    U_2b3[6, :] = U_r[2, :]
    U_2b3[7, :] = U_r[3, :]
    
    V_2b3 = np.zeros((28, 11))
    # B[2:4, 4:7]
    # B[2,4] -> 18
    # B[2,5] -> 19
    # B[2,6] -> 20
    # B[3,4] -> 25
    # B[3,5] -> 26
    # B[3,6] -> 27
    V_2b3[18, :] = V_r[0, :]
    V_2b3[19, :] = V_r[1, :]
    V_2b3[20, :] = V_r[2, :]
    V_2b3[25, :] = V_r[3, :]
    V_2b3[26, :] = V_r[4, :]
    V_2b3[27, :] = V_r[5, :]
    
    W_2b3 = np.zeros((21, 11))
    # C[0:2, 4:7]
    W_2b3[4, :] = W_r[0, :]
    W_2b3[5, :] = W_r[1, :]
    W_2b3[6, :] = W_r[2, :]
    W_2b3[7, :] = W_r[3, :]
    W_2b3[8, :] = W_r[4, :]
    W_2b3[9, :] = W_r[5, :]
    
    add_block_full(U_2b3, V_2b3, W_2b3, 0, 0, 0)
    
    # Check offset
    # 28 + 7 + 7 + 11 + 7 + 7 + 11 = 78. Correct.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
