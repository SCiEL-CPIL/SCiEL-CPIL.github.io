import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Base blocks
    
    def block_2_1_2():
        # A(2,1), B(1,2) -> C(2,2)
        # Rank 4
        U = np.zeros((2*1, 4))
        V = np.zeros((1*2, 4))
        W = np.zeros((2*2, 4))
        
        # i in 0..1, j in 0..1
        # C[i,j] = A[i] B[j]
        idx = 0
        for i in range(2):
            for j in range(2):
                U[i, idx] = 1
                V[j, idx] = 1
                W[j*2 + i, idx] = 1 # k=j, i=i -> k*2+i
                idx += 1
        return U, V, W, 4

    def block_2_1_3():
        # A(2,1), B(1,3) -> C(2,3)
        # Rank 6
        U = np.zeros((2*1, 6))
        V = np.zeros((1*3, 6))
        W = np.zeros((3*2, 6))
        
        idx = 0
        for i in range(2):
            for j in range(3):
                U[i, idx] = 1
                V[j, idx] = 1
                W[j*2 + i, idx] = 1
                idx += 1
        return U, V, W, 6

    def block_2_2_2():
        # A(2,2), B(2,2) -> C(2,2)
        # Rank 7 (Strassen)
        U = np.zeros((2*2, 7))
        V = np.zeros((2*2, 7))
        W = np.zeros((2*2, 7))
        
        # A indices: 0,1,2,3 -> 00,01,10,11
        # B indices: 0,1,2,3 -> 00,01,10,11
        # C indices: 0,1,2,3 -> 00,01,10,11 (k*2+i)
        
        # M1 = (A00+A11)(B00+B11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        # C00 += M1, C11 += M1
        W[0,0] = 1 # k=0,i=0
        W[3,0] = 1 # k=1,i=1
        
        # M2 = (A10+A11)B00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        # C10 += M2
        W[2,1] = 1 # k=1,i=0
        
        # M3 = A00(B01-B11)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        # C01 += M3
        W[1,2] = 1 # k=0,i=1
        
        # M4 = A11(B10-B00)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        # C00 += M4, C10 += M4
        W[0,3] = 1
        W[2,3] = 1
        
        # M5 = (A00+A10)B11
        U[0,4] = 1; U[2,4] = 1
        V[3,4] = 1
        # C00 -= M5, C01 += M5
        W[0,4] = -1
        W[1,4] = 1
        
        # M6 = (A01-A00)(B00+B10)
        U[1,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[2,5] = 1
        # C11 += M6
        W[3,5] = 1
        
        # M7 = (A11-A10)(B01+B11)
        U[3,6] = 1; U[2,6] = -1
        V[1,6] = 1; V[3,6] = 1
        # C00 += M7, C11 += M7
        W[0,6] = 1
        W[3,6] = 1
        
        return U, V, W, 7

    def block_2_2_1():
        # A(2,2), B(2,1) -> C(2,1)
        # Rank 4
        U = np.zeros((2*2, 4))
        V = np.zeros((2*1, 4))
        W = np.zeros((1*2, 4))
        
        # C[i,0] = A[i,0]B[0] + A[i,1]B[1]
        idx = 0
        for i in range(2):
            for j in range(2):
                U[i*2+j, idx] = 1
                V[j, idx] = 1
                W[i, idx] = 1 # k=0, i=i
                idx += 1
        return U, V, W, 4

    def block_2_2_3():
        # A(2,2), B(2,3) -> C(2,3)
        # Rank 11 = R(2,2,2) + R(2,2,1)
        U1, V1, W1, r1 = block_2_2_2()
        U2, V2, W2, r2 = block_2_2_1()
        
        U = np.zeros((2*2, r1+r2))
        V = np.zeros((2*3, r1+r2))
        W = np.zeros((3*2, r1+r2))
        
        # Copy block 1 (cols 0,1 of B)
        U[:, 0:r1] = U1
        V[:, 0:r1] = V1
        W[:, 0:r1] = W1
        
        # Copy block 2 (col 2 of B)
        # B indices for block 2: local j 0..1 map to global j 0..1?
        # No, block_2_2_1 expects B(2,1).
        # In block_2_2_3, B is (2,3). Col 2 is the last one.
        # The V matrix for block 2 should have non-zeros in V[0, :] and V[1, :] corresponding to B's col 2.
        # In local indices of block_2_2_1, V is (2,1).
        # We need to place it in V[0:2, r1:r1+r2] of the (2,3) matrix.
        # But V in block_2_2_3 is (2,3) = 6 elements.
        # B indices: 0,1,2,3,4,5 -> (0,0),(0,1),(0,2),(1,0),(1,1),(1,2).
        # Block 2 uses B col 2. So indices 2 and 5.
        # V2 has shape (2,4).
        # We map V2[0, :] to V[2, r1:r1+r2] and V2[1, :] to V[5, r1:r1+r2].
        
        V[2, r1:r1+r2] = V2[0, :]
        V[5, r1:r1+r2] = V2[1, :]
        
        # W for block 2: C(2,1) -> C(2,3) col 2.
        # W2 has shape (2,4) (k=0, i=0..1).
        # We need to place it in W[:, r1:r1+r2] of W(3,2) = 6 elements.
        # C indices: k*2+i.
        # Col 2 of C means k=2.
        # Indices: 2*2+0=4, 2*2+1=5.
        # W2[0] corresponds to i=0, W2[1] to i=1.
        W[4, r1:r1+r2] = W2[0, :]
        W[5, r1:r1+r2] = W2[1, :]
        
        return U, V, W, r1+r2

    def block_2_2_5():
        # A(2,2), B(2,5) -> C(2,5)
        # Rank 18 = R(2,2,2) + R(2,2,3)
        U1, V1, W1, r1 = block_2_2_2()
        U2, V2, W2, r2 = block_2_2_3()
        
        U = np.zeros((2*2, r1+r2))
        V = np.zeros((2*5, r1+r2))
        W = np.zeros((5*2, r1+r2))
        
        # Block 1: B cols 0,1
        U[:, 0:r1] = U1
        V[:, 0:r1] = V1
        W[:, 0:r1] = W1
        
        # Block 2: B cols 2,3,4
        # V2 shape (2,3) -> 6 elements.
        # Map to V[0:2, r1:r1+r2] and V[2:4, ...]? No.
        # V in block_2_2_5 is (2,5) = 10 elements.
        # B indices: (0,0)..(0,4), (1,0)..(1,4).
        # Block 2 uses B cols 2,3,4.
        # V2 has non-zeros in cols 0,1,2 of its local (2,3) matrix.
        # We place V2[0, :] into V[2:5, r1:r1+r2] (row 0 of B, cols 2-4).
        # We place V2[1, :] into V[7:10, r1:r1+r2] (row 1 of B, cols 2-4).
        
        V[2:5, r1:r1+r2] = V2[0, :]
        V[7:10, r1:r1+r2] = V2[1, :]
        
        # W for block 2: C(2,3) -> C(2,5) cols 2-4.
        # W2 shape (3,2) = 6 elements.
        # Place into W[4:10, r1:r1+r2].
        # W indices: k*2+i.
        # k=2: 4,5. k=3: 6,7. k=4: 8,9.
        W[4:10, r1:r1+r2] = W2
        
        return U, V, W, r1+r2

    def block_2_3_3():
        # A(2,3), B(3,3) -> C(2,3)
        # Rank 17 = R(2,1,3) + R(2,2,3)
        U1, V1, W1, r1 = block_2_1_3()
        U2, V2, W2, r2 = block_2_2_3()
        
        U = np.zeros((2*3, r1+r2))
        V = np.zeros((3*3, r1+r2))
        W = np.zeros((3*2, r1+r2))
        
        # Block 1: A col 0, B all.
        # U1 shape (2,6). Place in U[0:2, 0:r1].
        U[0:2, 0:r1] = U1
        # V1 shape (3,6). Place in V[0:9, 0:r1].
        V[0:9, 0:r1] = V1
        # W1 shape (6,6). Place in W[0:6, 0:r1].
        W[0:6, 0:r1] = W1
        
        # Block 2: A cols 1,2, B all.
        # U2 shape (4,11). Place in U[2:6, r1:r1+r2].
        U[2:6, r1:r1+r2] = U2
        # V2 shape (6,11). Place in V[0:9, r1:r1+r2].
        V[0:9, r1:r1+r2] = V2
        # W2 shape (6,11). Place in W[0:6, r1:r1+r2].
        W[0:6, r1:r1+r2] = W2
        
        return U, V, W, r1+r2

    def block_2_3_2():
        # A(2,3), B(3,2) -> C(2,2)
        # Rank 11 = R(2,1,2) + R(2,2,2)
        U1, V1, W1, r1 = block_2_1_2()
        U2, V2, W2, r2 = block_2_2_2()
        
        U = np.zeros((2*3, r1+r2))
        V = np.zeros((3*2, r1+r2))
        W = np.zeros((2*2, r1+r2))
        
        # Block 1: A col 0
        U[0:2, 0:r1] = U1
        V[0:6, 0:r1] = V1
        W[0:4, 0:r1] = W1
        
        # Block 2: A cols 1,2
        U[2:6, r1:r1+r2] = U2
        V[0:6, r1:r1+r2] = V2
        W[0:4, r1:r1+r2] = W2
        
        return U, V, W, r1+r2

    def block_2_3_5():
        # A(2,3), B(3,5) -> C(2,5)
        # Rank 28 = R(2,3,3) + R(2,3,2)
        U1, V1, W1, r1 = block_2_3_3()
        U2, V2, W2, r2 = block_2_3_2()
        
        U = np.zeros((2*3, r1+r2))
        V = np.zeros((3*5, r1+r2))
        W = np.zeros((5*2, r1+r2))
        
        # Block 1: B cols 0,1,2
        U[:, 0:r1] = U1
        V[:, 0:r1] = V1
        W[:, 0:r1] = W1
        
        # Block 2: B cols 3,4
        # V2 shape (6,11).
        # Place in V[15:21, r1:r1+r2] (cols 3,4 of B).
        V[15:21, r1:r1+r2] = V2
        # W2 shape (4,11).
        # Place in W[12:20, r1:r1+r2] (cols 3,4 of C).
        W[12:20, r1:r1+r2] = W2
        
        return U, V, W, r1+r2

    def block_4_2_5():
        # A(4,2), B(2,5) -> C(4,5)
        # Rank 36 = 2 * R(2,2,5)
        U1, V1, W1, r1 = block_2_2_5()
        U2, V2, W2, r2 = block_2_2_5()
        
        U = np.zeros((4*2, r1+r2))
        V = np.zeros((2*5, r1+r2))
        W = np.zeros((5*4, r1+r2))
        
        # Block 1: A rows 0,1
        U[0:4, 0:r1] = U1
        V[:, 0:r1] = V1
        # W1 shape (10,18). Place in W[0:10, 0:r1].
        W[0:10, 0:r1] = W1
        
        # Block 2: A rows 2,3
        U[4:8, r1:r1+r2] = U2
        V[:, r1:r1+r2] = V2
        # W2 shape (10,18). Place in W[10:30, r1:r1+r2].
        # W indices: k*4+i.
        # Rows 2,3 of C.
        # i=2,3.
        # k=0: 2,3. k=1: 6,7. ...
        # Actually W1 covers i=0,1. W2 covers i=2,3.
        # So we just shift W2 by 20? No.
        # W1 is for i=0,1. Indices 0..9?
        # W shape (5*4, 18) = (20, 18).
        # W1 covers C rows 0,1.
        # W2 covers C rows 2,3.
        # We need to map W2's local i (0,1) to global i (2,3).
        # Local W index k*2+i. Global W index k*4+i.
        # This mapping is not a simple slice if n changes from 2 to 4.
        # We must build W2 with n=4?
        # No, block_2_2_5 assumes n=2.
        # We need a version of block_2_2_5 that outputs W for n=4?
        # Or we can just construct the final U,V,W directly.
        
        # Let's refactor to avoid W mapping issues.
        # We will construct the final matrices directly.
        
        pass

    # Direct construction for block_4_5_5 (Rank 92)
    
    U = np.zeros((4*5, 92))
    V = np.zeros((5*5, 92))
    W = np.zeros((5*4, 92))
    
    # Part 1: A[:, 0:2] * B[0:2, :] (Rank 36)
    # Sub-parts: A[0:2, 0:2] * B[0:2, :] and A[2:4, 0:2] * B[0:2, :]
    
    # Helper to add a block
    def add_block(U_loc, V_loc, W_loc, offset, i_start, j_start, k_start, n_loc, m_loc, p_loc, n_glob, m_glob, p_glob):
        # This is complex. Let's just do it manually for the specific cases.
        pass

    # We will generate the blocks and merge them.
    
    # Block A: 4x2x5 (Rank 36)
    # Block B: 4x3x5 (Rank 56)
    
    # We need to generate 4_2_5 and 4_3_5 correctly.
    
    # Re-implementing 4_2_5 with correct W mapping
    def get_4_2_5():
        U1, V1, W1, r1 = block_2_2_5()
        U2, V2, W2, r2 = block_2_2_5()
        
        U = np.zeros((4*2, r1+r2))
        V = np.zeros((2*5, r1+r2))
        W = np.zeros((5*4, r1+r2))
        
        # Part 1: A rows 0,1
        U[0:4, 0:r1] = U1
        V[:, 0:r1] = V1
        # W1 is for n=2. We need to map to n=4.
        # W1 index k*2+i.
        # Global W index k*4+i.
        for k in range(5):
            for i in range(2):
                W[k*4+i, 0:r1] = W1[k*2+i, 0:r1]
                
        # Part 2: A rows 2,3
        U[4:8, r1:r1+r2] = U2
        V[:, r1:r1+r2] = V2
        # W2 is for n=2.
        for k in range(5):
            for i in range(2):
                W[k*4+(2+i), r1:r1+r2] = W2[k*2+i, r1:r1+r2]
                
        return U, V, W, r1+r2

    def get_4_3_5():
        U1, V1, W1, r1 = block_2_3_5()
        U2, V2, W2, r2 = block_2_3_5()
        
        U = np.zeros((4*3, r1+r2))
        V = np.zeros((3*5, r1+r2))
        W = np.zeros((5*4, r1+r2))
        
        # Part 1: A rows 0,1
        U[0:6, 0:r1] = U1
        V[:, 0:r1] = V1
        for k in range(5):
            for i in range(2):
                W[k*4+i, 0:r1] = W1[k*2+i, 0:r1]
                
        # Part 2: A rows 2,3
        U[6:12, r1:r1+r2] = U2
        V[:, r1:r1+r2] = V2
        for k in range(5):
            for i in range(2):
                W[k*4+(2+i), r1:r1+r2] = W2[k*2+i, r1:r1+r2]
                
        return U, V, W, r1+r2

    U1, V1, W1, r1 = get_4_2_5()
    U2, V2, W2, r2 = get_4_3_5()
    
    # Combine
    U_final = np.zeros((4*5, r1+r2))
    V_final = np.zeros((5*5, r1+r2))
    W_final = np.zeros((5*4, r1+r2))
    
    # Part 1: A cols 0,1 (j=0,1)
    U_final[0:10, 0:r1] = U1
    V_final[0:10, 0:r1] = V1 # B rows 0,1 (j=0,1)
    W_final[:, 0:r1] = W1
    
    # Part 2: A cols 2,3,4 (j=2,3,4)
    U_final[10:20, r1:r1+r2] = U2
    V_final[10:25, r1:r1+r2] = V2 # B rows 2,3,4 (j=2,3,4)
    W_final[:, r1:r1+r2] = W2
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
