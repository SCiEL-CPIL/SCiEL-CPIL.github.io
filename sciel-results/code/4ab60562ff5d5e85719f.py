import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4 * 4x4 (columns 0-3 of B)
    # Rank 49
    U1, V1, W1 = get_strassen_4x4()
    
    # Part 2: 4x4 * 4x4 (columns 4-7 of B)
    # Rank 49
    U2, V2, W2 = get_strassen_4x4()
    
    # Part 3: 4x4 * 4x1 (column 8 of B)
    # Rank 16
    U3, V3, W3 = get_schoolbook_4x4x1()
    
    # Concatenate
    # U: (16, rank)
    U = np.hstack([U1, U2, U3])
    
    # V: (36, rank)
    # V1 is for rows 0-15 (cols 0-3)
    # V2 is for rows 16-31 (cols 4-7)
    # V3 is for rows 32-35 (col 8)
    V = np.zeros((36, U.shape[1]))
    V[0:16, :] = V1
    V[16:32, :] = V2
    V[32:36, :] = V3
    
    # W: (36, rank)
    # W1 is for rows 0-15 (cols 0-3)
    # W2 is for rows 16-31 (cols 4-7)
    # W3 is for rows 32-35 (col 8)
    W = np.zeros((36, U.shape[1]))
    W[0:16, :] = W1
    W[16:32, :] = W2
    W[32:36, :] = W3
    
    # Convert to lists of lists of floats/ints
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def get_strassen_2x2():
    # Returns U, V, W for 2x2x2 matrix multiplication
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11) -> C00+=, C11+=
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10+=, C11-=
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1 # C10 is W[0*2+1]=W[2]? No.
    # W index is k*n + i.
    # C10: k=0, i=1 -> W[0*2+1] = W[1]
    # C11: k=1, i=1 -> W[1*2+1] = W[3]
    # So W[1]=1, W[3]=-1. Correct.
    
    # M3: A00(B01-B11) -> C01+=, C11+=
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C01: k=1, i=0 -> W[1*2+0]=W[2]. C11: k=1, i=1 -> W[3].
    # Wait. C01 is row 0, col 1. i=0, k=1. W[1*2+0] = W[2].
    # C11 is row 1, col 1. i=1, k=1. W[1*2+1] = W[3].
    # So W[2]=1, W[3]=1.
    # My previous line: W[1,2]=1 is wrong index.
    # Let's re-verify W indexing.
    # W[k*n + i]. n=2.
    # C_ik.
    # C_00: k=0, i=0 -> W[0]
    # C_01: k=1, i=0 -> W[2]
    # C_10: k=0, i=1 -> W[1]
    # C_11: k=1, i=1 -> W[3]
    
    # M1: C00, C11. W[0]=1, W[3]=1.
    # M2: C10, C11. W[1]=1, W[3]=-1.
    # M3: C01, C11. W[2]=1, W[3]=1.
    # M4: C00, C10. W[0]=1, W[1]=1.
    # M5: C00, C01. W[0]=-1, W[2]=1.
    # M6: C11. W[3]=1.
    # M7: C00. W[0]=1.
    
    # Correcting M3
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00+=, C10+=
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A00+A01)B11 -> C00-=, C01+=
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11+=
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00+=
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def get_strassen_4x4():
    # Returns U, V, W for 4x4x4 matrix multiplication
    # Rank 49
    U2, V2, W2 = get_strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Strassen blocks for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mk is a 2x2 matrix product.
    # We decompose each Mk using U2, V2, W2.
    
    # Block indices for 4x4:
    # A00: rows 0-1, cols 0-1. U indices 0-3.
    # A01: rows 0-1, cols 2-3. U indices 4-7.
    # A10: rows 2-3, cols 0-1. U indices 8-11.
    # A11: rows 2-3, cols 2-3. U indices 12-15.
    
    # B00: rows 0-1, cols 0-1. V indices 0-3.
    # B01: rows 0-1, cols 2-3. V indices 4-7.
    # B10: rows 2-3, cols 0-1. V indices 8-11.
    # B11: rows 2-3, cols 2-3. V indices 12-15.
    
    # C00: rows 0-1, cols 0-1. W indices 0-3.
    # C01: rows 0-1, cols 2-3. W indices 4-7.
    # C10: rows 2-3, cols 0-1. W indices 8-11.
    # C11: rows 2-3, cols 2-3. W indices 12-15.
    
    # Helper to add a block product
    def add_block_product(m_idx, u_blocks, v_blocks, w_blocks):
        # u_blocks: list of (block_idx, coeff)
        # v_blocks: list of (block_idx, coeff)
        # w_blocks: list of (block_idx, coeff)
        base_c = m_idx * 7
        for c in range(7):
            curr_c = base_c + c
            # U
            for (ub, coeff_u) in u_blocks:
                # ub is 0..3
                # U2[c] is 4x1
                for i in range(4):
                    U[ub*4 + i, curr_c] += U2[i, c] * coeff_u
            # V
            for (vb, coeff_v) in v_blocks:
                for i in range(4):
                    V[vb*4 + i, curr_c] += V2[i, c] * coeff_v
            # W
            for (wb, coeff_w) in w_blocks:
                for i in range(4):
                    W[wb*4 + i, curr_c] += W2[i, c] * coeff_w

    # M1: (A00+A11)(B00+B11) -> C00+=, C11+=
    add_block_product(0, 
                      [(0, 1), (3, 1)], 
                      [(0, 1), (3, 1)], 
                      [(0, 1), (3, 1)])
                      
    # M2: (A10+A11)B00 -> C10+=, C11-=
    add_block_product(1, 
                      [(2, 1), (3, 1)], 
                      [(0, 1)], 
                      [(2, 1), (3, -1)])
                      
    # M3: A00(B01-B11) -> C01+=, C11+=
    add_block_product(2, 
                      [(0, 1)], 
                      [(1, 1), (3, -1)], 
                      [(1, 1), (3, 1)])
                      
    # M4: A11(B10-B00) -> C00+=, C10+=
    add_block_product(3, 
                      [(3, 1)], 
                      [(2, 1), (0, -1)], 
                      [(0, 1), (2, 1)])
                      
    # M5: (A00+A01)B11 -> C00-=, C01+=
    add_block_product(4, 
                      [(0, 1), (1, 1)], 
                      [(3, 1)], 
                      [(0, -1), (1, 1)])
                      
    # M6: (A10-A00)(B00+B01) -> C11+=
    add_block_product(5, 
                      [(2, 1), (0, -1)], 
                      [(0, 1), (1, 1)], 
                      [(3, 1)])
                      
    # M7: (A01-A11)(B10+B11) -> C00+=
    add_block_product(6, 
                      [(1, 1), (3, -1)], 
                      [(2, 1), (3, 1)], 
                      [(0, 1)])
                      
    return U, V, W

def get_schoolbook_4x4x1():
    # Returns U, V, W for 4x4 * 4x1
    # U: (16, 16), V: (4, 16), W: (4, 16)
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))
    
    c = 0
    for i in range(4):
        for j in range(4):
            U[i*4 + j, c] = 1
            V[j, c] = 1
            W[i, c] = 1
            c += 1
            
    return U, V, W

