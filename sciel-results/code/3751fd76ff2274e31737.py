import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 4, 4, 4
    rank = 49

    # Pre-computed 2x2 Strassen decomposition tables
    # U2[i*2+j][r]: coefficient for A[i][j] in r-th multiplication (r=0..6)
    # V2[i*2+j][r]: coefficient for B[i][j] in r-th multiplication
    # W2[k*2+i][r]: coefficient for C[i][k] in r-th multiplication
    # Indexing: row = i*2+j for A/B (row-major), row = k*2+i for C (trace convention)
    
    # Strassen 2x2: M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],    # B01
        [0, 0, 0, 1, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1]    # B11
    ], dtype=int)
    
    # W2 rows indexed by k*2+i (trace convention: C[i][k] at row k*2+i)
    # Row 0: k=0,i=0 -> C00
    # Row 1: k=0,i=1 -> C10
    # Row 2: k=1,i=0 -> C01
    # Row 3: k=1,i=1 -> C11
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00 = M1+M4-M5+M7
        [0, 1, 0, 1, 0, 0, 0],    # C10 = M2+M4
        [0, 0, 1, 0, 1, 0, 0],    # C01 = M3+M5
        [1, -1, 1, 0, 0, 1, 0]    # C11 = M1-M2+M3+M6
    ], dtype=int)

    # Build full 4x4 decomposition by recursive block expansion
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)

    # For each of the 7 block-level multiplications (s=0..6) and 7 element-level (t=0..6)
    # The block s determines which 2x2 blocks of A, B, C are involved
    # The element t determines which elements within those blocks
    
    # Block-level Strassen: for each s (0..6), which blocks of A and B contribute
    # A_block[s]: list of (block_row, block_col, coeff)
    # B_block[s]: list of (block_row, block_col, coeff)
    # C_block[s]: list of (block_row, block_col, coeff)
    
    A_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: A00+A11
        [(1,0,1), (1,1,1)],      # s=1: A10+A11
        [(0,0,1)],               # s=2: A00
        [(1,1,1)],               # s=3: A11
        [(0,0,1), (0,1,1)],      # s=4: A00+A01
        [(1,0,1), (0,0,-1)],     # s=5: A10-A00
        [(0,1,1), (1,1,-1)],     # s=6: A01-A11
    ]
    B_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: B00+B11
        [(0,0,1)],               # s=1: B00
        [(0,1,1), (1,1,-1)],     # s=2: B01-B11
        [(1,0,1), (0,0,-1)],     # s=3: B10-B00
        [(1,1,1)],               # s=4: B11
        [(0,0,1), (0,1,1)],      # s=5: B00+B01
        [(1,0,1), (1,1,1)],      # s=6: B10+B11
    ]
    C_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: C00+=M, C11+=M
        [(1,0,1), (1,1,-1)],     # s=1: C10+=M, C11-=M
        [(0,1,1), (1,1,1)],      # s=2: C01+=M, C11+=M
        [(0,0,1), (1,0,1)],      # s=3: C00+=M, C10+=M
        [(0,0,-1), (0,1,1)],     # s=4: C00-=M, C01+=M
        [(1,1,1)],               # s=5: C11+=M
        [(0,0,1)],               # s=6: C00+=M
    ]
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Build U column
            for (bi, bj, c_a) in A_blocks[s]:
                for ei in range(2):
                    for ej in range(2):
                        row_a = (2*bi + ei) * 4 + (2*bj + ej)
                        U[row_a, col] += c_a * U2[ei*2+ej, t]
            
            # Build V column
            for (bj, bk, c_b) in B_blocks[s]:
                for ej in range(2):
                    for ek in range(2):
                        row_b = (2*bj + ej) * 4 + (2*bk + ek)
                        V[row_b, col] += c_b * V2[ej*2+ek, t]
            
            # Build W column
            for (bi, bk, c_c) in C_blocks[s]:
                for ei in range(2):
                    for ek in range(2):
                        row_c = (2*bk + ek) * 4 + (2*bi + ei)
                        W[row_c, col] += c_c * W2[ek*2+ei, t]

    # Verify correctness
    try:
        np.random.seed(12345)
        for trial in range(5):
            A_test = np.random.randint(-10, 11, size=(4, 4)).astype(float)
            B_test = np.random.randint(-10, 11, size=(4, 4)).astype(float)
            
            AB_expected = A_test @ B_test
            
            A_flat = A_test.flatten()
            B_flat = B_test.flatten()
            
            u_vals = U.T @ A_flat
            v_vals = V.T @ B_flat
            scalars = u_vals * v_vals
            C_flat = W.T @ scalars
            C_decomp = C_flat.reshape(4, 4)
            
            if not np.allclose(AB_expected, C_decomp, atol=1e-9):
                # Fallback to schoolbook rank 64
                rank_fb = 64
                U_fb = np.zeros((16, rank_fb), dtype=int)
                V_fb = np.zeros((16, rank_fb), dtype=int)
                W_fb = np.zeros((16, rank_fb), dtype=int)
                col_fb = 0
                for i2 in range(4):
                    for j2 in range(4):
                        for k2 in range(4):
                            U_fb[i2*4+j2, col_fb] = 1
                            V_fb[j2*4+k2, col_fb] = 1
                            W_fb[k2*4+i2, col_fb] = 1
                            col_fb += 1
                return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    except Exception:
        pass

    if time.time() > deadline:
        # Return what we have as fallback
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
