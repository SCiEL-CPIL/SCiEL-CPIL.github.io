import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # We decompose the 2x4 * 4x4 product into 4 blocks of 2x2 * 2x2.
    # A is 2x4, B is 4x4.
    # Split A into A_top (2x2, cols 0-1) and A_bot (2x2, cols 2-3)? 
    # No, A is 2 rows, 4 cols. 
    # Let's split along the columns of A and rows of B.
    # A = [A1 A2] where A1 is 2x2 (cols 0-1), A2 is 2x2 (cols 2-3)
    # B = [[B11] [B21]] ? No, B is 4x4.
    # B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = A * B = [A1 A2] * [[B11, B12], [B21, B22]]
    # C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # So C is composed of two 2x2 blocks: C1 = A1*B11 + A2*B21 and C2 = A1*B12 + A2*B22.
    
    # We need to compute A1*B11, A2*B21, A1*B12, A2*B22.
    # Each 2x2*2x2 product can be done with rank 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    
    # Strassen decomposition for 2x2 * 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 - M5 + M3
    # C12 = M4 + M5
    # C21 = M2 + M7
    # C22 = M1 - M2 + M4 + M6
    
    # We define a helper to add a Strassen block to the global tensors.
    # Args: 
    #   a_r0, a_c0: Top-left of A block (2x2)
    #   b_r0, b_c0: Top-left of B block (2x2)
    #   c_r0, c_c0: Top-left of C block (2x2) -- Note: C is p*n = 4*2 = 8 indices.
    #              Indexing for C (result of 2x4 * 4x4):
    #              C is 2x4. 
    #              Row i of C comes from row i of A.
    #              Col j of C comes from col j of B.
    #              The task says output factor follows W[k*n+i, r]. 
    #              This means W is flattened C transposed? Or C flattened column-major?
    #              "W[k*n+i, r]" where k is col of C, i is row of C.
    #              So W index = col * n + row.
    #              n=2.
    #              C is 2x4. Rows 0..1, Cols 0..3.
    #              C[0,0] -> W[0*2+0] = W[0]
    #              C[1,0] -> W[0*2+1] = W[1]
    #              C[0,1] -> W[1*2+0] = W[2]
    #              C[1,1] -> W[1*2+1] = W[3]
    #              etc.
    
    def apply_strassen_block(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, base_rank):
        # Local indices for 2x2 A block
        # A global row = a_r0 + i, col = a_c0 + j
        # A index in U = row * m + col = (a_r0+i) * 4 + (a_c0+j)
        a_idx = [
            a_r0 * m + a_c0, a_r0 * m + a_c0 + 1,
            (a_r0 + 1) * m + a_c0, (a_r0 + 1) * m + a_c0 + 1
        ] # [11, 12, 21, 22]
        
        # Local indices for 2x2 B block
        # B global row = b_r0 + i, col = b_c0 + j
        # B index in V = row * p + col = (b_r0+i) * 4 + (b_c0+j)
        b_idx = [
            b_r0 * p + b_c0, b_r0 * p + b_c0 + 1,
            (b_r0 + 1) * p + b_c0, (b_r0 + 1) * p + b_c0 + 1
        ] # [11, 12, 21, 22]
        
        # Local indices for 2x2 C block
        # C global row = c_r0 + i, col = c_c0 + j
        # W index = col * n + row = (c_c0+j) * 2 + (c_r0+i)
        c_idx = [
            c_c0 * n + c_r0, (c_c0 + 1) * n + c_r0, # C11, C12 (row 0)
            c_c0 * n + (c_r0 + 1), (c_c0 + 1) * n + (c_r0 + 1) # C21, C22 (row 1)
        ]
        # Wait, the previous code used:
        # c11_idx = C_c0 * n + C_r0
        # c12_idx = (C_c0 + 1) * n + C_r0
        # c21_idx = C_c0 * n + (C_r0 + 1)
        # c22_idx = (C_c0 + 1) * n + (C_r0 + 1)
        # This matches col * n + row.
        
        r = base_rank
        
        # M1 = (A11 + A22)(B11 + B22)
        U[a_idx[0]][r] += 1
        U[a_idx[3]][r] += 1
        V[b_idx[0]][r] += 1
        V[b_idx[3]][r] += 1
        W[c_idx[0]][r] += 1 # C11
        W[c_idx[3]][r] += 1 # C22
        r += 1
        
        # M2 = (A21 + A22)B11
        U[a_idx[2]][r] += 1
        U[a_idx[3]][r] += 1
        V[b_idx[0]][r] += 1
        W[c_idx[2]][r] += 1 # C21
        W[c_idx[3]][r] += -1 # C22
        r += 1
        
        # M3 = A11(B12 - B22)
        U[a_idx[0]][r] += 1
        V[b_idx[1]][r] += 1
        V[b_idx[3]][r] += -1
        W[c_idx[1]][r] += 1 # C12
        W[c_idx[3]][r] += 1 # C22
        r += 1
        
        # M4 = A22(B21 - B11)
        U[a_idx[3]][r] += 1
        V[b_idx[2]][r] += 1
        V[b_idx[0]][r] += -1
        W[c_idx[0]][r] += 1 # C11
        W[c_idx[2]][r] += 1 # C21
        r += 1
        
        # M5 = (A11 + A12)B22
        U[a_idx[0]][r] += 1
        U[a_idx[1]][r] += 1
        V[b_idx[3]][r] += 1
        W[c_idx[0]][r] += -1 # C11
        W[c_idx[1]][r] += 1 # C12
        r += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[a_idx[2]][r] += 1
        U[a_idx[0]][r] += -1
        V[b_idx[0]][r] += 1
        V[b_idx[1]][r] += 1
        W[c_idx[3]][r] += 1 # C22
        r += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[a_idx[1]][r] += 1
        U[a_idx[3]][r] += -1
        V[b_idx[2]][r] += 1
        V[b_idx[3]][r] += 1
        W[c_idx[0]][r] += 1 # C11
        r += 1
        
        return r

    # 4 Blocks:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> contributes to C[0:2, 0:2]
    #    A rows 0-1, cols 0-1. B rows 0-1, cols 0-1.
    #    C rows 0-1, cols 0-1.
    r = apply_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> contributes to C[0:2, 0:2]
    #    A rows 0-1, cols 2-3. B rows 2-3, cols 0-1.
    #    C rows 0-1, cols 0-1.
    r = apply_strassen_block(0, 2, 2, 0, 0, 0, r)
    
    # 3. A[0:2, 0:2] * B[0:2, 2:4] -> contributes to C[0:2, 2:4]
    #    A rows 0-1, cols 0-1. B rows 0-1, cols 2-3.
    #    C rows 0-1, cols 2-3.
    r = apply_strassen_block(0, 0, 0, 2, 0, 2, r)
    
    # 4. A[0:2, 2:4] * B[2:4, 2:4] -> contributes to C[0:2, 2:4]
    #    A rows 0-1, cols 2-3. B rows 2-3, cols 2-3.
    #    C rows 0-1, cols 2-3.
    r = apply_strassen_block(0, 2, 2, 2, 0, 2, r)
    
    assert r == 28, f"Rank mismatch: {r}"
    
    if time.time() > deadline:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
