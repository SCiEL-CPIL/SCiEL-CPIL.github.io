import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication (rank 7).
    U, V, W are 4x7 matrices.
    Indexing:
    U[i*2+j, r]
    V[j*2+k, r]
    W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # A: 00, 11. B: 00, 11. C: 00, 11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # A: 10, 11. B: 00. C: 10
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # C10: k=0, i=1 -> idx 1
    
    # M3 = A00(B01-B11)
    # A: 00. B: 01, 11. C: 01, 11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1  # C01: k=1, i=0 -> idx 2
    W[3, 2] = -1 # C11: k=1, i=1 -> idx 3
    
    # M4 = A11(B10-B00)
    # A: 11. B: 10, 00. C: 00, 10
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C00: k=0, i=0 -> idx 0
    W[1, 3] = 1  # C10: k=0, i=1 -> idx 1
    
    # M5 = (A00+A01)B11
    # A: 00, 01. B: 11. C: 00, 01
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00
    W[2, 4] = 1  # C01
    
    # M6 = (A10-A00)(B00+B01)
    # A: 10, 00. B: 00, 01. C: 11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    # A: 01, 11. B: 10, 11. C: 00, 11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00
    W[3, 6] = -1 # C11
    
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=4):
    """
    Verify decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Check: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    # Vectorized check
    # T_ijk = sum_r U_ijk,r V_ijk,r W_ijk,r
    # We can reshape U, V, W to (n, m, R), (m, p, R), (p, n, R)
    U_reshaped = U.reshape(n, m, R)
    V_reshaped = V.reshape(m, p, R)
    W_reshaped = W.reshape(p, n, R)
    
    # Sum over r
    # U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
    # We need to sum over r for each i, j, k.
    # Note: U depends on i, j. V on j, k. W on k, i.
    # This is a tensor contraction.
    
    # Expand to full tensors? 4x4x4 is small.
    # T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    # But W is indexed by k, i. W_reshaped[k, i, r].
    # So we need to swap axes of W_reshaped?
    # W_reshaped is (p, n, R). i.e., (k, i, r).
    # U_reshaped is (n, m, R). i.e., (i, j, r).
    # V_reshaped is (m, p, R). i.e., (j, k, r).
    
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    return np.allclose(T, 1.0)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try Rank 49 (2-level Strassen)
    U2, V2, W2 = _strassen_2x2()
    
    U = np.zeros((n*m, 49))
    V = np.zeros((m*p, 49))
    W = np.zeros((p*n, 49))
    
    # Block definitions for 4x4 Strassen
    # r: 0..6
    # A blocks, B blocks, C blocks
    # (block_row, block_col) -> coeff
    
    blocks = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): -1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),
    ]
    
    for r in range(7):
        a_blocks, b_blocks, c_blocks = blocks[r]
        
        for s in range(7):
            R_idx = r * 7 + s
            
            # U
            for (bi, bj), ca in a_blocks.items():
                for i in range(2):
                    for j in range(2):
                        val = U2[i*2+j, s]
                        if val != 0:
                            gi = bi*2 + i
                            gj = bj*2 + j
                            U[gi*4 + gj, R_idx] += ca * val
            
            # V
            for (bi, bj), cb in b_blocks.items():
                for j in range(2):
                    for k in range(2):
                        val = V2[j*2+k, s]
                        if val != 0:
                            gj = bi*2 + j
                            gk = bj*2 + k
                            V[gj*4 + gk, R_idx] += cb * val
            
            # W
            for (bi, bj), cc in c_blocks.items():
                for i in range(2):
                    for k in range(2):
                        val = W2[k*2+i, s]
                        if val != 0:
                            gi = bi*2 + i
                            gk = bj*2 + k
                            # W index: k*n + i
                            W[gk*4 + gi, R_idx] += cc * val
    
    if _verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: Rank 56 (1-level Strassen + Schoolbook 2x2)
    # This is guaranteed to work if 49 fails.
    U56 = np.zeros((n*m, 56))
    V56 = np.zeros((m*p, 56))
    W56 = np.zeros((p*n, 56))
    
    # 7 block multiplications
    # Each block mult uses 8 scalar mults (schoolbook 2x2)
    for r in range(7):
        a_blocks, b_blocks, c_blocks = blocks[r]
        
        for s in range(8):
            R_idx = r * 8 + s
            
            # Schoolbook 2x2:
            # M0: A00*B00 -> C00
            # M1: A01*B01 -> C01? No.
            # C00 = A00B00 + A01B10
            # C01 = A00B01 + A01B11
            # C10 = A10B00 + A11B10
            # C11 = A10B01 + A11B11
            
            # s=0: A00, B00, C00
            # s=1: A01, B10, C00
            # s=2: A00, B01, C01
            # s=3: A01, B11, C01
            # s=4: A10, B00, C10
            # s=5: A11, B10, C10
            # s=6: A10, B01, C11
            # s=7: A11, B11, C11
            
            if s == 0:
                ai, aj = 0, 0; bi, bj = 0, 0; ci, cj = 0, 0
            elif s == 1:
                ai, aj = 0, 1; bi, bj = 1, 0; ci, cj = 0, 0
            elif s == 2:
                ai, aj = 0, 0; bi, bj = 0, 1; ci, cj = 0, 1
            elif s == 3:
                ai, aj = 0, 1; bi, bj = 1, 1; ci, cj = 0, 1
            elif s == 4:
                ai, aj = 1, 0; bi, bj = 0, 0; ci, cj = 1, 0
            elif s == 5:
                ai, aj = 1, 1; bi, bj = 1, 0; ci, cj = 1, 0
            elif s == 6:
                ai, aj = 1, 0; bi, bj = 0, 1; ci, cj = 1, 1
            else:
                ai, aj = 1, 1; bi, bj = 1, 1; ci, cj = 1, 1
                
            # Apply to all blocks in a_blocks, b_blocks, c_blocks
            # This is tricky because the local indices (ai, aj) are relative to the block.
            # We need to add to U for each A block, V for each B block, W for each C block.
            
            for (bri, brc), ca in a_blocks.items():
                gi = bri*2 + ai
                gj = brc*2 + aj
                U56[gi*4 + gj, R_idx] += ca
                
            for (bri, brc), cb in b_blocks.items():
                gj = bri*2 + bi
                gk = brc*2 + bj
                V56[gj*4 + gk, R_idx] += cb
                
            for (bri, brc), cc in c_blocks.items():
                gi = bri*2 + ci
                gk = brc*2 + cj
                W56[gk*4 + gi, R_idx] += cc
                
    if _verify(U56, V56, W56):
        return {
            "U": U56.tolist(),
            "V": V56.tolist(),
            "W": W56.tolist()
        }
        
    # Fallback: Schoolbook Rank 64
    U64 = np.zeros((n*m, 64))
    V64 = np.zeros((m*p, 64))
    W64 = np.zeros((p*n, 64))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U64[i*m+j, col] = 1
                V64[j*p+k, col] = 1
                W64[k*n+i, col] = 1
                col += 1
                
    return {
        "U": U64.tolist(),
        "V": V64.tolist(),
        "W": W64.tolist()
    }
# EVOLVE-BLOCK-END
