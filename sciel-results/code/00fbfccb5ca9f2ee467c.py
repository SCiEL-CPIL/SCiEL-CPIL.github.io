"""Matrix multiplication 4x4x4 tensor decomposition via recursive Strassen (rank 49)."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 exact tensor decomposition for 4x4 x 4x4 matrix multiplication.
    
    Uses two levels of Strassen:
      - Outer level: 4x4 split into 2x2 blocks of 2x2 matrices, 7 block-multiplications
      - Inner level: each 2x2 x 2x2 uses Strassen, 7 scalar multiplications
      - Total rank: 7 * 7 = 49
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      Reconstruction: sum_r U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r] = 1 for all i,k,j
    """
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n = m = p = 4
    rank = 49

    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((16, rank), dtype=np.float64)

    # Build the 2x2 Strassen decomposition (rank 7)
    # U2[i*2+k, r], V2[k*2+j, r], W2[j*2+i, r]
    U2 = np.zeros((4, 7), dtype=np.float64)
    V2 = np.zeros((4, 7), dtype=np.float64)
    W2 = np.zeros((4, 7), dtype=np.float64)
    
    # s1: (A00+A11)(B00+B11) -> C00+=s1, C11+=s1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # s2: (A10+A11)B00 -> C10+=s2, C11-=s2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # s3: A00(B01-B11) -> C01+=s3, C11+=s3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # s4: A11(B10-B00) -> C00+=s4, C10+=s4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # s5: (A00+A01)B11 -> C00-=s5, C01+=s5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # s6: (A10-A00)(B00+B01) -> C11+=s6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # s7: (A01-A11)(B10+B11) -> C00+=s7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Define the 7 outer Strassen terms
    # Each term: (left_block_coeffs, right_block_coeffs, c_block_coeffs)
    # Block positions: (0,0)=top-left, (0,1)=top-right, (1,0)=bottom-left, (1,1)=bottom-right
    
    outer_def = [
        # M1: (A11+A22)(B11+B22) -> C11+=M1, C22+=M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (A21+A22)B11 -> C21+=M2, C22-=M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3: A11(B12-B22) -> C12+=M3, C22+=M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4: A22(B21-B11) -> C11+=M4, C21+=M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5: (A11+A12)B22 -> C11-=M5, C12+=M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (A21-A11)(B11+B12) -> C22+=M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7: (A12-A22)(B21+B22) -> C11+=M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Build U, V, W for the 4x4 decomposition
    for t in range(7):
        left_c, right_c, c_c = outer_def[t]
        for k in range(7):
            li = t * 7 + k
            
            if time.time() - t0 > deadline:
                break
            
            # Build U: U[i*4+ak, li]
            # A element (i, ak) is in block (i//2, ak//2) at local pos (i%2, ak%2)
            # U2[local_i*2+local_k, k] gives coefficient for L[local_i, local_k]
            for i in range(4):
                for ak in range(4):
                    br = i // 2
                    bc = ak // 2
                    li_local = i % 2
                    lk_local = ak % 2
                    coeff = U2[li_local*2 + lk_local, k] * left_c.get((br, bc), 0)
                    if coeff != 0:
                        U[i*4+ak, li] = coeff
            
            # Build V: V[bk*4+j, li]
            # B element (bk, j) is in block (bk//2, j//2) at local pos (bk%2, j%2)
            # V2[local_k*2+local_j, k] gives coefficient for R[local_k, local_j]
            for bk in range(4):
                for j in range(4):
                    br = bk // 2
                    bc = j // 2
                    lk_local = bk % 2
                    lj_local = j % 2
                    coeff = V2[lk_local*2 + lj_local, k] * right_c.get((br, bc), 0)
                    if coeff != 0:
                        V[bk*4+j, li] = coeff
            
            # Build W: W[cj*4+ci, li]
            # C element (ci, cj) is in block (ci//2, cj//2) at local pos (ci%2, cj%2)
            # W2[local_j*2+local_i, k] gives coefficient for P[local_j, local_i]
            for cj in range(4):
                for ci in range(4):
                    cbr = ci // 2
                    cbc = cj // 2
                    li_local = ci % 2
                    lj_local = cj % 2
                    coeff = W2[lj_local*2 + li_local, k] * c_c.get((cbr, cbc), 0)
                    if coeff != 0:
                        W[cj*4+ci, li] = coeff
    
    # Verify decomposition
    err = 0.0
    for i in range(4):
        for k in range(4):
            for j in range(4):
                s = 0.0
                for r in range(49):
                    s += U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r]
                err += abs(s - 1.0)
    
    if err > 1e-6:
        # Fallback to schoolbook rank-64
        U_f = np.zeros((16, 64), dtype=np.float64)
        V_f = np.zeros((16, 64), dtype=np.float64)
        W_f = np.zeros((16, 64), dtype=np.float64)
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    U_f[i*4+k, idx] = 1.0
                    V_f[k*4+j, idx] = 1.0
                    W_f[j*4+i, idx] = 1.0
                    idx += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
