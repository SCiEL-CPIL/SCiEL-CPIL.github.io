import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct Strassen decomposition for 2x2x2
    # U, V, W are 4x7 matrices
    U222 = np.zeros((4, 7))
    V222 = np.zeros((4, 7))
    W222 = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U222[0, 0] = 1 # A00
    U222[3, 0] = 1 # A11 (index 1*2+1=3)
    V222[0, 0] = 1 # B00
    V222[3, 0] = 1 # B11 (index 1*2+1=3)
    W222[0, 0] = 1 # C00 (index 0*2+0=0)
    W222[3, 0] = 1 # C11 (index 1*2+1=3)
    
    # r=1: M2 = (A10+A11)B00
    U222[2, 1] = 1 # A10 (index 1*2+0=2)
    U222[3, 1] = 1 # A11
    V222[0, 1] = 1 # B00
    W222[2, 1] = 1 # C10 (index 0*2+1=2? No. C index is k*n+i. k=0, i=1 -> 1? 
                    # Wait. W index is k*n + i.
                    # For 2x2, n=2.
                    # C10: k=0, i=1. Index = 0*2 + 1 = 1.
                    # C11: k=1, i=1. Index = 1*2 + 1 = 3.
                    # C00: k=0, i=0. Index = 0.
                    # C01: k=1, i=0. Index = 2.
    
    # Let's re-verify W indices for 2x2.
    # W[k*n + i, r]
    # k is row of C (0,1). i is col of C? No.
    # Prompt: W[k*n+i, r].
    # Usually C is n x p. Here n=2, p=2.
    # C_{i,k}? Or C_{k,i}?
    # Prompt says: "output factor follows the official trace convention W[k*n+i,r]"
    # And U is n*m, V is m*p.
    # Standard matrix mult: C = A B. A is n x m, B is m x p. C is n x p.
    # Indices of C are (row, col) = (i, k) where i in 0..n-1, k in 0..p-1.
    # But W index is k*n + i. This implies the first index is k (col of C) and second is i (row of C).
    # So W is indexed by (col, row).
    # Let's check the sum condition again.
    # Sum_r U[i*m+j, r] V[j'*p+k, r] W[k'*n+i', r] = delta(j,j') delta(i,i') delta(k,k')
    # Here i is row of A, j is col of A.
    # j' is row of B, k is col of B.
    # i' is row of C, k' is col of C.
    # The delta(i,i') and delta(k,k') match C indices.
    # So W index is k'*n + i'.
    # In my 2x2 example:
    # C00: i'=0, k'=0. Index = 0*2 + 0 = 0.
    # C01: i'=0, k'=1. Index = 1*2 + 0 = 2.
    # C10: i'=1, k'=0. Index = 0*2 + 1 = 1.
    # C11: i'=1, k'=1. Index = 1*2 + 1 = 3.
    
    # So:
    # M1 contributes to C00 (idx 0) and C11 (idx 3).
    # M2 contributes to C10 (idx 1) and C11 (idx 3).
    # M3 contributes to C01 (idx 2) and C11 (idx 3).
    # M4 contributes to C00 (idx 0) and C10 (idx 1).
    # M5 contributes to C01 (idx 2) and C00 (idx 0).
    # M6 contributes to C11 (idx 3).
    # M7 contributes to C00 (idx 0).
    
    # Re-filling W222:
    W222[0, 0] = 1 # C00
    W222[3, 0] = 1 # C11
    
    W222[1, 1] = 1 # C10
    W222[3, 1] = -1 # C11
    
    W222[2, 2] = 1 # C01
    W222[3, 2] = 1 # C11
    
    W222[0, 3] = 1 # C00
    W222[1, 3] = 1 # C10
    
    W222[2, 4] = 1 # C01
    W222[0, 4] = -1 # C00
    
    W222[3, 5] = 1 # C11
    
    W222[0, 6] = 1 # C00

    # 2. Construct 4x4x4 decomposition using Kronecker product
    rank_444 = 49
    U444 = np.zeros((16, rank_444))
    V444 = np.zeros((16, rank_444))
    W444 = np.zeros((16, rank_444))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            idx_u = i * 4 + j
                            val = U222[i1 * 2 + j1, r1] * U222[i2 * 2 + j2, r2]
                            U444[idx_u, r] = val
                            
                            # V indices: j' (row of B), k (col of B)
                            # B is 4x4. j' in 0..3, k in 0..3.
                            # j' = j1 * 2 + j2'? No.
                            # B indices: row j', col k.
                            # In Kronecker product, B is block matrix.
                            # B_{j', k} = B1_{j1, k1} * B2_{j2, k2}
                            # j' = j1 * 2 + j2, k = k1 * 2 + k2
                            for k1 in range(2):
                                for k2 in range(2):
                                    jp = j1 * 2 + j2
                                    k = k1 * 2 + k2
                                    idx_v = jp * 4 + k
                                    val_v = V222[j1 * 2 + k1, r1] * V222[j2 * 2 + k2, r2]
                                    V444[idx_v, r] = val_v
                                    
                                    # W indices: k' (col of C), i' (row of C)
                                    # C_{i', k'} = C1_{i1, k1} * C2_{i2, k2}
                                    # i' = i1 * 2 + i2, k' = k1 * 2 + k2
                                    ip = i1 * 2 + i2
                                    kp = k1 * 2 + k2
                                    idx_w = kp * 4 + ip
                                    val_w = W222[k1 * 2 + i1, r1] * W222[k2 * 2 + i2, r2]
                                    W444[idx_w, r] = val_w

    # 3. Construct 4x4x1 decomposition (rank 16)
    rank_441 = 16
    U441 = np.zeros((16, rank_441))
    V441 = np.zeros((4, rank_441)) # m*p = 4*1 = 4
    W441 = np.zeros((4, rank_441)) # p*n = 1*4 = 4
    
    for r in range(16):
        i = r // 4
        j = r % 4
        U441[i * 4 + j, r] = 1
        V441[j, r] = 1 # j is row of B (0..3), k=0. Index j*1+0 = j.
        W441[i, r] = 1 # k=0, i is row of C. Index 0*4+i = i.

    # 4. Combine into total decomposition
    total_rank = rank_444 + rank_441
    U_total = np.zeros((16, total_rank))
    V_total = np.zeros((20, total_rank)) # m*p = 4*5 = 20
    W_total = np.zeros((20, total_rank)) # p*n = 5*4 = 20
    
    # Copy 444 part
    U_total[:, :rank_444] = U444
    V_total[:16, :rank_444] = V444
    W_total[:16, :rank_444] = W444
    
    # Copy 441 part
    U_total[:, rank_444:] = U441
    V_total[16:, rank_444:] = V441
    W_total[16:, rank_444:] = W441
    
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

