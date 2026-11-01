import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # Rank 42 solution: 2 blocks of j (each 2x2) × 3 blocks of k (each 2x2) × 7 Strassen products
    rank = 42
    
    U = np.zeros((n*m, rank), dtype=np.int8)
    V = np.zeros((m*p, rank), dtype=np.int8)
    W = np.zeros((p*n, rank), dtype=np.int8)
    
    # Strassen 2x2 @ 2x2 decomposition
    # Local index mapping: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # U[i,j], V[j,k], W[k,i]
    
    # Standard Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_U = [
        {0: 1, 3: 1},       # M1: A00+A11
        {2: 1, 3: 1},       # M2: A10+A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00+A01
        {2: 1, 0: -1},      # M6: A10-A00
        {1: 1, 3: -1},      # M7: A01-A11
    ]
    strassen_V = [
        {0: 1, 3: 1},       # M1: B00+B11
        {0: 1},             # M2: B00
        {2: 1, 3: -1},      # M3: B10-B11
        {1: 1, 2: -1},      # M4: B01-B10
        {3: 1},             # M5: B11
        {0: 1, 2: 1},       # M6: B00+B10
        {1: 1, 3: 1},       # M7: B01+B11
    ]
    strassen_W = [
        {0: 1, 3: 1},       # M1: C00+C11
        {2: 1, 3: -1},      # M2: C10-C11
        {1: 1, 3: 1},       # M3: C01+C11
        {0: 1, 2: 1},       # M4: C00+C10
        {0: -1, 1: 1},      # M5: -C00+C01
        {3: 1},             # M6: C11
        {0: 1},             # M7: C00
    ]
    
    col = 0
    for j_block in range(2):  # j = 0,1 and j = 2,3
        for k_block in range(3):  # k = 0,1 and k = 2,3 and k = 4,5
            for prod_idx in range(7):
                r = col + prod_idx
                
                # U part: A[i, j]
                for local_ij, sign in strassen_U[prod_idx].items():
                    gi = local_ij // 2
                    gj = local_ij % 2
                    U[gi * m + (j_block * 2 + gj), r] = sign
                
                # V part: B[j, k]
                for local_jk, sign in strassen_V[prod_idx].items():
                    gj = local_jk // 2
                    gk = local_jk % 2
                    V[(j_block * 2 + gj) * p + (k_block * 2 + gk), r] = sign
                
                # W part: C[i, k]
                for local_ik, sign in strassen_W[prod_idx].items():
                    gi = local_ik // 2
                    gk = local_ik % 2
                    W[(k_block * 2 + gk) * n + gi, r] = sign
            
            col += 7
    
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
