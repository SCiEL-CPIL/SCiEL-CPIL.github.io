import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=15
    # Rank 53 construction using Strassen for 2x2x2 blocks and trivial for 2x2x1
    
    rank = 53
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term
    # u_idx: (i, j) -> i*m+j
    # v_idx: (j, k) -> j*p+k
    # w_idx: (k, i) -> k*n+i
    # val: coefficient
    
    def add_term(r_idx, u_entries, v_entries, w_entries):
        for (i, j), val in u_entries.items():
            if val != 0:
                U[i * m + j, r_idx] += val
        for (j, k), val in v_entries.items():
            if val != 0:
                V[j * p + k, r_idx] += val
        for (k, i), val in w_entries.items():
            if val != 0:
                W[k * n + i, r_idx] += val

    # Strassen terms for 2x2x2
    # Local indices i,j,k in {0,1}
    # Global k offset by k_start
    
    strassen_terms = [
        # (U_entries, V_entries, W_entries)
        # Term 1: M1 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # Term 2: M2 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # Term 3: M3 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        # Term 4: M4 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # Term 5: M5 = (A00+A10)B11
        ({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        # Term 6: M6 = (A01-A00)(B00+B10)
        ({(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        # Term 7: M7 = (A10-A11)(B01+B11)
        ({(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Process 7 blocks of size 2 (total 14 columns)
    for block_idx in range(7):
        k_start = block_idx * 2
        for t_idx, (u_ent, v_ent, w_ent) in enumerate(strassen_terms):
            r_global = block_idx * 7 + t_idx
            
            # Map local indices to global
            u_global = u_ent
            v_global = {}
            for (j, k_local), val in v_ent.items():
                k_global = k_start + k_local
                v_global[(j, k_global)] = val
            w_global = {}
            for (k_local, i), val in w_ent.items():
                k_global = k_start + k_local
                w_global[(k_global, i)] = val
                
            add_term(r_global, u_global, v_global, w_global)
            
    # Process remaining 1 column (k=14)
    # Trivial: C[i, 14] = A[i,0]B[0,14] + A[i,1]B[1,14]
    # 4 terms
    
    # Term 49: i=0, j=0
    add_term(49, {(0,0): 1}, {(0,14): 1}, {(14,0): 1})
    # Term 50: i=1, j=0
    add_term(50, {(1,0): 1}, {(0,14): 1}, {(14,1): 1})
    # Term 51: i=0, j=1
    add_term(51, {(0,1): 1}, {(1,14): 1}, {(14,0): 1})
    # Term 52: i=1, j=1
    add_term(52, {(1,1): 1}, {(1,14): 1}, {(14,1): 1})
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
