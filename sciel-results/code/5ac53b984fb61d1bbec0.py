import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use 3 Strassen blocks of size 2x2x2
    # Each block handles 2 columns of B
    # Rank per block is 7
    total_rank = 21
    
    U = [[0] * total_rank for _ in range(n * m)] # 4 x 21
    V = [[0] * total_rank for _ in range(m * p)] # 12 x 21
    W = [[0] * total_rank for _ in range(p * n)] # 12 x 21
    
    # Strassen factors for 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A11 - A01)(B10 + B11)
    
    # Indices in U (A): 0=A00, 1=A01, 2=A10, 3=A11
    # Indices in V (B): 0=B00, 1=B01, 2=B10, 3=B11
    # Indices in W (C): 0=C00, 1=C01, 2=C10, 3=C11
    
    # Local factors for one Strassen block
    # Each entry is (U_coeff, V_coeff, W_coeff)
    # We need to add these to the global U, V, W
    
    strassen_terms = [
        # M1
        ( {0: 1, 3: 1}, {0: 1, 2: 1}, {0: 1, 1: 0, 2: 0, 3: 1} ), # C00+=1, C11+=1
        # M2
        ( {2: 1, 3: 1}, {0: 1}, {0: 0, 1: 0, 2: 1, 3: 0} ), # C10+=1
        # M3
        ( {0: 1}, {2: 1, 3: -1}, {0: 0, 1: 1, 2: 0, 3: 0} ), # C01+=1
        # M4
        ( {3: 1}, {1: 1, 0: -1}, {0: -1, 1: 0, 2: 1, 3: 0} ), # C00-=-1, C10+=1
        # M5
        ( {0: 1, 1: 1}, {2: 1}, {0: 0, 1: 1, 2: 0, 3: 0} ), # C01+=1
        # M6
        ( {2: 1, 0: -1}, {0: 1, 1: 1}, {0: 1, 1: 0, 2: 0, 3: 0} ), # C00+=1
        # M7
        ( {3: 1, 1: -1}, {2: 1, 3: 1}, {0: 0, 1: 0, 2: 0, 3: -1} ), # C11-=-1
    ]
    
    # Wait, the W coefficients above are for the local C matrix.
    # We need to map local C indices to global C indices.
    # Global C is 2x6.
    # W index is k*n + i.
    # Local C00 -> global C[0, col_offset] -> W[0*2 + 0] = W[0]
    # Local C01 -> global C[0, col_offset+1] -> W[1*2 + 0] = W[2]
    # Local C10 -> global C[1, col_offset] -> W[0*2 + 1] = W[1]
    # Local C11 -> global C[1, col_offset+1] -> W[1*2 + 1] = W[3]
    
    # Let's re-verify the W mapping.
    # W[k*n + i, r]
    # i is row of C (0,1)
    # k is col of C (0..5)
    # Local col 0 -> global col offset
    # Local col 1 -> global col offset + 1
    
    # Local C00 (i=0, k=0) -> W[0*2 + 0] = W[0]
    # Local C01 (i=0, k=1) -> W[1*2 + 0] = W[2]
    # Local C10 (i=1, k=0) -> W[0*2 + 1] = W[1]
    # Local C11 (i=1, k=1) -> W[1*2 + 1] = W[3]
    
    # My previous W dict keys were 0,1,2,3 for C00,C01,C10,C11.
    # I need to map these to W indices.
    # C00 -> 0
    # C01 -> 2
    # C10 -> 1
    # C11 -> 3
    
    # Let's rewrite the terms with explicit W indices
    # Term: (U_dict, V_dict, W_dict)
    # U_dict keys: 0=A00, 1=A01, 2=A10, 3=A11
    # V_dict keys: 0=B00, 1=B01, 2=B10, 3=B11
    # W_dict keys: 0=C00, 1=C10, 2=C01, 3=C11 (mapped to W indices)
    
    # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # U: {0:1, 3:1}, V: {0:1, 2:1}, W: {0:1, 3:1}
    # M2: (A10+A11)B00 -> C10 += 1
    # U: {2:1, 3:1}, V: {0:1}, W: {1:1}
    # M3: A00(B10-B11) -> C01 += 1
    # U: {0:1}, V: {2:1, 3:-1}, W: {2:1}
    # M4: A11(B01-B00) -> C00 -= 1, C10 += 1
    # U: {3:1}, V: {1:1, 0:-1}, W: {0:-1, 1:1}
    # M5: (A00+A01)B11 -> C01 += 1
    # U: {0:1, 1:1}, V: {2:1}, W: {2:1}
    # M6: (A10-A00)(B00+B01) -> C00 += 1
    # U: {2:1, 0:-1}, V: {0:1, 1:1}, W: {0:1}
    # M7: (A11-A01)(B10+B11) -> C11 -= 1
    # U: {3:1, 1:-1}, V: {2:1, 3:1}, W: {3:-1}
    
    terms = [
        ({0:1, 3:1}, {0:1, 2:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {1:1}),
        ({0:1}, {2:1, 3:-1}, {2:1}),
        ({3:1}, {1:1, 0:-1}, {0:-1, 1:1}),
        ({0:1, 1:1}, {2:1}, {2:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {0:1}),
        ({3:1, 1:-1}, {2:1, 3:1}, {3:-1})
    ]
    
    for block_idx in range(3):
        col_offset = block_idx * 2
        for term_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
            r = block_idx * 7 + term_idx
            # Update U
            for u_idx, val in u_dict.items():
                U[u_idx][r] += val
            # Update V
            # V index is j*p + k
            # Local B00 (j=0, k=0) -> global j=0, k=col_offset
            # Local B01 (j=0, k=1) -> global j=0, k=col_offset+1
            # Local B10 (j=1, k=0) -> global j=1, k=col_offset
            # Local B11 (j=1, k=1) -> global j=1, k=col_offset+1
            for v_idx, val in v_dict.items():
                j = v_idx // 2
                k_local = v_idx % 2
                k_global = col_offset + k_local
                v_global_idx = j * p + k_global
                V[v_global_idx][r] += val
            # Update W
            # W index is k*n + i
            # Local C00 (i=0, k=0) -> global i=0, k=col_offset
            # Local C01 (i=0, k=1) -> global i=0, k=col_offset+1
            # Local C10 (i=1, k=0) -> global i=1, k=col_offset
            # Local C11 (i=1, k=1) -> global i=1, k=col_offset+1
            # My w_dict keys: 0=C00, 1=C10, 2=C01, 3=C11
            for w_idx, val in w_dict.items():
                if w_idx == 0: # C00
                    i = 0
                    k_local = 0
                elif w_idx == 1: # C10
                    i = 1
                    k_local = 0
                elif w_idx == 2: # C01
                    i = 0
                    k_local = 1
                elif w_idx == 3: # C11
                    i = 1
                    k_local = 1
                k_global = col_offset + k_local
                w_global_idx = k_global * n + i
                W[w_global_idx][r] += val
                
    return {"U": U, "V": V, "W": W}
