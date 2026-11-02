import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition for 4x4x4 (rank 49) and 4x4x1 (rank 16)
    # Total rank 65
    
    # Helper to get base 2x2x2 Strassen terms
    def get_base_2x2_terms():
        terms = []
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        terms.append((
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}
        ))
        # M2 = (A10+A11)B00 -> C10, C11
        terms.append((
            {(1,0): 1, (1,1): 1}, 
            {(0,0): 1}, 
            {(0,1): 1, (1,1): -1} # C10 is (1,0) in C? No.
            # C indices are (i,k).
            # C10: i=1, k=0.
            # C11: i=1, k=1.
            # M2 contributes to C10 (+1) and C11 (-1).
            # So W coeffs: {(0,1): 1, (1,1): -1}
        ))
        # M3 = A00(B01-B11) -> C01, C11
        terms.append((
            {(0,0): 1}, 
            {(0,1): 1, (1,1): -1}, 
            {(1,0): 1, (1,1): 1} # C01: i=0,k=1. C11: i=1,k=1.
        ))
        # M4 = A11(B10-B00) -> C00, C10
        terms.append((
            {(1,1): 1}, 
            {(1,0): 1, (0,0): -1}, 
            {(0,0): 1, (0,1): 1} # C00: i=0,k=0. C10: i=1,k=0.
        ))
        # M5 = (A00+A01)B11 -> C00, C01
        terms.append((
            {(0,0): 1, (0,1): 1}, 
            {(1,1): 1}, 
            {(0,0): -1, (1,0): 1} # C00: -1. C01: 1.
        ))
        # M6 = (A10-A00)(B00+B01) -> C11
        terms.append((
            {(1,0): 1, (0,0): -1}, 
            {(0,0): 1, (0,1): 1}, 
            {(1,1): 1} # C11: 1.
        ))
        # M7 = (A01-A11)(B10+B11) -> C00, C11
        terms.append((
            {(0,1): 1, (1,1): -1}, 
            {(1,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1} # C00: 1. C11: 1.
        ))
        return terms

    def get_strassen_terms(n):
        if n == 2:
            return get_base_2x2_terms()
        
        m = n // 2
        sub_terms = get_strassen_terms(m)
        
        # Define the 7 block-level Strassen combinations
        # Each is (U_blocks, V_blocks, W_blocks)
        # U_blocks: dict mapping (bi, bj) to coeff
        block_combos = [
            # M1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
            # M3
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
            # M4
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            # M5
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
            # M6
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
        ]
        
        terms = []
        for u_blocks, v_blocks, w_blocks in block_combos:
            for u_sub, v_sub, w_sub in sub_terms:
                # Combine block indices and sub indices
                u_full = {}
                for (bi, bj), cu in u_blocks.items():
                    for (si, sj), cs in u_sub.items():
                        i = bi * m + si
                        j = bj * m + sj
                        u_full[(i, j)] = u_full.get((i, j), 0) + cu * cs
                
                v_full = {}
                for (bi, bj), cv in v_blocks.items():
                    for (si, sj), cs in v_sub.items():
                        i = bi * m + si
                        j = bj * m + sj
                        v_full[(i, j)] = v_full.get((i, j), 0) + cv * cs
                
                w_full = {}
                for (bi, bj), cw in w_blocks.items():
                    for (si, sj), cs in w_sub.items():
                        i = bi * m + si
                        j = bj * m + sj
                        w_full[(i, j)] = w_full.get((i, j), 0) + cw * cs
                
                terms.append((u_full, v_full, w_full))
        
        return terms

    # Get terms for 4x4x4
    terms_444 = get_strassen_terms(4)
    
    # Add terms for the last column (4x4x1)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # W index for C[i, 4] is k=4, i=i. Row = 4*4 + i = 16 + i.
    terms_last_col = []
    for i in range(4):
        for j in range(4):
            u_full = {(i, j): 1}
            v_full = {(j, 4): 1}
            w_full = {(4, i): 1} # k=4, i=i
            terms_last_col.append((u_full, v_full, w_full))
    
    all_terms = terms_444 + terms_last_col
    rank = len(all_terms)
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_full, v_full, w_full) in enumerate(all_terms):
        for (i, j), val in u_full.items():
            U[i * m + j, r] = val
        for (j, k), val in v_full.items():
            V[j * p + k, r] = val
        for (k, i), val in w_full.items():
            W[k * n + i, r] = val
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
