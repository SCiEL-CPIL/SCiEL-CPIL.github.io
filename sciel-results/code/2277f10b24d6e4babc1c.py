import time
import sys

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # We will build the decomposition by concatenating smaller decompositions.
    # Part 1: Top 2 rows (i=0,1)
    # This is a 2x4x8 multiplication.
    # Decomposed into 8 blocks of 2x2x2.
    # j ranges 0..3, split into [0,1] and [2,3].
    # k ranges 0..7, split into [0,1], [2,3], [4,5], [6,7].
    
    # Part 2: Bottom 1 row (i=2)
    # This is a 1x4x8 multiplication.
    # Decomposed into 8 blocks of 1x4x1.
    # k ranges 0..7.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        U_cols.append([0] * len(U_cols[0]) if U_cols else [0]) # This is inefficient, let's build lists of lists
        # Better to build lists of (index, coeff) pairs and then expand
        pass

    # Let's build lists of tuples (index, coeff) for U, V, W
    # Then convert to dense matrices at the end.
    
    u_terms = []
    v_terms = []
    w_terms = []
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # We need to define the linear combinations for U (A matrix) and V (B matrix)
    # and the resulting W (C matrix) coefficients.
    
    # Strassen U coefficients (for A indices 00, 01, 10, 11)
    # Local indices: 00=0, 01=1, 10=2, 11=3 (row-major i*m+j)
    # U is 2x2.
    # M1: a00=1, a11=1
    # M2: a10=1, a11=1
    # M3: a00=1
    # M4: a11=1
    # M5: a00=1, a01=1
    # M6: a10=1, a00=-1
    # M7: a01=1, a11=-1
    
    strassen_u = [
        {0: 1, 3: 1},   # M1
        {2: 1, 3: 1},   # M2
        {0: 1},         # M3
        {3: 1},         # M4
        {0: 1, 1: 1},   # M5
        {2: 1, 0: -1},  # M6
        {1: 1, 3: -1}   # M7
    ]
    
    # Strassen V coefficients (for B indices 00, 01, 10, 11)
    # V is 2x2.
    # M1: b00=1, b11=1
    # M2: b00=1
    # M3: b01=1, b11=-1
    # M4: b10=1, b00=-1
    # M5: b11=1
    # M6: b00=1, b01=1
    # M7: b10=1, b11=1
    
    strassen_v = [
        {0: 1, 3: 1},   # M1
        {0: 1},         # M2
        {1: 1, 3: -1},  # M3
        {2: 1, 0: -1},  # M4
        {3: 1},         # M5
        {0: 1, 1: 1},   # M6
        {2: 1, 3: 1}    # M7
    ]
    
    # Strassen W coefficients (for C indices 00, 01, 10, 11)
    # W is 2x2.
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # W index mapping: k*n + i.
    # For 2x2 block, n=2 (local).
    # c00: k=0, i=0 -> idx 0
    # c01: k=1, i=0 -> idx 1
    # c10: k=0, i=1 -> idx 2
    # c11: k=1, i=1 -> idx 3
    
    strassen_w = [
        {0: 1, 2: 1, 1: -1, 3: 1}, # M1 contributes to c00(1), c11(1)? No.
        # Let's re-evaluate c00 = M1 + M4 - M5 + M7
        # M1: c00 += 1, c11 += 1
        # M2: c10 += 1, c11 -= 1
        # M3: c01 += 1, c11 += 1
        # M4: c00 += 1, c10 += 1
        # M5: c00 -= 1, c01 += 1
        # M6: c11 += 1
        # M7: c00 += 1
        # 
        # So for each M term, we add to W columns.
        # We are building columns of W.
        # Each Strassen term is a rank-1 component.
        # The W column for that term will have non-zero entries at the C indices.
        
        # M1: c00=1, c11=1
        # M2: c10=1, c11=-1
        # M3: c01=1, c11=1
        # M4: c00=1, c10=1
        # M5: c00=-1, c01=1
        # M6: c11=1
        # M7: c00=1
    ]
    
    strassen_w_cols = [
        {0: 1, 3: 1},  # M1: c00, c11
        {2: 1, 3: -1}, # M2: c10, c11
        {1: 1, 3: 1},  # M3: c01, c11
        {0: 1, 2: 1},  # M4: c00, c10
        {0: -1, 1: 1}, # M5: c00, c01
        {3: 1},        # M6: c11
        {0: 1}         # M7: c00
    ]
    
    # Add Strassen blocks for i=0,1
    for j_base in [0, 2]:
        for k_base in [0, 2, 4, 6]:
            for t in range(7):
                # U terms
                u_dict = strassen_u[t]
                v_dict = strassen_v[t]
                w_dict = strassen_w_cols[t]
                
                u_col = [0] * (n * m)
                v_col = [0] * (m * p)
                w_col = [0] * (p * n)
                
                # Map local U indices to global
                # Local U index: i_local * 2 + j_local
                # Global U index: (i_base + i_local) * m + (j_base + j_local)
                # Here i_base=0, m=4.
                for u_loc, coeff in u_dict.items():
                    i_loc = u_loc // 2
                    j_loc = u_loc % 2
                    u_glob = (0 + i_loc) * m + (j_base + j_loc)
                    u_col[u_glob] = coeff
                
                # Map local V indices to global
                # Local V index: j_local * 2 + k_local
                # Global V index: (j_base + j_local) * p + (k_base + k_local)
                for v_loc, coeff in v_dict.items():
                    j_loc = v_loc // 2
                    k_loc = v_loc % 2
                    v_glob = (j_base + j_loc) * p + (k_base + k_loc)
                    v_col[v_glob] = coeff
                
                # Map local W indices to global
                # Local W index: k_local * 2 + i_local (n=2 locally)
                # Global W index: (k_base + k_local) * n + (0 + i_local)
                for w_loc, coeff in w_dict.items():
                    k_loc = w_loc // 2
                    i_loc = w_loc % 2
                    w_glob = (k_base + k_loc) * n + (0 + i_loc)
                    w_col[w_glob] = coeff
                
                u_terms.append(u_col)
                v_terms.append(v_col)
                w_terms.append(w_col)
                
    # Add Schoolbook blocks for i=2
    # 1x4x8
    # For each j in 0..3, k in 0..7
    for j in range(m):
        for k in range(p):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            u_glob = 2 * m + j
            v_glob = j * p + k
            w_glob = k * n + 2
            
            u_col[u_glob] = 1
            v_col[v_glob] = 1
            w_col[w_glob] = 1
            
            u_terms.append(u_col)
            v_terms.append(v_col)
            w_terms.append(w_col)
            
    rank = len(u_terms)
    
    # Convert lists of lists to matrices (lists of lists)
    # The interface expects {'U': U, 'V': V, 'W': W}
    # U is (n*m, rank)
    # V is (m*p, rank)
    # W is (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = u_terms[r][i]
        for i in range(m * p):
            V[i][r] = v_terms[r][i]
        for i in range(p * n):
            W[i][r] = w_terms[r][i]
            
    return {"U": U, "V": V, "W": W}

