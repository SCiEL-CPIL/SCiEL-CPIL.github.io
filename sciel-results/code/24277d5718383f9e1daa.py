import time

DIMENSIONS = [2, 3, 10]

def construct():
    t0 = time.time()
    n, m, p = 2, 3, 10
    # Split m = m1 + m2 = 2 + 1
    m1, m2 = 2, 1
    # Split p into pairs: p_pair = 2, 5 pairs
    p_pair = 2
    n_pairs = p // p_pair  # 5

    # Strassen for (n x m1) @ (m1 x p_pair) = 2x2 @ 2x2
    # 7 terms: (A_idx, B_idx, C_coeffs) where A_idx=(i,j1), B_idx=(j2,k), C_coeffs: list of (i,k,coef)
    strassen = [
        # M1 = (A00+A11)(B00+B11)
        ((0, 0, 1), (0, 0, 1), [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)]),
        ((1, 1, 1), (0, 1, 1), [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]),
        # Wait, let me redo this carefully.
        # Standard Strassen 2x2:
        # M1 = (A11 - A22)(B21 + B22)
        # M2 = (A11 + A22)(B22)
        # M3 = (A11 + A12)(B12 - B22)
        # M4 = (A22 + A21)(B11 - B21)
        # M5 = (A11 + A22)(B11 + B22)
        # M6 = (A11 - A12)(B12 + B22)
        # M7 = (A21 - A22)(B11 + B21)
        # C11 = M5 + M4 - M3 - M7
        # C12 = M1 + M7
        # C21 = M2 + M3
        # C22 = M5 - M1 - M2 + M6
        # Using 0-indexed: A[i][j], B[j][k], C[i][k]
        # M1 = (A[1][0] - A[1][1])(B[0][1] + B[1][1])
        # M2 = (A[0][0] + A[1][0])(B[1][1])
        # M3 = (A[0][0] + A[0][1])(B[0][1] - B[1][1])
        # M4 = (A[1][1] + A[1][0])(B[0][0] - B[1][0])
        # M5 = (A[0][0] + A[1][1])(B[0][0] + B[1][1])
        # M6 = (A[0][0] - A[0][1])(B[0][1] + B[1][1])
        # M7 = (A[1][0] - A[1][1])(B[0][0] + B[1][0])
        # C[0][0] = M5 + M4 - M3 - M7
        # C[0][1] = M1 + M7
        # C[1][0] = M2 + M3
        # C[1][1] = M5 - M1 - M2 + M6
    ]
    # Re-define strassen properly with (U_terms, V_terms, W_terms)
    # Each term: U_entries = list of (row, coef), V_entries = list of (row, coef), W_entries = list of (col, coef)
    # U row = i*m + j1, V row = j2*p + k, W col = k*n + i
    
    def strassen_terms(m1, p_pair, n):
        # Returns list of (U_entries, V_entries, W_entries)
        # U_entries: list of (i, j1, coef) -> row = i*m + j1
        # V_entries: list of (j2, k, coef) -> row = j2*p + k  (but k is relative to block start)
        # W_entries: list of (k_rel, i, coef) -> col = (k_block_start + k_rel)*n + i
        terms = []
        # M1 = (A[1][0]-A[1][1])(B[0][1]+B[1][1])
        terms.append(([ (1, 0, 1), (1, 1, -1) ], [ (0, 1, 1), (1, 1, 1) ], [ (0, 1, 1), (0, 0, 0), (1, 0, 0), (1, 1, 0) ]))
        # Wait, W for M1: C[0][1] += M1, C[1][1] += -M1
        # So W entries: (k=1, i=0, +1), (k=1, i=1, -1)
        terms[0] = ([ (1, 0, 1), (1, 1, -1) ], [ (0, 1, 1), (1, 1, 1) ], [ (1, 0, 1), (1, 1, -1) ])
        
        # M2 = (A[0][0]+A[1][0])(B[1][1])
        # C[1][0] += M2, C[1][1] += -M2
        terms.append(([ (0, 0, 1), (1, 0, 1) ], [ (1, 1, 1) ], [ (0, 1, 1), (1, 1, -1) ]))
        
        # M3 = (A[0][0]+A[0][1])(B[0][1]-B[1][1])
        # C[0][0] += -M3, C[1][0] += M3
        terms.append(([ (0, 0, 1), (0, 1, 1) ], [ (0, 1, 1), (1, 1, -1) ], [ (0, 0, -1), (0, 1, 1) ]))
        
        # M4 = (A[1][1]+A[1][0])(B[0][0]-B[1][0])
        # C[0][0] += M4
        terms.append(([ (1, 1, 1), (1, 0, 1) ], [ (0, 0, 1), (1, 0, -1) ], [ (0, 0, 1) ]))
        
        # M5 = (A[0][0]+A[1][1])(B[0][0]+B[1][1])
        # C[0][0] += M5, C[1][1] += M5
        terms.append(([ (0, 0, 1), (1, 1, 1) ], [ (0, 0, 1), (1, 1, 1) ], [ (0, 0, 1), (1, 1, 1) ]))
        
        # M6 = (A[0][0]-A[0][1])(B[0][1]+B[1][1])
        # C[1][1] += M6
        terms.append(([ (0, 0, 1), (0, 1, -1) ], [ (0, 1, 1), (1, 1, 1) ], [ (1, 1, 1) ]))
        
        # M7 = (A[1][0]-A[1][1])(B[0][0]+B[1][0])
        # C[0][0] += -M7, C[0][1] += M7
        terms.append(([ (1, 0, 1), (1, 1, -1) ], [ (0, 0, 1), (1, 0, 1) ], [ (0, 0, -1), (1, 0, 1) ]))
        
        return terms
    
    def direct_terms(m2, p_pair, n):
        # Direct multiplication for (n x m2) @ (m2 x p_pair) = 2x1 @ 1x2
        # 4 terms
        terms = []
        for i in range(n):
            for k in range(p_pair):
                terms.append(([ (i, 0, 1) ], [ (0, k, 1) ], [ (k, i, 1) ]))
        return terms
    
    s_terms = strassen_terms(m1, p_pair, n)
    d_terms = direct_terms(m2, p_pair, n)
    
    rank = n_pairs * (len(s_terms) + len(d_terms))
    # rank = 5 * (7 + 4) = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(n_pairs):
        k_start = b * p_pair
        # Strassen terms
        for ut, vt, wt in s_terms:
            for (i, j1, coef) in ut:
                U[i * m + j1][col] += coef
            for (j2, k_rel, coef) in vt:
                V[(j2) * p + (k_start + k_rel)][col] += coef
            for (k_rel, i, coef) in wt:
                W[(k_start + k_rel) * n + i][col] += coef
            col += 1
        # Direct terms
        for ut, vt, wt in d_terms:
            for (i, j2, coef) in ut:
                U[i * m + (m1 + j2)][col] += coef
            for (j2, k_rel, coef) in vt:
                V[(m1 + j2) * p + (k_start + k_rel)][col] += coef
            for (k_rel, i, coef) in wt:
                W[(k_start + k_rel) * n + i][col] += coef
            col += 1
    
    return {"U": U, "V": V, "W": W}
