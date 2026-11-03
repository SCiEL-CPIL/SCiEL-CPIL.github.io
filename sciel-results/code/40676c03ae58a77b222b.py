import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Schoolbook rank
    rank_school = n * m * p
    
    # Target rank 54
    rank_target = 54
    
    U = np.zeros((n*m, rank_target))
    V = np.zeros((m*p, rank_target))
    W = np.zeros((p*n, rank_target))
    
    col = 0
    
    # Strassen decomposition for 2x2x2 block
    # Local indices i, j, k in {0, 1}
    # Local a = 2*i + j
    # Local b = 2*j + k
    # Local c = 2*k + i
    # 
    # 7 terms (M1 to M7)
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10
    # M3: A00(B01-B11) -> C01, C11
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> C01, C00
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00, C11
    
    # Define the (U, V, W) coefficients for each of the 7 terms
    # Indices are local: a in 0..3, b in 0..3, c in 0..3
    strassen_terms = []
    
    # M1
    u = [0,0,0,1]; v = [0,0,0,1]; w = [1,0,0,1]
    # a=3 is i=1,j=1. a=0 is i=0,j=0.
    # Wait, local a = 2*i + j.
    # i=0,j=0 -> a=0. i=1,j=1 -> a=3.
    # b=0 is j=0,k=0. b=3 is j=1,k=1.
    # c=0 is k=0,i=0. c=3 is k=1,i=1.
    # M1 = (A00+A11)(B00+B11)
    # U: A00 (a=0), A11 (a=3)
    # V: B00 (b=0), B11 (b=3)
    # W: C00 (c=0), C11 (c=3)
    strassen_terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2: (A10+A11)B00
    # A10: i=1,j=0 -> a=2. A11: i=1,j=1 -> a=3.
    # B00: j=0,k=0 -> b=0.
    # C10: k=0,i=1 -> c=1.
    strassen_terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,0]))
    
    # M3: A00(B01-B11)
    # A00: a=0.
    # B01: j=0,k=1 -> b=1. B11: j=1,k=1 -> b=3.
    # C01: k=1,i=0 -> c=2. C11: k=1,i=1 -> c=3.
    # M3 contributes to C01 (+1) and C11 (-1)?
    # Formula: C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
    # So M3 adds +1 to C01 and +1 to C11?
    # Let's check:
    # C01 = A00 B01 + A01 B11.
    # M3 = A00(B01 - B11) = A00 B01 - A00 B11.
    # M5 = (A00+A01)B11 = A00 B11 + A01 B11.
    # M3 + M5 = A00 B01 - A00 B11 + A00 B11 + A01 B11 = A00 B01 + A01 B11. Correct.
    # So M3 has +1 for C01.
    # Does M3 contribute to C11?
    # C11 = A01 B01 + A11 B11.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # M2 = (A10+A11)B00 = A10B00 + A11B00.
    # M3 = A00(B01-B11) = A00B01 - A00B11.
    # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01.
    # C11 = M1 - M2 + M3 + M6
    # = (A00B00 + A00B11 + A11B00 + A11B11) - (A10B00 + A11B00) + (A00B01 - A00B11) + (A10B00 + A10B01 - A00B00 - A00B01)
    # Terms:
    # A00B00: 1 - 1 = 0.
    # A00B11: 1 - 1 = 0.
    # A11B00: 1 - 1 = 0.
    # A11B11: 1.
    # A10B00: -1 + 1 = 0.
    # A00B01: 1 - 1 = 0.
    # A10B01: 1.
    # Result: A11B11 + A10B01?
    # Wait, C11 should be A01B01 + A11B11.
    # I have A10B01 instead of A01B01.
    # Let's re-verify Strassen formulas.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # My indices:
    # A11 -> i=0,j=0 (a=0). A22 -> i=1,j=1 (a=3).
    # A21 -> i=1,j=0 (a=2). A12 -> i=0,j=1 (a=1).
    # B11 -> j=0,k=0 (b=0). B22 -> j=1,k=1 (b=3).
    # B12 -> j=0,k=1 (b=1). B21 -> j=1,k=0 (b=2).
    # C11 -> k=0,i=0 (c=0). C22 -> k=1,i=1 (c=3).
    # C12 -> k=1,i=0 (c=2). C21 -> k=0,i=1 (c=1).
    #
    # Let's re-evaluate M3 contribution to C22 (c=3).
    # C22 = M1 - M2 + M3 + M6.
    # M3 = A11(B12-B22) = A00(B01-B11).
    # So M3 has +1 for C22?
    # Yes, coefficient is +1.
    #
    # Let's re-evaluate M3 contribution to C12 (c=2).
    # C12 = M3 + M5.
    # M3 has +1 for C12.
    
    # So for M3:
    # U: A00 (a=0).
    # V: B01 (b=1) with +1, B11 (b=3) with -1.
    # W: C12 (c=2) with +1, C22 (c=3) with +1.
    strassen_terms.append(([1,0,0,0], [0,1,0,-1], [0,0,1,1]))
    
    # M4: A22(B21-B11)
    # A22: a=3.
    # B21: b=2. B11: b=0.
    # C11 (c=0) +1. C21 (c=1) +1.
    strassen_terms.append(([0,0,0,1], [0,0,1,-1], [1,1,0,0]))
    
    # M5: (A11+A12)B22
    # A11: a=0. A12: a=1.
    # B22: b=3.
    # C12 (c=2) +1. C11 (c=0) -1.
    strassen_terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    
    # M6: (A21-A11)(B11+B12)
    # A21: a=2. A11: a=0.
    # B11: b=0. B12: b=1.
    # C22 (c=3) +1.
    strassen_terms.append(([0,-1,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7: (A12-A22)(B21+B22)
    # A12: a=1. A22: a=3.
    # B21: b=2. B22: b=3.
    # C11 (c=0) +1. C22 (c=3) +1.
    strassen_terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,1]))
    
    def add_strassen_block(i_off, j_off, k_off, col_start):
        # i_off: offset for i (0 or 1)
        # j_off: offset for j (0, 2, 4)
        # k_off: offset for k (0, 2, 4)
        for t in range(7):
            u_local, v_local, w_local = strassen_terms[t]
            for la in range(4):
                if u_local[la] != 0:
                    li = la // 2
                    lj = la % 2
                    gi = i_off + li
                    gj = j_off + lj
                    ga = gi * m + gj
                    U[ga, col_start + t] = u_local[la]
            for lb in range(4):
                if v_local[lb] != 0:
                    lj = lb // 2
                    lk = lb % 2
                    gj = j_off + lj
                    gk = k_off + lk
                    gb = gj * p + gk
                    V[gb, col_start + t] = v_local[lb]
            for lc in range(4):
                if w_local[lc] != 0:
                    lk = lc // 2
                    li = lc % 2
                    gk = k_off + lk
                    gi = i_off + li
                    gc = gk * n + gi
                    W[gc, col_start + t] = w_local[lc]
        return col_start + 7

    # 6 Strassen blocks
    col = add_strassen_block(0, 0, 0, col)
    col = add_strassen_block(0, 0, 2, col)
    col = add_strassen_block(0, 0, 4, col)
    col = add_strassen_block(0, 2, 0, col)
    col = add_strassen_block(0, 2, 2, col)
    col = add_strassen_block(0, 2, 4, col)
    
    # Schoolbook block for j=4
    # i in 0,1; j=4; k in 0..5
    for i in range(n):
        for k in range(p):
            a = i * m + 4
            b = 4 * p + k
            c = k * n + i
            U[a, col] = 1
            V[b, col] = 1
            W[c, col] = 1
            col += 1
            
    # col should be 54
    if col != rank_target:
        # Fallback to schoolbook if construction fails (should not happen)
        U = np.zeros((n*m, rank_school))
        V = np.zeros((m*p, rank_school))
        W = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
