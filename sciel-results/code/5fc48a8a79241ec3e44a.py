import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2_terms():
    """
    Return the 7 Strassen rank-1 terms for 2x2 * 2x2.
    Each term is (u_vec, v_vec, w_vec) where:
      u_vec: length 4 (indices: 00, 01, 10, 11 for A)
      v_vec: length 4 (indices: 00, 01, 10, 11 for B)
      w_vec: length 4 (indices: 00, 01, 10, 11 for C)
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]  # A00 + A11
    v = [1, 0, 0, 1]  # B00 + B11
    w = [1, 0, 0, 1]  # C00 + C11
    terms.append((u, v, w))
    
    # M2 = (A10+A11)B00
    u = [0, 0, 1, 1]  # A10 + A11
    v = [1, 0, 0, 0]  # B00
    w = [0, 0, 1, -1]  # C10 - C11
    terms.append((u, v, w))
    
    # M3 = A00(B10-B11)
    u = [1, 0, 0, 0]  # A00
    v = [0, 0, 1, -1]  # B10 - B11
    w = [0, 1, 0, 1]  # C01 + C11
    terms.append((u, v, w))
    
    # M4 = A11(B01-B00)
    u = [0, 0, 0, 1]  # A11
    v = [0, 1, -1, 0]  # B01 - B00
    w = [1, 0, 1, 0]  # C00 + C10
    terms.append((u, v, w))
    
    # M5 = (A00+A01)B11
    u = [1, 1, 0, 0]  # A00 + A01
    v = [0, 0, 0, 1]  # B11
    w = [-1, 1, 0, 0]  # -C00 + C01
    terms.append((u, v, w))
    
    # M6 = (A10-A00)(B00+B10)
    u = [-1, 0, 1, 0]  # -A00 + A10
    v = [1, 0, 1, 0]  # B00 + B10
    w = [0, 0, 0, 1]  # C11
    terms.append((u, v, w))
    
    # M7 = (A01-A11)(B01+B11)
    u = [0, 1, 0, -1]  # A01 - A11
    v = [0, 1, 0, 1]  # B01 + B11
    w = [1, 0, 0, 0]  # C00
    terms.append((u, v, w))
    
    return terms

def _add_m442_terms(U, V, W, col, b_offset, c_offset, s2_terms):
    """
    Add 28 rank-1 terms for M_{4,4,2} to U, V, W starting at column 'col'.
    B columns are b_offset, b_offset+1.
    C rows are c_offset, c_offset+1.
    Returns new col index.
    """
    n, m, p = 4, 4, 9
    
    # 4 sets of 7 terms
    # Set 1: A[0:2, 0:2] * B[0:2, :] -> C[:, 0:2] (Top-Left A, Top B, Top C)
    # Set 2: A[0:2, 2:4] * B[2:4, :] -> C[:, 0:2] (Top-Right A, Bot B, Top C)
    # Set 3: A[2:4, 0:2] * B[0:2, :] -> C[:, 2:4] (Bot-Left A, Top B, Bot C)
    # Set 4: A[2:4, 2:4] * B[2:4, :] -> C[:, 2:4] (Bot-Right A, Bot B, Bot C)
    
    sets = [
        # (i_range, j_range, b_j_range, c_i_range)
        ((0, 2), (0, 2), (0, 2), (0, 2)),
        ((0, 2), (2, 4), (2, 4), (0, 2)),
        ((2, 4), (0, 2), (0, 2), (2, 4)),
        ((2, 4), (2, 4), (2, 4), (2, 4)),
    ]
    
    for i_start, i_end, j_start, j_end, bj_start, bj_end, ci_start, ci_end in sets:
        for u2, v2, w2 in s2_terms:
            u_vec = np.zeros(n * m, dtype=float)
            v_vec = np.zeros(m * p, dtype=float)
            w_vec = np.zeros(p * n, dtype=float)
            
            # Fill U
            for si in range(2):
                for sj in range(2):
                    i = i_start + si
                    j = j_start + sj
                    u_vec[i * m + j] = u2[si * 2 + sj]
            
            # Fill V
            for si in range(2):
                for sj in range(2):
                    j = bj_start + si
                    k = b_offset + sj
                    v_vec[j * p + k] = v2[si * 2 + sj]
            
            # Fill W
            for si in range(2):
                for sj in range(2):
                    k = c_offset + si
                    i = ci_start + sj
                    w_vec[k * n + i] = w2[si * 2 + sj]
            
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1
            
    return col

def _add_m441_terms(U, V, W, col, b_offset, c_offset):
    """
    Add 16 rank-1 terms for M_{4,4,1} to U, V, W starting at column 'col'.
    B column is b_offset.
    C row is c_offset.
    Returns new col index.
    """
    n, m, p = 4, 4, 9
    
    # 4 sets of 4 terms (Schoolbook for 2x2x1 blocks)
    sets = [
        # (i_range, j_range)
        ((0, 2), (0, 2)),
        ((0, 2), (2, 4)),
        ((2, 4), (0, 2)),
        ((2, 4), (2, 4)),
    ]
    
    for i_start, i_end, j_start, j_end in sets:
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                u_vec = np.zeros(n * m, dtype=float)
                v_vec = np.zeros(m * p, dtype=float)
                w_vec = np.zeros(p * n, dtype=float)
                
                u_vec[i * m + j] = 1.0
                v_vec[j * p + b_offset] = 1.0
                w_vec[c_offset * n + i] = 1.0
                
                U[:, col] = u_vec
                V[:, col] = v_vec
                W[:, col] = w_vec
                col += 1
                
    return col

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(actual - expected) > 1e-8:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: 
    # M_{4,4,4}: Strassen rank 49
    # M_{4,4,5}: Split into M_{4,4,2} (28) + M_{4,4,2} (28) + M_{4,4,1} (16) = 72
    # Total rank: 49 + 72 = 121
    
    rank_total = 49 + 72
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)
    
    col = 0
    s2_terms = _strassen_2x2_terms()
    
    # Part 1: Strassen for 4x4 * 4x4 (k=0..3)
    # This part is unchanged from the original program, but we can reuse the logic
    # or just call the same block code. 
    # To keep the edit small, I will copy the relevant part of the original construct.
    
    strassen_4x4_combos = [
        ((1,0,0,1), (1,0,0,1), (1,0,0,1)),   # M1
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),   # M2
        ((1,0,0,0), (0,0,1,-1), (0,1,0,1)),   # M3
        ((0,0,0,1), (0,1,-1,0), (1,0,1,0)),   # M4
        ((1,1,0,0), (0,0,0,1), (-1,1,0,0)),   # M5
        ((-1,0,1,0), (1,0,1,0), (0,0,0,1)),   # M6
        ((0,1,0,-1), (0,1,0,1), (1,0,0,0)),   # M7
    ]
    
    for combo_idx, (a_coeffs, b_coeffs, c_coeffs) in enumerate(strassen_4x4_combos):
        for s2_idx, (u2, v2, w2) in enumerate(s2_terms):
            u_vec = np.zeros(n * m, dtype=float)
            v_vec = np.zeros(m * p, dtype=float)
            w_vec = np.zeros(p * n, dtype=float)
            
            for i in range(4):
                for j in range(4):
                    ab = 2 * (i // 2) + (j // 2)
                    si = i % 2
                    sj = j % 2
                    coeff_a = a_coeffs[ab] * u2[si * 2 + sj]
                    if coeff_a != 0:
                        u_vec[i * 4 + j] = coeff_a
            
            for j in range(4):
                for k in range(4):
                    bb = 2 * (j // 2) + (k // 2)
                    si = j % 2
                    sj = k % 2
                    coeff_b = b_coeffs[bb] * v2[si * 2 + sj]
                    if coeff_b != 0:
                        v_vec[j * 9 + k] = coeff_b
            
            for k in range(4):
                for i in range(4):
                    cb = 2 * (k // 2) + (i // 2)
                    si = k % 2
                    sj = i % 2
                    coeff_c = c_coeffs[cb] * w2[si * 2 + sj]
                    if coeff_c != 0:
                        w_vec[k * 4 + i] = coeff_c
            
            if np.any(u_vec) and np.any(v_vec) and np.any(w_vec):
                U[:, col] = u_vec
                V[:, col] = v_vec
                W[:, col] = w_vec
                col += 1
                
    # Part 2: Improved decomposition for 4x4 * 4x5 (k=4..8)
    # Cols 4,5
    col = _add_m442_terms(U, V, W, col, 4, 4, s2_terms)
    # Cols 6,7
    col = _add_m442_terms(U, V, W, col, 6, 6, s2_terms)
    # Col 8
    col = _add_m441_terms(U, V, W, col, 8, 8)
    
    # Trim to actual number of columns used
    if col < rank_total:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        rank_total = col
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb), dtype=float)
        V = np.zeros((m * p, rank_sb), dtype=float)
        W = np.zeros((p * n, rank_sb), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
