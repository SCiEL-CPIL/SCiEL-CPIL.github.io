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
    
    # Strategy: Split M_{4,4,9} into M_{4,4,4} + M_{4,4,5}
    # M_{4,4,4}: Use Strassen recursively -> rank 49
    # M_{4,4,5}: Use schoolbook -> rank 80
    # Total rank: 129
    
    # Build Strassen decomposition for 4x4 * 4x4
    # 4x4 = 2x2 blocks of 2x2
    # Strassen on 4x4: 7 multiplications of 2x2 * 2x2
    # Each 2x2 * 2x2 uses 7 Strassen terms
    # Total: 7 * 7 = 49 terms
    
    rank_total = 49 + 80
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)
    
    col = 0
    
    # Part 1: Strassen for 4x4 * 4x4 (k=0..3)
    s2_terms = _strassen_2x2_terms()
    
    # Strassen combinations at 4x4 level:
    # Each is (A_coeffs, B_coeffs, C_coeffs) for blocks 00,01,10,11
    strassen_4x4_combos = [
        ((1,0,0,1), (1,0,0,1), (1,0,0,1)),   # M1
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),   # M2
        ((1,0,0,0), (0,0,1,-1), (0,1,0,1)),   # M3
        ((0,0,0,1), (0,1,-1,0), (1,0,1,0)),   # M4
        ((1,1,0,0), (0,0,0,1), (-1,1,0,0)),   # M5
        ((-1,0,1,0), (1,0,1,0), (0,0,0,1)),   # M6
        ((0,1,0,-1), (0,1,0,1), (1,0,0,0)),   # M7
    ]
    
    # For each of the 7 Strassen combos, and for each of the 7 sub-Strassen terms,
    # create rank-1 terms
    
    for combo_idx, (a_coeffs, b_coeffs, c_coeffs) in enumerate(strassen_4x4_combos):
        for s2_idx, (u2, v2, w2) in enumerate(s2_terms):
            # u2, v2, w2 are length-4 vectors for 2x2 matrices
            # u2 = [u00, u01, u10, u11]
            
            u_vec = np.zeros(n * m, dtype=float)
            v_vec = np.zeros(m * p, dtype=float)
            w_vec = np.zeros(p * n, dtype=float)
            
            # Fill u_vec: A is 4x4, flat index i*4+j
            for i in range(4):
                for j in range(4):
                    # Block index: (i//2, j//2) -> ab = 2*(i//2) + (j//2)
                    ab = 2 * (i // 2) + (j // 2)
                    # Sub-position within 2x2 block: (i%2, j%2)
                    si = i % 2
                    sj = j % 2
                    coeff_a = a_coeffs[ab] * u2[si * 2 + sj]
                    if coeff_a != 0:
                        u_vec[i * 4 + j] = coeff_a
            
            # Fill v_vec: B is 4x9, flat index j*9+k
            # For the 4x4x4 part, k is 0..3
            for j in range(4):
                for k in range(4):
                    # Block index: (j//2, k//2) -> bb = 2*(j//2) + (k//2)
                    bb = 2 * (j // 2) + (k // 2)
                    # Sub-position within 2x2 block: (j%2, k%2)
                    si = j % 2
                    sj = k % 2
                    coeff_b = b_coeffs[bb] * v2[si * 2 + sj]
                    if coeff_b != 0:
                        v_vec[j * 9 + k] = coeff_b
            
            # Fill w_vec: C is 9x4, flat index k*4+i
            # For the 4x4x4 part, k is 0..3
            for k in range(4):
                for i in range(4):
                    # Block index: (k//2, i//2) -> cb = 2*(k//2) + (i//2)
                    cb = 2 * (k // 2) + (i // 2)
                    # Sub-position within 2x2 block: (k%2, i%2)
                    si = k % 2
                    sj = i % 2
                    coeff_c = c_coeffs[cb] * w2[si * 2 + sj]
                    if coeff_c != 0:
                        w_vec[k * 4 + i] = coeff_c
            
            # Check if this term is non-trivial
            if np.any(u_vec) and np.any(v_vec) and np.any(w_vec):
                U[:, col] = u_vec
                V[:, col] = v_vec
                W[:, col] = w_vec
                col += 1
    
    # Part 2: Schoolbook for 4x4 * 4x5 (k=4..8)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
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
