import time
import numpy as np

DIMENSIONS = [4, 4, 9]

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
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1]
    terms.append((u, v, w))
    
    # M2 = (A10+A11)B00
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, -1]
    terms.append((u, v, w))
    
    # M3 = A00(B10-B11)
    u = [1, 0, 0, 0]
    v = [0, 0, 1, -1]
    w = [0, 1, 0, 1]
    terms.append((u, v, w))
    
    # M4 = A11(B01-B00)
    u = [0, 0, 0, 1]
    v = [0, 1, -1, 0]
    w = [1, 0, 1, 0]
    terms.append((u, v, w))
    
    # M5 = (A00+A01)B11
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    w = [-1, 1, 0, 0]
    terms.append((u, v, w))
    
    # M6 = (A10-A00)(B00+B10)
    u = [-1, 0, 1, 0]
    v = [1, 0, 1, 0]
    w = [0, 0, 0, 1]
    terms.append((u, v, w))
    
    # M7 = (A01-A11)(B01+B11)
    u = [0, 1, 0, -1]
    v = [0, 1, 0, 1]
    w = [1, 0, 0, 0]
    terms.append((u, v, w))
    
    return terms

def _build_strassen_decomposition(n, m, p, s2_terms):
    """
    Build Strassen-based decomposition for the 4x4x4 part.
    Returns (U, V, W) arrays for the 4x4x4 sub-problem.
    """
    # Strassen combinations at 4x4 level:
    # Each is (A_coeffs, B_coeffs, C_coeffs) for blocks 00,01,10,11
    # Block indices: 00=(0,0), 01=(0,1), 10=(1,0), 11=(1,1)
    strassen_4x4_combos = [
        ((1,0,0,1), (1,0,0,1), (1,0,0,1)),   # M1
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),   # M2
        ((1,0,0,0), (0,0,1,-1), (0,1,0,1)),   # M3
        ((0,0,0,1), (0,1,-1,0), (1,0,1,0)),   # M4
        ((1,1,0,0), (0,0,0,1), (-1,1,0,0)),   # M5
        ((-1,0,1,0), (1,0,1,0), (0,0,0,1)),   # M6
        ((0,1,0,-1), (0,1,0,1), (1,0,0,0)),   # M7
    ]
    
    # Build Strassen decomposition for 4x4 * 4x4
    # 7 * 7 = 49 terms
    rank_strassen = 49
    U = np.zeros((n * m, rank_strassen), dtype=float)
    V = np.zeros((m * p, rank_strassen), dtype=float)
    W = np.zeros((p * n, rank_strassen), dtype=float)
    
    col = 0
    for combo_idx, (a_coeffs, b_coeffs, c_coeffs) in enumerate(strassen_4x4_combos):
        for s2_idx, (u2, v2, w2) in enumerate(s2_terms):
            u_vec = np.zeros(n * m, dtype=float)
            v_vec = np.zeros(m * p, dtype=float)
            w_vec = np.zeros(p * n, dtype=float)
            
            # Fill u_vec: A is 4x4, flat index i*4+j
            # Block (bi, bj) where bi = i//2, bj = j//2, block_idx = 2*bi + bj
            for i in range(4):
                for j in range(4):
                    bi = i // 2
                    bj = j // 2
                    block_idx = 2 * bi + bj
                    si = i % 2
                    sj = j % 2
                    coeff_a = a_coeffs[block_idx] * u2[si * 2 + sj]
                    if coeff_a != 0:
                        u_vec[i * 4 + j] = coeff_a
            
            # Fill v_vec: B is 4x9, flat index j*9+k
            # For the 4x4x4 part, k is 0..3
            # Block (bj, bk) where bj = j//2, bk = k//2, block_idx = 2*bj + bk
            for j in range(4):
                for k in range(4):
                    bj = j // 2
                    bk = k // 2
                    block_idx = 2 * bj + bk
                    si = j % 2
                    sj = k % 2
                    coeff_b = b_coeffs[block_idx] * v2[si * 2 + sj]
                    if coeff_b != 0:
                        v_vec[j * 9 + k] = coeff_b
            
            # Fill w_vec: C is 9x4, flat index k*4+i
            # For the 4x4x4 part, k is 0..3
            # Block (bk, bi) where bk = k//2, bi = i//2, block_idx = 2*bk + bi
            for k in range(4):
                for i in range(4):
                    bk = k // 2
                    bi = i // 2
                    block_idx = 2 * bk + bi
                    si = k % 2
                    sj = i % 2
                    coeff_c = c_coeffs[block_idx] * w2[si * 2 + sj]
                    if coeff_c != 0:
                        w_vec[k * 4 + i] = coeff_c
            
            if np.any(u_vec) and np.any(v_vec) and np.any(w_vec):
                U[:, col] = u_vec
                V[:, col] = v_vec
                W[:, col] = w_vec
                col += 1
    
    # Trim if needed
    if col < rank_strassen:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
    
    return U, V, W

def _build_schoolbook_decomposition(n, m, p, k_start, k_end):
    """
    Build schoolbook decomposition for columns k_start..k_end-1 of B.
    Returns (U, V, W) arrays.
    """
    rank = n * m * (k_end - k_start)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(k_start, k_end):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Use Strassen recursively for 4x4x4 (49 terms) + schoolbook for 4x4x5 (80 terms)
    # Total rank: 129
    
    s2_terms = _strassen_2x2_terms()
    
    # Part 1: Strassen for 4x4x4 (k=0..3)
    U1, V1, W1 = _build_strassen_decomposition(n, m, p, s2_terms)
    
    # Part 2: Schoolbook for 4x4x5 (k=4..8)
    U2, V2, W2 = _build_schoolbook_decomposition(n, m, p, 4, 9)
    
    # Combine
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to pure schoolbook
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
    
    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
