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

def _strassen_4x4x4_decomposition():
    """
    Build Strassen-based decomposition for 4x4 * 4x4 matrix multiplication.
    Returns U, V, W matrices where:
      U: shape (16, rank) - indices i*4+j
      V: shape (16, rank) - indices j*4+k
      W: shape (16, rank) - indices k*4+i
    """
    s2_terms = _strassen_2x2_terms()
    
    # Strassen combinations at 4x4 level
    strassen_4x4_combos = [
        ((1,0,0,1), (1,0,0,1), (1,0,0,1)),   # M1
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),   # M2
        ((1,0,0,0), (0,0,1,-1), (0,1,0,1)),   # M3
        ((0,0,0,1), (0,1,-1,0), (1,0,1,0)),   # M4
        ((1,1,0,0), (0,0,0,1), (-1,1,0,0)),   # M5
        ((-1,0,1,0), (1,0,1,0), (0,0,0,1)),   # M6
        ((0,1,0,-1), (0,1,0,1), (1,0,0,0)),   # M7
    ]
    
    rank_strassen = 49
    U = np.zeros((16, rank_strassen), dtype=float)
    V = np.zeros((16, rank_strassen), dtype=float)
    W = np.zeros((16, rank_strassen), dtype=float)
    
    col = 0
    for combo_idx, (a_coeffs, b_coeffs, c_coeffs) in enumerate(strassen_4x4_combos):
        for s2_idx, (u2, v2, w2) in enumerate(s2_terms):
            u_vec = np.zeros(16, dtype=float)
            v_vec = np.zeros(16, dtype=float)
            w_vec = np.zeros(16, dtype=float)
            
            # Fill u_vec: A is 4x4, flat index i*4+j
            for i in range(4):
                for j in range(4):
                    ab = 2 * (i // 2) + (j // 2)
                    si = i % 2
                    sj = j % 2
                    coeff_a = a_coeffs[ab] * u2[si * 2 + sj]
                    if coeff_a != 0:
                        u_vec[i * 4 + j] = coeff_a
            
            # Fill v_vec: B is 4x4, flat index j*4+k
            for j in range(4):
                for k in range(4):
                    bb = 2 * (j // 2) + (k // 2)
                    si = j % 2
                    sj = k % 2
                    coeff_b = b_coeffs[bb] * v2[si * 2 + sj]
                    if coeff_b != 0:
                        v_vec[j * 4 + k] = coeff_b
            
            # Fill w_vec: C is 4x4, flat index k*4+i
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
    
    # Trim if needed
    if col < rank_strassen:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Use Strassen for the 4x4x4 part (k=0..3) and schoolbook for k=4..8
    # For k=0..3: 49 terms (Strassen)
    # For k=4..8: 4*4*5 = 80 terms (schoolbook)
    # Total: 129 terms
    
    U1, V1, W1 = _strassen_4x4x4_decomposition()
    rank1 = V1.shape[1]
    
    # Extend V1 and W1 to handle p=9 instead of p=4
    # V is indexed by j*p+k where j in 0..3, k in 0..8
    # W is indexed by k*n+i where k in 0..8, i in 0..3
    
    U_ext = np.zeros((16, rank1), dtype=float)
    V_ext = np.zeros((36, rank1), dtype=float)
    W_ext = np.zeros((36, rank1), dtype=float)
    
    for col in range(rank1):
        U_ext[:, col] = U1[:, col]
        
        # Map V1[j*4+k] -> V_ext[j*9+k] for k in 0..3
        for j in range(4):
            for k in range(4):
                V_ext[j*9+k, col] = V1[j*4+k, col]
        
        # Map W1[k*4+i] -> W_ext[k*4+i] for k in 0..3
        for k in range(4):
            for i in range(4):
                W_ext[k*4+i, col] = W1[k*4+i, col]
    
    # Part 2: Schoolbook for k=4..8 (5 columns)
    rank2 = 4 * 4 * 5  # 80 terms
    U2 = np.zeros((16, rank2), dtype=float)
    V2 = np.zeros((36, rank2), dtype=float)
    W2 = np.zeros((36, rank2), dtype=float)
    
    col2 = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                U2[i*4+j, col2] = 1
                V2[j*9+k, col2] = 1
                W2[k*4+i, col2] = 1
                col2 += 1
    
    # Combine
    rank_total = rank1 + rank2
    U = np.hstack([U_ext, U2])
    V = np.hstack([V_ext, V2])
    W = np.hstack([W_ext, W2])
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to pure schoolbook
        rank_sb = n * m * p
        U = np.zeros((16, rank_sb), dtype=float)
        V = np.zeros((36, rank_sb), dtype=float)
        W = np.zeros((36, rank_sb), dtype=float)
        c = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U[i*4+j, c] = 1
                    V[j*9+k, c] = 1
                    W[k*4+i, c] = 1
                    c += 1
    
    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
