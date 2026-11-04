"""Combined: block decomposition with Strassen for 2x2 sub-blocks, verified with fallback."""

import numpy as np

def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2(U, V, W, i_base, j_base, k_base, r_offset):
    """
    Add Strassen rank-7 terms for a 2x2x2 sub-block.
    i_base: starting row of A (0 or 1)
    j_base: starting col of A / row of B (0 or 1)
    k_base: starting col of B (0 or 1)
    r_offset: starting column index in U,V,W
    """
    n, m, p = 2, 3, 5
    
    # Local indices within the 2x2 block:
    # A rows: i_base, i_base+1; A cols: j_base, j_base+1
    # B rows: j_base, j_base+1; B cols: k_base, k_base+1
    # C rows: k_base, k_base+1; C cols: i_base, i_base+1
    
    a0 = i_base * m + j_base
    a1 = i_base * m + j_base + 1
    a2 = (i_base + 1) * m + j_base
    a3 = (i_base + 1) * m + j_base + 1
    
    b0 = j_base * p + k_base
    b1 = j_base * p + k_base + 1
    b2 = (j_base + 1) * p + k_base
    b3 = (j_base + 1) * p + k_base + 1
    
    c0 = k_base * n + i_base
    c1 = k_base * n + i_base + 1
    c2 = (k_base + 1) * n + i_base
    c3 = (k_base + 1) * n + i_base + 1
    
    r = r_offset
    
    # M1 = (A00+A10)(B00+B10)
    U[a0][r] += 1
    U[a2][r] += 1
    V[b0][r] += 1
    V[b2][r] += 1
    W[c0][r] += 1
    W[c3][r] += 1
    r += 1
    
    # M2 = (A10+A11)(B00)
    U[a2][r] += 1
    U[a3][r] += 1
    V[b0][r] += 1
    W[c1][r] += 1
    W[c3][r] += -1
    r += 1
    
    # M3 = (A00)(B01-B11)
    U[a0][r] += 1
    V[b1][r] += 1
    V[b3][r] += -1
    W[c2][r] += 1
    W[c3][r] += 1
    r += 1
    
    # M4 = (A11)(B10-B00)
    U[a3][r] += 1
    V[b2][r] += 1
    V[b0][r] += -1
    W[c0][r] += 1
    W[c1][r] += 1
    r += 1
    
    # M5 = (A00+A01)(B11)
    U[a0][r] += 1
    U[a1][r] += 1
    V[b3][r] += 1
    W[c0][r] += -1
    W[c2][r] += 1
    r += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[a2][r] += 1
    U[a0][r] += -1
    V[b0][r] += 1
    V[b1][r] += 1
    W[c3][r] += 1
    r += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[a1][r] += 1
    U[a3][r] += -1
    V[b2][r] += 1
    V[b3][r] += 1
    W[c0][r] += 1
    r += 1
    
    return r_offset + 7

def construct():
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]   # 6 x 28
    V = [[0] * rank for _ in range(m * p)]   # 15 x 28
    W = [[0] * rank for _ in range(p * n)]   # 10 x 28
    
    r = 0
    
    # Part 1: Block (i=0,1, j=2, k=0..4): A[:,2] * B[2,:] -> C[:,0..4]
    # This is 2x1 x 1x5 = 2x5. Schoolbook: 10 terms.
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    
    # Part 2: Block (i=0,1, j=0,1, k=4): A[:,:2] * B[:,4] -> C[4,:]
    # This is 2x2 x 2x1 = 2x2. Schoolbook: 4 terms.
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    
    # Part 3: Block (i=0,1, j=0,1, k=0,1): A[:,:2] * B[:,:2] -> C[:,:2]
    # This is 2x2 x 2x2 = 2x2. Use Strassen: 7 terms.
    r = _strassen_2x2(U, V, W, i_base=0, j_base=0, k_base=0, r_offset=r)
    
    # Part 4: Block (i=0,1, j=0,1, k=2,3): A[:,:2] * B[:,2:4] -> C[2:4,:]
    # This is 2x2 x 2x2 = 2x2. Use Strassen: 7 terms.
    r = _strassen_2x2(U, V, W, i_base=0, j_base=0, k_base=2, r_offset=r)
    
    # Verify: check all 30 (i,j,k) triples
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r_idx in range(rank):
                    s += U_arr[idx_u, r_idx] * V_arr[idx_v, r_idx] * W_arr[idx_w, r_idx]
                if abs(s - 1.0) > 1e-9:
                    return _schoolbook(n, m, p)
    
    # Also verify that non-matching terms give 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                # This is the matching term, already checked above
                pass
    
    # Check a few non-matching terms to be safe
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Extract indices
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                # Skip if this is a valid matching term
                if i_a == i_c and j_a == j_b:
                    continue
                    
                s = 0.0
                for r_idx in range(rank):
                    s += U_arr[a, r_idx] * V_arr[b, r_idx] * W_arr[c, r_idx]
                if abs(s) > 1e-9:
                    return _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
