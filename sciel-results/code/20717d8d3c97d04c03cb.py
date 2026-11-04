import time
import sys

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2_UVW():
    """
    Return Strassen's 7-term decomposition for 2x2 x 2x2 -> 2x2 matrix multiplication.
    
    U: shape (4, 7) for A[0,0], A[0,1], A[1,0], A[1,1]
    V: shape (4, 7) for B[0,0], B[0,1], B[1,0], B[1,1]
    W: shape (4, 7) for C[0,0], C[0,1], C[1,0], C[1,1]
    
    C[i,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    """
    # Strassen's 7 products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    U = [
        [1, 1, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 1, 0, 0, -1],  # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, 1],   # a11
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [1, 0, -1, 0, 1, 0, 1],  # b10
        [1, 0, 0, 0, 1, 0, 1],   # b11
    ]
    W = [
        [1, 0, 0, 1],   # C00 = M1 - M4 + M6
        [0, 1, 1, 0],   # C01 = M3 + M5
        [0, 1, 0, 1],   # C10 = M2 + M4
        [1, 0, -1, 1],  # C11 = M1 - M3 + M7
    ]
    return U, V, W

def _block_4x4x4_UVW():
    """
    Build rank-97 decomposition for 4x4x7:
    - 4x4x4 part using recursive Strassen (rank 49)
    - 4x4x3 part using trivial (rank 48)
    """
    import numpy as np
    
    # Get the 2x2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2_2x2_UVW()
    U2 = np.array(U2, dtype=float)
    V2 = np.array(V2, dtype=float)
    W2 = np.array(W2, dtype=float)
    
    # Build 4x4x4 decomposition using recursive Strassen (rank 49)
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Strassen block products:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 (+1), C11 (+1)
    # M2 = (A10+A11)B00       -> contributes to C10 (+1)
    # M3 = A00(B01-B11)       -> contributes to C01 (+1), C11 (-1)
    # M4 = A11(B10-B00)       -> contributes to C00 (-1), C10 (+1)
    # M5 = (A00+A01)B11       -> contributes to C01 (+1)
    # M6 = (A10-A00)(B00+B01) -> contributes to C00 (+1)
    # M7 = (A01-A11)(B10+B11) -> contributes to C11 (+1)
    
    rank_444 = 49
    U444 = np.zeros((16, rank_444), dtype=float)
    V444 = np.zeros((16, rank_444), dtype=float)
    W444 = np.zeros((16, rank_444), dtype=float)
    
    # Block product definitions: (a_blocks, b_blocks, c_contrib)
    # Each block is a 2x2 submatrix. Block index: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    block_products = [
        # M1 = (A00+A11)(B00+B11)
        {'a': [(0, 1), (3, 1)], 'b': [(0, 1), (3, 1)], 'c': [(0, 1), (3, 1)]},
        # M2 = (A10+A11)B00
        {'a': [(2, 1), (3, 1)], 'b': [(0, 1)], 'c': [(2, 1)]},
        # M3 = A00(B01-B11)
        {'a': [(0, 1)], 'b': [(1, 1), (3, -1)], 'c': [(1, 1), (3, -1)]},
        # M4 = A11(B10-B00)
        {'a': [(3, 1)], 'b': [(2, 1), (0, -1)], 'c': [(0, -1), (2, 1)]},
        # M5 = (A00+A01)B11
        {'a': [(0, 1), (1, 1)], 'b': [(3, 1)], 'c': [(1, 1)]},
        # M6 = (A10-A00)(B00+B01)
        {'a': [(2, 1), (0, -1)], 'b': [(0, 1), (1, 1)], 'c': [(0, 1)]},
        # M7 = (A01-A11)(B10+B11)
        {'a': [(1, 1), (3, -1)], 'b': [(2, 1), (3, 1)], 'c': [(3, 1)]},
    ]
    
    for bp_idx, bp in enumerate(block_products):
        for s_idx in range(7):
            r = bp_idx * 7 + s_idx
            
            # U2[s_idx] gives coefficients for the 4 entries of a 2x2 A-block
            # V2[s_idx] gives coefficients for the 4 entries of a 2x2 B-block
            # W2[s_idx] gives coefficients for the 4 entries of a 2x2 C-block
            
            for a_blk, a_coeff in bp['a']:
                for li in range(2):
                    for lj in range(2):
                        gi = a_blk * 2 + li  # global row within 4x4
                        gj = lj               # global col within 4x4 (for A)
                        u_row = gi * 4 + gj
                        local_idx = li * 2 + lj
                        U444[u_row, r] += a_coeff * U2[s_idx, local_idx]
            
            for b_blk, b_coeff in bp['b']:
                for lj in range(2):
                    for lk in range(2):
                        gj = b_blk * 2 + lj  # global row within 4x4 (for B)
                        gk = lk               # global col within 4x4
                        v_row = gj * 4 + gk
                        local_idx = lj * 2 + lk
                        V444[v_row, r] += b_coeff * V2[s_idx, local_idx]
            
            for c_blk, c_coeff in bp['c']:
                for li in range(2):
                    for lk in range(2):
                        gi = c_blk * 2 + li  # global row within 4x4
                        gk = lk               # global col within 4x4
                        w_row = gk * 4 + gi
                        local_idx = li * 2 + lk
                        W444[w_row, r] += c_coeff * W2[s_idx, local_idx]
    
    # Build 4x4x3 trivial decomposition (rank 48)
    rank_443 = 48
    U443 = np.zeros((16, rank_443), dtype=float)
    V443 = np.zeros((12, rank_443), dtype=float)  # 4x3 = 12
    W443 = np.zeros((12, rank_443), dtype=float)  # 3x4 = 12
    
    t = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u_idx = i * 4 + j
                v_idx = j * 3 + k
                w_idx = k * 4 + i
                U443[u_idx, t] = 1.0
                V443[v_idx, t] = 1.0
                W443[w_idx, t] = 1.0
                t += 1
    
    # Combine: total rank = 49 + 48 = 97
    total_rank = rank_444 + rank_443
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((28, total_rank), dtype=float)  # 4x7 = 28
    W = np.zeros((28, total_rank), dtype=float)  # 7x4 = 28
    
    # Copy 4x4x4 part (columns 0-3 of B, rows 0-3 of C)
    U[:, :rank_444] = U444
    
    for j in range(4):
        for k in range(4):
            v_idx = j * 7 + k
            V[v_idx, :rank_444] = V444[j * 4 + k, :]
    
    for k in range(4):
        for i in range(4):
            w_idx = k * 4 + i
            W[w_idx, :rank_444] = W444[k * 4 + i, :]
    
    # Copy 4x4x3 part (columns 4-6 of B, rows 4-6 of C)
    offset = rank_444
    U[:, offset:offset+rank_443] = U443
    
    for j in range(4):
        for k in range(3):
            v_idx = j * 7 + (4 + k)
            V[v_idx, offset:offset+rank_443] = V443[j * 3 + k, :]
    
    for k in range(3):
        for i in range(4):
            w_idx = (4 + k) * 4 + i
            W[w_idx, offset:offset+rank_443] = W443[k * 4 + i, :]
    
    return U.tolist(), V.tolist(), W.tolist()

def _trivial_decomposition(n, m, p):
    """
    Return the trivial (rank n*m*p) exact tensor decomposition for n x m x p matrix multiplication.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """
    Verify that the decomposition (U, V, W) correctly computes matrix multiplication.
    Uses NumPy vectorization for speed.
    """
    try:
        import numpy as np
        np.random.seed(12345)
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        for trial in range(2):
            A = np.random.randint(-5, 6, (n, m), dtype=float)
            B = np.random.randint(-5, 6, (m, p), dtype=float)
            C_expected = A @ B
            
            # Compute C_computed[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_computed = np.zeros((n, p), dtype=float)
            for i in range(n):
                for k in range(p):
                    w_row = W_arr[k * n + i, :]
                    s = 0.0
                    for j in range(m):
                        u_row = U_arr[i * m + j, :]
                        v_row = V_arr[j * p + k, :]
                        s += np.dot(u_row * v_row, w_row)
                    C_computed[i, k] = s
            
            if not np.allclose(C_computed, C_expected, atol=1e-8):
                return False
        
        return True
    except Exception:
        return False

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 7
    
    # Try the hierarchical decomposition (rank 97)
    try:
        if time.time() < deadline - 30:
            U, V, W = _block_4x4x4_UVW()
            
            # Verify
            if time.time() < deadline - 20:
                if _verify_decomposition(n, m, p, U, V, W):
                    return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to trivial decomposition (rank 112)
    U, V, W = _trivial_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
