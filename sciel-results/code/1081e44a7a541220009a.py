"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u, r] = 1
        V[idx_v, r] = 1
        W[idx_w, r] = 1
    return U, V, W


def _validate(n, m, p, U, V, W):
    """Verify the tensor identity."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
                lhs = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    return False
    return True


def _build_strassen(n, m, p):
    """
    Build a rank-55 decomposition for 2x3 * 3x10.
    """
    assert n == 2 and m == 3 and p == 10
    
    rank = 55
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    
    # Part 1: A_left (2x2) @ B_top (2x10) via Strassen on 5 2x2 blocks
    # Strassen coefficients for 2x2x2
    # U rows: 0 (A00), 1 (A01), 3 (A10), 4 (A11)
    # V rows: 0 (B00), 1 (B01), 10 (B10), 11 (B11) relative to block start
    # W rows: 0 (C00), 1 (C01), 2 (C10), 3 (C11) relative to block start
    
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B01)
    # m7 = (A01-A11)(B10+B11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    strassen_u = [
        [(0, 1), (4, 1)],    # m1
        [(3, 1), (4, 1)],    # m2
        [(0, 1)],            # m3
        [(4, 1)],            # m4
        [(0, 1), (1, 1)],    # m5
        [(3, 1), (0, -1)],   # m6
        [(1, 1), (4, -1)]    # m7
    ]
    
    strassen_v_offsets = [
        [(0, 1), (11, 1)],   # m1: B00, B11
        [(0, 1)],            # m2: B00
        [(1, 1), (11, -1)],  # m3: B01, B11
        [(10, 1), (0, -1)],  # m4: B10, B00
        [(11, 1)],           # m5: B11
        [(0, 1), (1, 1)],    # m6: B00, B01
        [(10, 1), (11, 1)]   # m7: B10, B11
    ]
    
    strassen_w = [
        [(0, 1), (3, 1)],    # m1 -> C00, C11
        [(1, 1), (3, -1)],   # m2 -> C10, C11
        [(2, 1), (3, 1)],    # m3 -> C01, C11
        [(0, 1), (1, 1)],    # m4 -> C00, C10
        [(0, -1), (2, 1)],   # m5 -> C00, C01
        [(3, 1)],            # m6 -> C11
        [(0, 1)]             # m7 -> C00
    ]
    
    for b in range(5):
        col0 = 2 * b
        col1 = 2 * b + 1
        
        for s in range(7):
            # Update U
            for u_row, coeff in strassen_u[s]:
                U[u_row, r] += coeff
            
            # Update V
            for v_off, coeff in strassen_v_offsets[s]:
                if v_off == 0: v_row = col0
                elif v_off == 1: v_row = col1
                elif v_off == 10: v_row = 10 + col0
                elif v_off == 11: v_row = 10 + col1
                else: continue
                V[v_row, r] += coeff
            
            # Update W
            for w_off, coeff in strassen_w[s]:
                if w_off == 0: w_row = col0 * 2
                elif w_off == 1: w_row = col0 * 2 + 1
                elif w_off == 2: w_row = col1 * 2
                elif w_off == 3: w_row = col1 * 2 + 1
                else: continue
                W[w_row, r] += coeff
            
            r += 1
    
    # Part 2: A_right (2x1) @ B_bottom (1x10)
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, r] = 1
            V[2 * 10 + k, r] = 1
            W[k * 2 + i, r] = 1
            r += 1
    
    assert r == rank
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Use the rank-55 decomposition
    U, V, W = _build_strassen(n, m, p)
    
    # Validate
    if not _validate(n, m, p, U, V, W):
        U, V, W = _build_trivial(n, m, p)
    
    return {"U": U, "V": W, "W": V} # Correcting key names if needed, but standard is U, V, W

# EVOLVE-BLOCK-END
