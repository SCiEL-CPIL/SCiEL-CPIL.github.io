# EVOLVE-BLOCK-START

import time
import numpy as np

def _build_strassen_28():
    """
    Constructs the standard rank-28 Strassen decomposition for 2x4 * 4x4.
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms for 2x2 matrices X, Y -> Z
    # Indices: 0:00, 1:01, 2:10, 3:11
    # M1 = (x0+x3)(y0+y3)
    # M2 = (x2+x3)y0
    # M3 = x0(y1-y3)
    # M4 = x3(y2-y0)
    # M5 = (x0+x1)y3
    # M6 = (x2-x0)(y0+y1)
    # M7 = (x1-x3)(y2+y3)
    # Z0 = M1 + M4 - M5 + M7
    # Z1 = M3 + M5
    # Z2 = M2 + M4
    # Z3 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Define mapping functions
    def get_u_idx(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        gc = c if side == 'L' else 2 + c
        return r * m + gc

    def get_v_idx(block, local_idx):
        r_off, c_off = {
            'TL': (0, 0), 'TR': (0, 2), 'BL': (2, 0), 'BR': (2, 2)
        }[block]
        r = local_idx // 2
        c = local_idx % 2
        gr = r_off + r
        gc = c_off + c
        return gr * p + gc

    def get_w_idx(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        gc = c if side == 'L' else 2 + c
        return gc * n + r

    blocks = [
        ('L', 'TL', 'L'),
        ('R', 'BL', 'L'),
        ('L', 'TR', 'R'),
        ('R', 'BR', 'R'),
    ]
    
    col = 0
    for a_side, b_block, c_side in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for local_idx in range(4):
                u_idx = get_u_idx(a_side, local_idx)
                v_idx = get_v_idx(b_block, local_idx)
                w_idx = get_w_idx(c_side, local_idx)
                
                U[u_idx, col] += u_coeffs[local_idx]
                V[v_idx, col] += v_coeffs[local_idx]
                W[w_idx, col] += w_coeffs[local_idx]
            col += 1
            
    return U, V, W

def _build_naive_32():
    """
    Constructs the naive rank-32 decomposition.
    """
    n, m, p = 2, 4, 4
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
                
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify exact decomposition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    
    # Try Rank 28
    U28, V28, W28 = _build_strassen_28()
    if _verify_decomposition(U28, V28, W28):
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
        
    # Fallback to Rank 32
    U32, V32, W32 = _build_naive_32()
    if _verify_decomposition(U32, V32, W32):
        return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
        
    # Should never happen
    return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}

# EVOLVE-BLOCK-END
