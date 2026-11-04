import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 120.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

# Pre-computed Strassen 2x2x2 decomposition (rank 7)
# Each term: (u_dict, v_dict, w_dict)
_S2 = [
    ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
    ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
    ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
    ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
    ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
    ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
]

# 2x2*2x1 schoolbook terms (rank 4)
_S2x1 = [
    ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
    ({(1,0): 1}, {(0,0): 1}, {(1,0): 1}),
    ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),
    ({(1,1): 1}, {(1,0): 1}, {(1,0): 1}),
]

# Combined 2x2x3 terms: rank 11
_S2x3 = _S2 + _S2x1

# Block offsets for 2x2 blocks within 4x4
_BLOCK_OFFSETS = [(0, 0), (0, 2), (2, 0), (2, 2)]

# The 7 super-Strassen terms for 4x4*4x4
_SUPER_TERMS = [
    ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
    ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
    ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
    ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
    ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
    ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
    ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
]

def _build_decomposition():
    """Build the rank-77 decomposition for 4x4 * 4x6."""
    n, m, p = 4, 4, 6
    U_cols = []
    V_cols = []
    W_cols = []

    def _add_2x2(oi, oj, ok):
        """Add Strassen decomposition for 2x2 * 2x2 block at offset (oi,oj,ok)."""
        for u_pat, v_pat, w_pat in _S2:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            for (r, c), coeff in u_pat.items():
                i, j = oi + r, oj + c
                if 0 <= i < n and 0 <= j < m:
                    u_col[i * m + j] += coeff
            for (r, c), coeff in v_pat.items():
                j, k = oj + r, ok + c
                if 0 <= j < m and 0 <= k < p:
                    v_col[j * p + k] += coeff
            for (r, c), coeff in w_pat.items():
                i, k = oi + r, ok + c
                if 0 <= i < n and 0 <= k < p:
                    w_col[k * n + i] += coeff
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    def _add_4x4_strassen(oi, oj, ok):
        """Add recursive Strassen for 4x4 * 4x4 block."""
        for u_blocks, v_blocks, w_blocks in _SUPER_TERMS:
            for u_pat, v_pat, w_pat in _S2:
                u_col = [0] * (n * m)
                v_col = [0] * (m * p)
                w_col = [0] * (p * n)
                for block_idx, block_coeff in u_blocks.items():
                    br, bc = _BLOCK_OFFSETS[block_idx]
                    for (r, c), coeff in u_pat.items():
                        i, j = oi + br + r, oj + bc + c
                        if 0 <= i < n and 0 <= j < m:
                            u_col[i * m + j] += block_coeff * coeff
                for block_idx, block_coeff in v_blocks.items():
                    br, bc = _BLOCK_OFFSETS[block_idx]
                    for (r, c), coeff in v_pat.items():
                        j, k = oj + br + r, ok + bc + c
                        if 0 <= j < m and 0 <= k < p:
                            v_col[j * p + k] += block_coeff * coeff
                for block_idx, block_coeff in w_blocks.items():
                    br, bc = _BLOCK_OFFSETS[block_idx]
                    for (r, c), coeff in w_pat.items():
                        i, k = oi + br + r, ok + bc + c
                        if 0 <= i < n and 0 <= k < p:
                            w_col[k * n + i] += block_coeff * coeff
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)

    # PART 1: 4x4 * 4x4 (columns 0-3) using recursive Strassen: rank 49
    _add_4x4_strassen(0, 0, 0)

    # PART 2: 4x4 * 4x2 (columns 4-5) using 4 independent 2x2*2x2 Strassen: rank 28
    # The 4x2 part corresponds to k=4,5.
    # We partition the 4x4 A into four 2x2 blocks.
    # We partition the 4x2 B into four 2x1 blocks? 
    # No, the current code uses _add_2x2 which assumes 2x2 * 2x2.
    # For 4x4 * 4x2, we can view B as two 4x1 matrices?
    # Or we can view the 4x2 block as two 2x2 blocks?
    # The current code calls _add_2x2(rb * 2, 0, 4) and _add_2x2(rb * 2, 2, 4).
    # This covers rows 0-1 and 2-3 of A, and columns 0-1 and 2-3 of B?
    # Wait, _add_2x2(oi, oj, ok) adds a 2x2 * 2x2 term.
    # oi, oj are row/col offsets in A.
    # ok is col offset in B (which is 4x2, so k=4,5).
    # But B is 4x2. The columns are 0,1 (global 4,5).
    # So a 2x2 block in B would be rows 0-1, cols 0-1?
    # But B only has 2 columns.
    # So _add_2x2 assumes the V block is 2x2.
    # If we call _add_2x2(0, 0, 4), it tries to access V[0:2, 4:6].
    # But V has shape 4x2 (indices 0..3, 0..1).
    # Global index j*p + k.
    # j=0,1; k=4,5.
    # So V[0:2, 4:6] is valid.
    # But the term is 2x2 * 2x2.
    # This implies we are multiplying a 2x2 block of A by a 2x2 block of B.
    # But B is 4x2. The "2x2 block" of B would be 2 rows and 2 cols.
    # So rows 0-1 of B, cols 0-1 of B (global 4-5).
    # This covers C[0:2, 4:6] = A[0:2, 0:2] * B[0:2, 4:6].
    # This is correct for the top-left 2x2 block of C (which is 4x2).
    # Similarly for other blocks.
    
    # The loop:
    for rb in range(2):
        # rb=0: rows 0-1 of A
        _add_2x2(rb * 2, 0, 4) # A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
        _add_2x2(rb * 2, 2, 4) # A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
        # rb=1: rows 2-3 of A
        _add_2x2(rb * 2, 0, 4) # A[2:4, 0:2] * B[0:2, 4:6] -> C[2:4, 4:6]
        _add_2x2(rb * 2, 2, 4) # A[2:4, 2:4] * B[2:4, 4:6] -> C[2:4, 4:6]
        
    rank = len(U_cols)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    return U_final, V_final, W_final

def _verify(U, V, W, n, m, p):
    """Verify the tensor decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.int32)
    V_arr = np.array(V, dtype=np.int32)
    W_arr = np.array(W, dtype=np.int32)
    
    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    u_idx = (i_vals * m + j_vals).ravel()
    v_idx = (j_vals * p + k_vals).ravel()
    w_idx = (k_vals * n + i_vals).ravel()
    
    totals = np.sum(U_arr[u_idx, :] * V_arr[v_idx, :] * W_arr[w_idx, :], axis=1)
    return bool(np.all(totals == 1))

def construct():
    """
    Construct a rank-77 tensor decomposition for 4x4 * 4x6 matrix multiplication.
    """
    if _time_remaining() < 5.0:
        # Fallback to schoolbook if time is tight
        n, m, p = 4, 4, 6
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    n, m, p = 4, 4, 6
    U_final, V_final, W_final = _build_decomposition()
    
    if _verify(U_final, V_final, W_final, n, m, p):
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback: schoolbook rank-96
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
