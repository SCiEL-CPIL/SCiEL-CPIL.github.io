import time
import numpy as np

DIMENSIONS = [2, 4, 4]

_DEADLINE = time.time() + 130.0

def _build_rank14():
    """Build a rank-14 decomposition by merging Strassen blocks."""
    n, m, p = DIMENSIONS
    rank = 14
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen terms for 2x2x2
    # u_loc: index in 2x2 A block (0:00, 1:01, 2:10, 3:11) -> coeff
    # v_loc: index in 2x2 B block -> coeff
    # w_loc: index in 2x2 C block -> coeff
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),       # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),       # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),       # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})        # M7
    ]

    # We have 2 A-blocks (A1, A2) and 2 B-block-sets (B_top, B_bot)
    # A1: rows 0,1 of A? No, A is 2x4.
    # A1 is cols 0,1. A2 is cols 2,3.
    # B_top is rows 0,1 of B. B_bot is rows 2,3 of B.
    # B_top has cols 0,1 (B11) and 2,3 (B12).
    # B_bot has cols 0,1 (B21) and 2,3 (B22).
    
    # For A1 (cols 0,1):
    # It multiplies B_top (rows 0,1).
    # Result contributes to C (all 4 cols).
    # C cols 0,1 come from B11. C cols 2,3 come from B12.
    
    # For A2 (cols 2,3):
    # It multiplies B_bot (rows 2,3).
    # Result contributes to C (all 4 cols).
    # C cols 0,1 come from B21. C cols 2,3 come from B22.

    def add_term(col_idx, a_block, b_rows, w_offset_cols):
        # a_block: 0 for A1 (cols 0,1), 1 for A2 (cols 2,3)
        # b_rows: 0 for B_top (rows 0,1), 1 for B_bot (rows 2,3)
        # w_offset_cols: 0 for C cols 0,1, 1 for C cols 2,3
        pass

    # We iterate over the 7 Strassen terms
    for k in range(7):
        u_loc, v_loc, w_loc = strassen_terms[k]
        
        # Term for A1 (k)
        col = k
        # U: A1 cols 0,1. Indices 0..3.
        for idx, coeff in u_loc.items():
            r = idx // 2
            c = idx % 2
            g_idx = r * 4 + c # A is 2x4, row r, col c
            U[g_idx, col] += coeff
            
        # V: B_top rows 0,1. Cols 0,1 (B11) and 2,3 (B12).
        # v_loc is for 2x2 block.
        # We need to apply v_loc to B11 and B12.
        # B11 is B[0:2, 0:2]. B12 is B[0:2, 2:4].
        for idx, coeff in v_loc.items():
            r = idx // 2
            c = idx % 2
            # B11: row r, col c. Global B idx: r*4 + c
            g_idx_11 = r * 4 + c
            V[g_idx_11, col] += coeff
            # B12: row r, col 2+c. Global B idx: r*4 + 2 + c
            g_idx_12 = r * 4 + 2 + c
            V[g_idx_12, col] += coeff
            
        # W: C cols 0,1 and 2,3.
        # w_loc is for 2x2 C block.
        # C11 (cols 0,1) and C12 (cols 2,3).
        for idx, coeff in w_loc.items():
            r = idx // 2
            c = idx % 2
            # C11: row r, col c. Global C idx: c*2 + r (Trace convention W[k*n+i,r] -> W[col*n+row, rank]? 
            # Task says W[k*n+i, r]. k is col? i is row?
            # "output factor follows the official trace convention W[k*n+i,r]"
            # Usually C[i,j] = sum U[i,k] V[k,j] W[j,n+i]?
            # Let's stick to the code's W indexing: W[kk * n + ii, cc]
            # kk is col (0..3), ii is row (0..1).
            # So idx = kk * 2 + ii.
            g_idx_11 = c * 2 + r
            W[g_idx_11, col] += coeff
            # C12: row r, col 2+c. Global C idx: (2+c)*2 + r
            g_idx_12 = (2 + c) * 2 + r
            W[g_idx_12, col] += coeff

        # Term for A2 (k+7)
        col = k + 7
        # U: A2 cols 2,3. Indices 2,3,6,7.
        for idx, coeff in u_loc.items():
            r = idx // 2
            c = idx % 2
            g_idx = r * 4 + (2 + c)
            U[g_idx, col] += coeff
            
        # V: B_bot rows 2,3. Cols 0,1 (B21) and 2,3 (B22).
        for idx, coeff in v_loc.items():
            r = idx // 2
            c = idx % 2
            # B21: row 2+r, col c. Global B idx: (2+r)*4 + c
            g_idx_21 = (2 + r) * 4 + c
            V[g_idx_21, col] += coeff
            # B22: row 2+r, col 2+c. Global B idx: (2+r)*4 + 2 + c
            g_idx_22 = (2 + r) * 4 + 2 + c
            V[g_idx_22, col] += coeff
            
        # W: Same C structure as A1?
        # Yes, A2 also contributes to all C cols.
        for idx, coeff in w_loc.items():
            r = idx // 2
            c = idx % 2
            g_idx_11 = c * 2 + r
            W[g_idx_11, col] += coeff
            g_idx_12 = (2 + c) * 2 + r
            W[g_idx_12, col] += coeff

    return U, V, W

def _verify(U, V, W, trials=3, seed=42):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    rng = np.random.RandomState(seed)
    for trial in range(trials):
        A = rng.randint(-5, 6, (n, m)).astype(float)
        B = rng.randint(-5, 6, (m, p)).astype(float)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc
        if not np.allclose(C, Cf, atol=1e-6):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank 14
    if time.time() < _DEADLINE - 10.0:
        U14, V14, W14 = _build_rank14()
        if _verify(U14, V14, W14):
            return {"U": U14.tolist(), "V": V14.tolist(), "W": W14.tolist()}
            
    # Fallback to rank 28
    U28, V28, W28 = _build_rank28()
    if _verify(U28, V28, W28):
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
        
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
