"""4x4x8 matrix multiplication tensor decomposition.

Strategy: Direct recursive Strassen tiling.
Decompose 4x4x8 into 8 blocks of 2x2x4.
Decompose each 2x2x4 into 2 blocks of 2x2x2.
Use rank-7 Strassen for 2x2x2.
Total rank = 8 * 2 * 7 = 112.

This is structurally different from the Kronecker product approach (rank 98)
by treating the rectangular dimension as a sequence of independent square
sub-problems rather than a tensor product factor.
"""

import time
import random

DEADLINE = 120.0
T_START = time.time()


def _remaining():
    return DEADLINE - (time.time() - T_START)


def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matmul.
    
    Each term is (u_dict, v_dict, w_dict).
    Indices:
    U: (i, j) for A[i, j]
    V: (j, k) for B[j, k]
    W: (k, i) for C[k, i]  <-- Note: Output index convention is W[k*n+i, r]
    """
    terms = []
    # M0 = (A00+A11)*(B00+B11)
    terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    # M1 = (A10+A11)*B00
    terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1}))
    # M2 = A00*(B01-B11)
    terms.append(({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1}))
    # M3 = A11*(B10-B00)
    terms.append(({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}))
    # M4 = (A00+A01)*B11
    terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1, (1,1): 1}))
    # M5 = (A10-A00)*(B00+B01)
    terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): -1}))
    # M6 = (A01-A11)*(B10+B11)
    terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1, (1,1): -1}))
    return terms


def _build_4x4x8_direct_tiling():
    """Build 4x4x8 decomposition using direct recursive Strassen tiling.
    
    1. Split 4x4x8 into 8 sub-problems of size 2x2x4.
       A is 4x4, split into 2x2 blocks A[ai, bi] where ai, bi in {0, 1}.
       B is 4x8, split into 2x4 blocks B[bi, ck] where bi in {0, 1}, ck in {0, 1, 2, 3}? 
       No, B is 4x8. Rows are 0..3, Cols are 0..7.
       The shared index is j (col of A, row of B).
       A is 4x4. B is 4x8. C is 4x8.
       
       Let's index by:
       i: row of A, row of C (0..3)
       j: col of A, row of B (0..3)
       k: col of B, col of C (0..7)
       
       Split i into (i0, i1) where i = 2*i0 + i1.
       Split j into (j0, j1) where j = 2*j0 + j1.
       Split k into (k0, k1) where k = 2*k0 + k1.
       
       This gives 2*2*2 = 8 sub-problems of size 2x2x2?
       Wait, if we split k into 2 parts of size 4, then each sub-problem is 2x2x4.
       If we split k into 4 parts of size 2, then each sub-problem is 2x2x2.
       
       Let's split k into 4 parts: k = 4*k0 + k1? No, k is 0..7.
       Let k0 in {0, 1, 2, 3} and k1 in {0, 1}.
       Then k = 2*k0 + k1.
       
       Sub-problem (i0, j0, k0):
       A sub-block: rows 2*i0..2*i0+1, cols 2*j0..2*j0+1 (2x2)
       B sub-block: rows 2*j0..2*j0+1, cols 2*k0..2*k0+1 (2x2)
       C sub-block: rows 2*i0..2*i0+1, cols 2*k0..2*k0+1 (2x2)
       
       So we have 2*2*4 = 16 sub-problems of size 2x2x2.
       Each uses rank-7 Strassen.
       Total rank = 16 * 7 = 112.
    """
    base_terms = _strassen_2x2x2_terms()
    
    n, m, p = 4, 4, 8
    U_size = n * m  # 16
    V_size = m * p  # 32
    W_size = p * n  # 32
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i0 in range(2):
        for j0 in range(2):
            for k0 in range(4):  # k is split into 4 blocks of size 2
                for (tu, tv, tw) in base_terms:
                    u_col = {}
                    v_col = {}
                    w_col = {}
                    
                    for (li, lj), coeff in tu.items():
                        # Global i = 2*i0 + li
                        # Global j = 2*j0 + lj
                        gi = 2*i0 + li
                        gj = 2*j0 + lj
                        u_idx = gi * m + gj
                        u_col[u_idx] = u_col.get(u_idx, 0) + coeff
                    
                    for (lj, lk), coeff in tv.items():
                        # Global j = 2*j0 + lj
                        # Global k = 2*k0 + lk
                        gj = 2*j0 + lj
                        gk = 2*k0 + lk
                        v_idx = gj * p + gk
                        v_col[v_idx] = v_col.get(v_idx, 0) + coeff
                    
                    for (lk, li), coeff in tw.items():
                        # Global k = 2*k0 + lk
                        # Global i = 2*i0 + li
                        gk = 2*k0 + lk
                        gi = 2*i0 + li
                        # Output convention W[k*n+i, r]
                        w_idx = gk * n + gi
                        w_col[w_idx] = w_col.get(w_idx, 0) + coeff
                    
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
    
    rank = len(U_cols)
    U = [[0]*rank for _ in range(U_size)]
    V = [[0]*rank for _ in range(V_size)]
    W = [[0]*rank for _ in range(W_size)]
    
    for r in range(rank):
        for row, val in U_cols[r].items():
            U[row][r] = val
        for row, val in V_cols[r].items():
            V[row][r] = val
        for row, val in W_cols[r].items():
            W[row][r] = val
            
    return U, V, W, rank


def _build_schoolbook():
    """Schoolbook 4x4x8 decomposition, rank 128."""
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank


def _verify(U, V, W, rank):
    """Verify the decomposition is exact using random integer inputs."""
    try:
        import numpy as np
    except ImportError:
        return False

    n, m, p = 4, 4, 8
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)

    rng = random.Random(42)
    for _ in range(3):
        A = np.array([[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)], dtype=np.int64)
        B = np.array([[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)], dtype=np.int64)
        C_true = A.astype(np.int64) @ B.astype(np.int64)
        C = np.zeros((n, p), dtype=np.int64)
        for r in range(rank):
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            w_vec = W[:, r].reshape(p, n)
            C += (u_vec @ v_vec) * w_vec
        if not np.array_equal(C, C_true):
            return False
    return True


def construct():
    if _remaining() < 5.0:
        U, V, W, _ = _build_schoolbook()
        return {"U": U, "V": V, "W": W}

    try:
        U, V, W, rank = _build_4x4x8_direct_tiling()
        if _remaining() < 10.0:
            U, V, W, _ = _build_schoolbook()
            return {"U": U, "V": V, "W": W}
        if _verify(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    U, V, W, _ = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
