"""Kronecker-structured Strassen decomposition for 4x5x6 matrix multiplication.

Strategy: exploit the tensor-product identity
    MM_{n*m, m*p, p*n} = (MM_{n,n,n} ⊗ MM_{m,m,m}) ⊗ ...
More concretely for 4x5x6:
  - The 4x4x4 core (i<4, j<4, k<4) uses recursive Strassen (rank 49).
  - The remaining slabs (j=4 or k>=4) are handled with minimal-rank
    structured blocks: 4x1x6 and 4x4x2.
  
Key structural improvement over the seed:
  - Use the Winograd/Strassen 2x2x2 rank-7 identity directly (not via
    recursive 4x4x4 which has known rank 49 but we verify correctness).
  - For the 4x4x2 slab, use 4 independent 2x2x2 Strassen blocks (rank 28).
  - For the 4x1x6 slab, schoolbook is optimal (rank 24).
  - Total rank = 49 + 24 + 28 = 101.
  
  The seed also achieves 101, but the implementation here is cleaner and
  we add a fast verification path using numpy for speed.
"""
import time
import random

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0


def remaining():
    return TIME_LIMIT - (time.time() - START_TIME)


def _strassen_2x2_terms():
    """Return 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (U_vec, V_vec, W_vec) of length 4 each.
    U_vec indexes [a11, a12, a21, a22]
    V_vec indexes [b11, b12, b21, b22]
    W_vec indexes [c11, c12, c21, c22]
    """
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    terms.append(([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]))
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    terms.append(([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return terms


def _embed_2x2x2_into_block(U, V, W, i0, j0, k0, n, m, p, col_offset):
    """Embed a 2x2x2 Strassen block at position (i0,j0,k0) into full matrices."""
    terms = _strassen_2x2_terms()
    for t, (u_vec, v_vec, w_vec) in enumerate(terms):
        col = col_offset + t
        for li in range(2):
            for lj in range(2):
                local_idx = li * 2 + lj
                val = u_vec[local_idx]
                if val != 0:
                    U[(i0 + li) * m + (j0 + lj)][col] = val
        for lj in range(2):
            for lk in range(2):
                local_idx = lj * 2 + lk
                val = v_vec[local_idx]
                if val != 0:
                    V[(j0 + lj) * p + (k0 + lk)][col] = val
        for lk in range(2):
            for li in range(2):
                local_idx = lk * 2 + li
                val = w_vec[local_idx]
                if val != 0:
                    W[(k0 + lk) * n + (i0 + li)][col] = val


def _embed_4x4x4_strassen(U, V, W, n, m, p, col_offset):
    """Embed 4x4x4 Strassen (rank 49) at top-left corner using recursive 2x2 structure."""
    terms_2x2 = _strassen_2x2_terms()
    # Top-level 7 terms for 4x4x4 (each is a 2x2x2 of 2x2 blocks)
    # Using standard Strassen recursion: 4x4x4 = 7 * (2x2x2)
    # Each top-level term corresponds to a combination of the 4 quadrants
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3
    # M4 = A11(B10-B00) -> C10 += M4, C00 -= M4, C01 += M4
    # M5 = (A00+A01)B10 -> C00 += M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C10 -= M6, C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 -= M7, C01 += M7, C10 += M7
    
    # For each top-level term s (0..6), and each bottom-level term t (0..6):
    # We need to embed the product of the top-level structure with the bottom-level Strassen
    
    # Top-level specs: for each s, which quadrants of A, B, C are involved and with what signs
    top_specs = [
        # s=0: M1 = (A00+A11)(B00+B11)
        {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
        # s=1: M2 = (A10+A11)B00
        {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)], 'C': [((1,0), 1), ((1,1), -1)]},
        # s=2: M3 = A00(B01-B11)
        {'A': [((0,0), 1)], 'B': [((0,1), 1), ((1,1), -1)], 'C': [((0,1), 1)]},
        # s=3: M4 = A11(B10-B00)
        {'A': [((1,1), 1)], 'B': [((1,0), 1), ((0,0), -1)], 'C': [((0,0), -1), ((0,1), 1), ((1,0), 1)]},
        # s=4: M5 = (A00+A01)B10
        {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,0), 1)], 'C': [((0,0), 1), ((0,1), 1)]},
        # s=5: M6 = (A10-A00)(B00+B01)
        {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)], 'C': [((1,0), -1), ((1,1), 1)]},
        # s=6: M7 = (A01-A11)(B10+B11)
        {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)], 'C': [((0,0), -1), ((0,1), 1), ((1,0), 1)]},
    ]
    
    for s, spec in enumerate(top_specs):
        for t in range(7):
            col = col_offset + s * 7 + t
            u_vec, v_vec, w_vec = terms_2x2[t]
            
            # Embed A part: for each (bi, bj) in spec['A'], add u_vec scaled
            for (bi, bj), sign_a in spec['A']:
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        val = u_vec[local_idx]
                        if val != 0:
                            U[(bi * 2 + li) * m + (bj * 2 + lj)][col] = val * sign_a
            
            # Embed B part: for each (bj, bk) in spec['B'], add v_vec scaled
            for (bj, bk), sign_b in spec['B']:
                for lj in range(2):
                    for lk in range(2):
                        local_idx = lj * 2 + lk
                        val = v_vec[local_idx]
                        if val != 0:
                            V[(bj * 2 + lj) * p + (bk * 2 + lk)][col] = val * sign_b
            
            # Embed C part: for each (ci, ck) in spec['C'], add w_vec scaled
            for (ci, ck), sign_c in spec['C']:
                for li in range(2):
                    for lk in range(2):
                        local_idx = lk * 2 + li
                        val = w_vec[local_idx]
                        if val != 0:
                            W[(ck * 2 + lk) * n + (ci * 2 + li)][col] = val * sign_c


def _embed_4x1x6_schoolbook(U, V, W, n, m, p, col_offset):
    """Embed 4x1x6 schoolbook block: A col 4, B row 4, C 4x6. Rank = 24."""
    for i in range(4):
        for k in range(6):
            col = col_offset + i * 6 + k
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1


def _embed_4x4x2_strassen(U, V, W, n, m, p, col_offset):
    """Embed 4x4x2 using 4 blocks of 2x2x2 Strassen. Rank = 28."""
    blocks = [(0, 0, 4), (0, 2, 4), (2, 0, 4), (2, 2, 4)]
    for idx, (i0, j0, k0) in enumerate(blocks):
        _embed_2x2x2_into_block(U, V, W, i0, j0, k0, n, m, p, col_offset + idx * 7)


def _verify_fast(U, V, W, rank, n, m, p):
    """Fast verification using numpy if available, else pure Python."""
    try:
        import numpy as np
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        for _ in range(3):
            A = np.random.randint(-2, 3, size=(n, m))
            B = np.random.randint(-2, 3, size=(m, p))
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for c in range(rank):
                uc = U_np[:, c].reshape(n, m)
                vc = V_np[:, c].reshape(m, p)
                wc = W_np[:, c].reshape(p, n)
                C_computed += (uc @ vc) * wc.T
            if not np.array_equal(C_computed, C_expected):
                return False
        return True
    except Exception:
        for _ in range(3):
            A = [[random.randint(-2, 2) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-2, 2) for _ in range(p)] for _ in range(m)]
            C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
            for i in range(n):
                for k in range(p):
                    acc = 0
                    for j in range(m):
                        for c in range(rank):
                            acc += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if acc != C[i][k]:
                        return False
        return True


def _schoolbook_fallback():
    n, m, p = DIMENSIONS
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


def construct():
    n, m, p = DIMENSIONS

    # Total rank: 49 (4x4x4 Strassen) + 24 (4x1x6 schoolbook) + 28 (4x4x2 Strassen) = 101
    total_rank = 49 + 24 + 28

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Block 1: 4x4x4 Strassen at (i:0-3, j:0-3, k:0-3)
    _embed_4x4x4_strassen(U, V, W, n, m, p, 0)

    # Block 2: 4x1x6 schoolbook at (i:0-3, j=4, k:0-5)
    _embed_4x1x6_schoolbook(U, V, W, n, m, p, 49)

    # Block 3: 4x4x2 Strassen at (i:0-3, j:0-3, k:4-5)
    _embed_4x4x2_strassen(U, V, W, n, m, p, 73)

    # Verify
    if remaining() > 5:
        if not _verify_fast(U, V, W, total_rank, n, m, p):
            return _schoolbook_fallback()

    return {"U": U, "V": V, "W": W}
