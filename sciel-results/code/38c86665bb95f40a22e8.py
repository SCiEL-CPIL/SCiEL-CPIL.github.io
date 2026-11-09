import time
import numpy as np

DIMENSIONS = [4, 4, 8]


def _strassen_2x2_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a dict with keys 'U', 'V', 'W', each a dict mapping
    (row, col) -> coefficient.
    """
    terms = []
    # M1 = (A00+A11)*(B00+B11)
    terms.append({
        'U': {(0, 0): 1, (1, 1): 1},
        'V': {(0, 0): 1, (1, 1): 1},
        'W': {(0, 0): 1, (1, 1): 1}
    })
    # M2 = (A10+A11)*B00
    terms.append({
        'U': {(1, 0): 1, (1, 1): 1},
        'V': {(0, 0): 1},
        'W': {(1, 0): 1, (1, 1): -1}
    })
    # M3 = A00*(B01-B11)
    terms.append({
        'U': {(0, 0): 1},
        'V': {(0, 1): 1, (1, 1): -1},
        'W': {(0, 1): 1, (1, 1): 1}
    })
    # M4 = A11*(B10-B00)
    terms.append({
        'U': {(1, 1): 1},
        'V': {(1, 0): 1, (0, 0): -1},
        'W': {(0, 0): 1, (1, 0): 1}
    })
    # M5 = (A00+A01)*B11
    terms.append({
        'U': {(0, 0): 1, (0, 1): 1},
        'V': {(1, 1): 1},
        'W': {(0, 0): -1, (0, 1): 1}
    })
    # M6 = (A10-A00)*(B00+B01)
    terms.append({
        'U': {(1, 0): 1, (0, 0): -1},
        'V': {(0, 0): 1, (0, 1): 1},
        'W': {(1, 1): 1}
    })
    # M7 = (A01-A11)*(B10+B11)
    terms.append({
        'U': {(0, 1): 1, (1, 1): -1},
        'V': {(1, 0): 1, (1, 1): 1},
        'W': {(0, 0): 1}
    })
    return terms


def _build_4x4x4_strassen():
    """
    Builds a rank-49 decomposition for 4x4x4 matrix multiplication
    using recursive Strassen (2x2 blocks of 2x2 matrices).
    Returns (U, V, W) each of shape (16, 49).
    U[i*4+j, r] = coefficient of A[i,j] in r-th term
    V[j*4+k, r] = coefficient of B[j,k] in r-th term
    W[k*4+i, r] = coefficient for C[k,i] in r-th term (trace convention)
    """
    strassen_terms = _strassen_2x2_terms()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))

    col = 0
    for meta_idx, meta_term in enumerate(strassen_terms):
        for inner_idx, inner_term in enumerate(strassen_terms):
            r = meta_idx * 7 + inner_idx

            # U: A[bi*2+ii, bj*2+jj]
            for (bi, bj), cu in meta_term['U'].items():
                for (ii, jj), cu_inner in inner_term['U'].items():
                    row_a = bi * 2 + ii
                    col_a = bj * 2 + jj
                    U[row_a * 4 + col_a, r] += cu * cu_inner

            # V: B[bj*2+jj, bk*2+kk]
            for (bj, bk), cv in meta_term['V'].items():
                for (jj, kk), cv_inner in inner_term['V'].items():
                    row_b = bj * 2 + jj
                    col_b = bk * 2 + kk
                    V[row_b * 4 + col_b, r] += cv * cv_inner

            # W: C[bk*2+kk, bi*2+ii]
            for (bi, bk), cw in meta_term['W'].items():
                for (ii, kk), cw_inner in inner_term['W'].items():
                    row_c = bk * 2 + kk
                    col_c = bi * 2 + ii
                    W[row_c * 4 + col_c, r] += cw * cw_inner

    return U, V, W


def _build_4x4x8_strassen():
    """
    Builds a rank-98 decomposition for 4x4x8 matrix multiplication
    by applying the 49-rank 4x4x4 Strassen decomposition to each
    4-column block of the 8-column B matrix.
    Returns (U, V, W) with shapes (16, 98), (32, 98), (32, 98).
    """
    U4, V4, W4 = _build_4x4x4_strassen()
    rank_444 = 49
    total_rank = 2 * rank_444

    U = np.zeros((16, total_rank))
    V = np.zeros((32, total_rank))
    W = np.zeros((32, total_rank))

    for half in range(2):
        for r in range(rank_444):
            col = half * rank_444 + r

            # U is the same for both halves (same A matrix)
            U[:, col] = U4[:, r]

            # V: columns half*4 to half*4+3 of B
            for j in range(4):
                for k in range(4):
                    V[j * 8 + half * 4 + k, col] = V4[j * 4 + k, r]

            # W: rows half*4 to half*4+3 of C
            for k in range(4):
                for i in range(4):
                    W[(half * 4 + k) * 4 + i, col] = W4[k * 4 + i, r]

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition correctly computes C = A @ B.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W[k*n+i, r] is the coefficient for C[i,k] (trace convention: W is indexed by output (k,i) flattened as k*n+i)
    """
    rank = U.shape[1]
    np.random.seed(12345)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    C_computed = np.zeros((n, p))
    for r in range(rank):
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)  # W_r[k, i] = W[k*n + i, r]

        a_part = np.sum(U_r * A_test)
        b_part = np.sum(V_r * B_test)

        C_computed += a_part * b_part * W_r.T

    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Build the 4x4x8 Strassen decomposition (rank 98)
    U, V, W = _build_4x4x8_strassen()

    # Verify correctness
    if time.time() < deadline:
        if _verify_decomposition(U, V, W, n, m, p):
            # Convert to list-of-lists as required by interface
            U_out = U.tolist()
            V_out = V.tolist()
            W_out = W.tolist()
            return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback: schoolbook decomposition (rank 128)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
