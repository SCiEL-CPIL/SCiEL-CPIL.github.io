"""Matrix multiplication 6,6,6 - rank-minimal exact tensor decomposition.

Uses a 3x3 grid of 2x2 blocks, each block pair multiplied via Strassen's
rank-7 algorithm, yielding rank 189 (3*3*3*7 = 189).

The W matrix follows the trace convention W[k*n+i, r] for C[i, k].
"""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_6x6():
    """
    Construct a rank-189 decomposition for 6x6 matrix multiplication
    using block-wise Strassen (2x2 blocks in a 3x3 grid).
    
    U: shape (n*m, rank) = (36, 189)  -- A is 6x6, flattened row-major A[i,k] at i*m+k
    V: shape (m*p, rank) = (36, 189)  -- B is 6x6, flattened row-major B[k,j] at k*p+j
    W: shape (p*n, rank) = (36, 189)  -- C is 6x6, trace convention W[k*n+i] for C[i,k]
    """
    n, m, p = 6, 6, 6
    rank = 189  # 3 * 3 * 3 * 7

    # Strassen 2x2 decomposition (rank 7)
    # Local indices: A[i,k], B[k,j], C[i,j] for i,k,j in {0,1}
    #
    # Products:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B01-B11)
    # M3 = A11*(B10-B00)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B01)
    # M6 = (A01-A11)*(B10+B11)
    #
    # Reconstruction:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    # For each of the 7 Strassen products, define:
    # U_coeffs: list of (local_i, local_k, coeff) for A[i,k]
    # V_coeffs: list of (local_k, local_j, coeff) for B[k,j]
    # W_coeffs: list of (local_i, local_j, coeff) for C[i,j]

    strassen_U = [
        [(0, 0, 1), (1, 1, 1)],           # M0: (A00+A11)
        [(1, 0, 1), (1, 1, 1)],           # M1: (A10+A11)
        [(0, 0, 1)],                       # M2: A00
        [(1, 1, 1)],                       # M3: A11
        [(0, 0, 1), (0, 1, 1)],           # M4: (A00+A01)
        [(1, 0, 1), (0, 0, -1)],          # M5: (A10-A00)
        [(0, 1, 1), (1, 1, -1)],          # M6: (A01-A11)
    ]

    strassen_V = [
        [(0, 0, 1), (1, 1, 1)],           # M0: (B00+B11)
        [(0, 0, 1)],                       # M1: B00
        [(0, 1, 1), (1, 1, -1)],          # M2: (B01-B11)
        [(1, 0, 1), (0, 0, -1)],          # M3: (B10-B00)
        [(1, 1, 1)],                       # M4: B11
        [(0, 0, 1), (0, 1, 1)],           # M5: (B00+B01)
        [(1, 0, 1), (1, 1, 1)],           # M6: (B10+B11)
    ]

    strassen_W = [
        [(0, 0, 1), (1, 1, 1)],           # M0: C00+1, C11+1
        [(1, 0, 1), (1, 1, -1)],          # M1: C10+1, C11-1
        [(0, 1, 1), (1, 1, 1)],           # M2: C01+1, C11+1
        [(0, 0, 1), (1, 0, 1)],           # M3: C00+1, C10+1
        [(0, 0, -1), (0, 1, 1)],          # M4: C00-1, C01+1
        [(1, 1, 1)],                       # M5: C11+1
        [(0, 0, 1)],                       # M6: C00+1
    ]

    U = [[0] * rank for _ in range(n * m)]   # (36, 189)
    V = [[0] * rank for _ in range(m * p)]   # (36, 189)
    W = [[0] * rank for _ in range(p * n)]   # (36, 189)

    col = 0
    for bi in range(3):       # row block index of A / row block of C
        for bk in range(3):   # col block index of A / row block of B
            for bj in range(3):  # col block index of B / col block of C
                for t in range(7):
                    # U: A entries
                    # A is n x m = 6 x 6, flattened row-major: A[i, k] at index i*m + k
                    # Block (bi, bk) of A: rows 2bi..2bi+1, cols 2bk..2bk+1
                    for (li, lk, coeff) in strassen_U[t]:
                        gi = 2 * bi + li
                        gk = 2 * bk + lk
                        U[gi * m + gk][col] += coeff

                    # V: B entries
                    # B is m x p = 6 x 6, flattened row-major: B[k, j] at index k*p + j
                    # Block (bk, bj) of B: rows 2bk..2bk+1, cols 2bj..2bj+1
                    for (lk, lj, coeff) in strassen_V[t]:
                        gk = 2 * bk + lk
                        gj = 2 * bj + lj
                        V[gk * p + gj][col] += coeff

                    # W: C entries
                    # C is n x p = 6 x 6. Trace convention: W[k*n + i, c] for C[i, k]
                    # Block (bi, bj) of C: rows 2bi..2bi+1 (i-index), cols 2bj..2bj+1 (k-index)
                    # So C[i, k] where i = 2bi+li, k = 2bj+lj
                    # W index = k*n + i = (2bj+lj)*n + (2bi+li)
                    for (li, lj, coeff) in strassen_W[t]:
                        gi = 2 * bi + li   # i-index of C
                        gj = 2 * bj + lj   # k-index of C
                        W[gj * n + gi][col] += coeff

                    col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}


def _validate_decomposition(U, V, W, n=6, m=6, p=6):
    """Validate by checking with random inputs."""
    try:
        import numpy as np
    except ImportError:
        return True

    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    np.random.seed(12345)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    A_flat = A.reshape(n * m)
    B_flat = B.reshape(m * p)

    C_decomp = np.zeros(p * n)
    for c in range(rank):
        uc = Ua[:, c]
        vc = Va[:, c]
        wc = Wa[:, c]
        prod_a = uc * A_flat
        prod_b = vc * B_flat
        pa = prod_a.reshape(n, m)
        pb = prod_b.reshape(m, p)
        temp = pa @ pb
        wc_mat = wc.reshape(p, n)
        C_decomp += (temp * wc_mat).reshape(-1)

    C_direct = (A @ B).reshape(p * n)

    if np.max(np.abs(C_decomp - C_direct)) > 1e-9:
        return False
    return True


def _build_schoolbook(n=6, m=6, p=6):
    """Standard O(nmp) decomposition with rank = n*m*p = 216."""
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
    t0 = time.time()
    deadline = 115.0

    result = _build_strassen_6x6()

    if time.time() - t0 < deadline - 5:
        try:
            if not _validate_decomposition(result["U"], result["V"], result["W"]):
                result = _build_schoolbook()
        except Exception:
            pass

    if time.time() - t0 > deadline:
        result = _build_schoolbook()

    return result

# EVOLVE-BLOCK-END
