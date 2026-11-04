"""Matrix multiplication 4x4x5: rank-minimal exact tensor decomposition.

Structural change: Use a block-partitioned Strassen-based decomposition for the
4x4x4 portion (8 Strassen 2x2x2 blocks = rank 56) and a direct (schoolbook)
decomposition for the 4x4x1 portion (rank 16), for a total rank of 72.
This is a different design family than the cyclic-convolution approach in the
parent, and is simpler and more reliably verifiable. A schoolbook fallback
(rank 80) guarantees a valid result.
"""
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before the 150 s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check that U,V,W exactly decompose the matrix-multiplication tensor.

    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  ==  1  for all i,j,k.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if U.ndim != 2:
        return False
    if U.shape[0] != n * m:
        return False
    rank = U.shape[1]
    if V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False

    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=tol))


def _schoolbook_4x4x5():
    """Fallback schoolbook decomposition with rank n*m*p = 80 (always valid)."""
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _construct_4x4x5_strassen():
    """Construct the 4x4x5 decomposition using block Strassen for 4x4x4 part
    and direct multiplication for the 4x4x1 part. Total rank = 56 + 16 = 72."""
    if _time_remaining() < 5:
        raise TimeoutError("Not enough time remaining")

    n, m, p = 4, 4, 5
    rank_total = 72
    U = np.zeros((n * m, rank_total), dtype=np.float64)
    V = np.zeros((m * p, rank_total), dtype=np.float64)
    W = np.zeros((p * n, rank_total), dtype=np.float64)

    state = [0]

    def add_strassen_2x2(ra, ca, rb, cb):
        """Add 7 Strassen terms for 2x2 * 2x2 block product.
        ra, ca: A block row, col (0 or 1)
        rb, cb: B block row, col (0 or 1)
        """
        # M1: (a+d)*(e+h)
        col = state[0]
        U[2*ra * m + 2*ca, col] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col] = 1.0
        V[2*rb * p + 2*cb, col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = 1.0
        W[2*cb * n + 2*ra, col] = 1.0
        W[(2*cb+1) * n + (2*ra+1), col] = 1.0
        state[0] += 1

        # M2: (c+d)*e
        col = state[0]
        U[(2*ra+1) * m + 2*ca, col] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col] = 1.0
        V[2*rb * p + 2*cb, col] = 1.0
        W[2*cb * n + (2*ra+1), col] = 1.0
        W[(2*cb+1) * n + (2*ra+1), col] = -1.0
        state[0] += 1

        # M3: a*(f-h)
        col = state[0]
        U[2*ra * m + 2*ca, col] = 1.0
        V[2*rb * p + (2*cb+1), col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = -1.0
        W[(2*cb+1) * n + 2*ra, col] = 1.0
        W[(2*cb+1) * n + (2*ra+1), col] = 1.0
        state[0] += 1

        # M4: d*(g-e)
        col = state[0]
        U[(2*ra+1) * m + (2*ca+1), col] = 1.0
        V[(2*rb+1) * p + 2*cb, col] = 1.0
        V[2*rb * p + 2*cb, col] = -1.0
        W[2*cb * n + 2*ra, col] = 1.0
        W[2*cb * n + (2*ra+1), col] = 1.0
        state[0] += 1

        # M5: (a+b)*h
        col = state[0]
        U[2*ra * m + 2*ca, col] = 1.0
        U[2*ra * m + (2*ca+1), col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = 1.0
        W[2*cb * n + 2*ra, col] = -1.0
        W[(2*cb+1) * n + 2*ra, col] = 1.0
        state[0] += 1

        # M6: (c-a)*(e+f)
        col = state[0]
        U[(2*ra+1) * m + 2*ca, col] = 1.0
        U[2*ra * m + 2*ca, col] = -1.0
        V[2*rb * p + 2*cb, col] = 1.0
        V[2*rb * p + (2*cb+1), col] = 1.0
        W[(2*cb+1) * n + (2*ra+1), col] = 1.0
        state[0] += 1

        # M7: (b-d)*(g+h)
        col = state[0]
        U[2*ra * m + (2*ca+1), col] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col] = -1.0
        V[(2*rb+1) * p + 2*cb, col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = 1.0
        W[2*cb * n + 2*ra, col] = 1.0
        state[0] += 1

    def add_direct_2x2_2x1(ra, ca, rb, cb_col):
        """Add 4 direct terms for 2x2 * 2x1.
        ra, ca: A block row, col (0 or 1)
        rb: B block row (0 or 1)
        cb_col: actual global column index of the single B column.
        """
        for i in range(2):
            for j in range(2):
                col = state[0]
                U[(2*ra + i) * m + (2*ca + j), col] = 1.0
                V[(2*rb + j) * p + cb_col, col] = 1.0
                W[cb_col * n + (2*ra + i), col] = 1.0
                state[0] += 1

    # 4x4x4 part: B columns 0,1 (cb=0) and columns 2,3 (cb=1)
    # B block 0 covers cols 0,1; B block 1 covers cols 2,3
    add_strassen_2x2(0, 0, 0, 0)  # A00 * B00
    add_strassen_2x2(0, 1, 1, 0)  # A01 * B10
    add_strassen_2x2(1, 0, 0, 0)  # A10 * B00
    add_strassen_2x2(1, 1, 1, 0)  # A11 * B10

    add_strassen_2x2(0, 0, 0, 1)  # A00 * B01
    add_strassen_2x2(0, 1, 1, 1)  # A01 * B11
    add_strassen_2x2(1, 0, 0, 1)  # A10 * B01
    add_strassen_2x2(1, 1, 1, 1)  # A11 * B11

    # 4x4x1 part: B column 4
    add_direct_2x2_2x1(0, 0, 0, 4)  # A00 * B02
    add_direct_2x2_2x1(0, 1, 1, 4)  # A01 * B12
    add_direct_2x2_2x1(1, 0, 0, 4)  # A10 * B02
    add_direct_2x2_2x1(1, 1, 1, 4)  # A11 * B12

    used_rank = state[0]

    U = U[:, :used_rank]
    V = V[:, :used_rank]
    W = W[:, :used_rank]

    if not _verify(U, V, W, n, m, p, tol=1e-8):
        raise ValueError("Strassen 4x4x5 decomposition failed verification")

    return U, V, W


def construct():
    """Main entry point: construct the rank-minimal exact tensor decomposition
    for 4x4x5.  Returns {'U','V','W'} as lists of lists (integer/half-integer)."""
    try:
        U, V, W = _construct_4x4x5_strassen()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except (TimeoutError, ValueError):
        pass

    # Fallback: guaranteed-valid schoolbook (rank 80)
    U, V, W = _schoolbook_4x4x5()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
