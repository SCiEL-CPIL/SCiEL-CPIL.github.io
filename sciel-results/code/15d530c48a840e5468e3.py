import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [6, 6, 6]
DEADLINE = 128  # safety margin before 150s kill


def _verify_vectorized(n, m, p, U, V, W):
    """Fast vectorized verification of the tensor decomposition."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.dot(Ua[a_idx], Va[b_idx] * Wa[c_idx])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _build_kronecker_6x6(n, m, p):
    """
    Build Kronecker product decomposition: Strassen(2x2) x Schoolbook(3x3).

    Split 6x6 matrices into 3x3 blocks of 2x2 sub-matrices.
    For each of the 27 (I,K,J) block triples, apply Strassen's 7-term algorithm
    to compute the 2x2 block product.

    Total rank = 27 * 7 = 189.
    """
    rank = 189
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    idx = 0
    for I in range(3):
        for K in range(3):
            for J in range(3):
                a00 = (2 * I) * m + 2 * K
                a01 = (2 * I) * m + 2 * K + 1
                a10 = (2 * I + 1) * m + 2 * K
                a11 = (2 * I + 1) * m + 2 * K + 1

                b00 = (2 * K) * p + 2 * J
                b01 = (2 * K) * p + 2 * J + 1
                b10 = (2 * K + 1) * p + 2 * J
                b11 = (2 * K + 1) * p + 2 * J + 1

                c00 = (2 * J) * n + 2 * I
                c01 = (2 * J + 1) * n + 2 * I
                c10 = (2 * J) * n + (2 * I + 1)
                c11 = (2 * J + 1) * n + (2 * I + 1)

                # M1
                U[a00, idx] = 1
                U[a11, idx] = 1
                V[b00, idx] = 1
                V[b11, idx] = 1
                W[c00, idx] = 1
                W[c11, idx] = 1
                idx += 1

                # M2
                U[a10, idx] = 1
                U[a11, idx] = 1
                V[b00, idx] = 1
                W[c10, idx] = 1
                W[c11, idx] = -1
                idx += 1

                # M3
                U[a00, idx] = 1
                V[b01, idx] = 1
                V[b11, idx] = -1
                W[c01, idx] = 1
                W[c11, idx] = 1
                idx += 1

                # M4
                U[a11, idx] = 1
                V[b10, idx] = 1
                V[b00, idx] = -1
                W[c00, idx] = 1
                W[c10, idx] = 1
                idx += 1

                # M5
                U[a00, idx] = 1
                U[a01, idx] = 1
                V[b11, idx] = 1
                W[c00, idx] = -1
                W[c01, idx] = 1
                idx += 1

                # M6
                U[a10, idx] = 1
                U[a00, idx] = -1
                V[b00, idx] = 1
                V[b01, idx] = 1
                W[c11, idx] = 1
                idx += 1

                # M7
                U[a01, idx] = 1
                U[a11, idx] = -1
                V[b10, idx] = 1
                V[b11, idx] = 1
                W[c00, idx] = 1
                idx += 1

    assert idx == rank, f"Expected {rank} terms, got {idx}"
    return U.tolist(), V.tolist(), W.tolist()


def _build_schoolbook(n, m, p):
    """Fallback: naive schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1

    return U.tolist(), V.tolist(), W.tolist()


def _try_strassen_kronecker_with_optimization(n, m, p):
    """
    Attempt to find a lower-rank decomposition by exploiting linear dependencies
    across the 27 Strassen block instances. This is a research direction that
    may reduce rank below 189 if linear dependencies exist.
    
    For now, we return the standard 189-rank decomposition as a baseline.
    A future improvement could use sparse linear algebra to detect and exploit
    common factors across block instances.
    """
    # Standard baseline
    U, V, W = _build_kronecker_6x6(n, m, p)
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    # Build the rank-189 Kronecker(Strassen-2x2 x Schoolbook-3x3) decomposition
    U, V, W = _build_kronecker_6x6(n, m, p)

    # Verify correctness with time guard
    if time.time() - start_time < DEADLINE - 10:
        if not _verify_vectorized(n, m, p, U, V, W):
            # Fallback to schoolbook if verification fails
            U, V, W = _build_schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
