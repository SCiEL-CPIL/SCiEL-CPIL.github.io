"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE = 120.0
SAFETY_MARGIN = 10.0  # seconds to keep before the hard limit

# EVOLVE-BLOCK-START

def _build_schoolbook():
    """Return (U, V, W) as lists-of-lists for the schoolbook rank-36 decomposition."""
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
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # For each (i,j,k), compute the sum over r.
    # U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
    # sum_r U_ij_r * V_jk_r * W_ki_r
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_block_rank_18():
    """
    Attempt to find a rank-9 decomposition of the 2x3x3 multiplication tensor
    via a structured ansatz (random initialisation + local refinement), then
    assemble a rank-18 decomposition for 2x3x6 by duplicating the block.
    Returns (U, V, W) as lists-of-lists if successful, else None.
    """
    n, m, p3 = 2, 3, 3
    R = 9

    def _rand_block():
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p3, R) * 0.5
        W = np.random.randn(p3 * n, R) * 0.5
        return U, V, W

    def _loss(U, V, W):
        # T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # We want T == 1 for all (i,j,k).
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p3):
                    s = np.dot(U[i * m + j], V[j * p3 + k] * W[k * n + i])
                    loss += (s - 1.0) ** 2
        return loss

    best_loss = float('inf')
    best_UVW = None

    start = time.time()
    max_time = DEADLINE - SAFETY_MARGIN - 5  # leave margin for assembly and verification

    for attempt in range(200):
        if time.time() - start > max_time:
            break
        U, V, W = _rand_block()
        lr = 0.01
        for it in range(500):
            if time.time() - start > max_time:
                break
            # Compute gradient numerically? Use simple SGD with finite differences on small subsample
            # For speed, use a few random coordinate updates.
            loss = _loss(U, V, W)
            if loss < best_loss:
                best_loss = loss
                best_UVW = (U.copy(), V.copy(), W.copy())
                if best_loss < 1e-8:
                    break
            # Random coordinate descent on one entry of U, V, or W
            mat = random.choice([U, V, W])
            row = random.randrange(mat.shape[0])
            col = random.randrange(mat.shape[1])
            eps = 0.01
            old = mat[row, col]
            mat[row, col] = old + eps
            l_plus = _loss(U, V, W)
            mat[row, col] = old - eps
            l_minus = _loss(U, V, W)
            mat[row, col] = old
            grad = (l_plus - l_minus) / (2 * eps)
            mat[row, col] -= lr * grad
            if time.time() - start > max_time:
                break

        if best_loss < 1e-8:
            break

    if best_UVW is None:
        return None

    U3, V3, W3 = best_UVW

    # Verify the 2x3x3 block
    if not _verify(U3.tolist(), V3.tolist(), W3.tolist(), n, m, p3, tol=1e-6):
        return None

    # Assemble rank-18 for 2x3x6:
    # n=2, m=3, p=6
    # Block 1: k in {0,1,2}, Block 2: k in {3,4,5}
    n, m, p = 2, 3, 6
    R_total = 2 * R  # 18
    U6 = np.zeros((n * m, R_total))
    V6 = np.zeros((m * p, R_total))
    W6 = np.zeros((p * n, R_total))

    # Block 1 occupies columns 0..8
    U6[:, 0:R] = U3
    for j in range(m):
        for k in range(p3):
            V6[j * p + k, 0:R] = V3[j * p3 + k]
    for k in range(p3):
        for i in range(n):
            W6[k * n + i, 0:R] = W3[k * n + i]

    # Block 2 occupies columns R..2R-1
    U6[:, R:2 * R] = U3
    for j in range(m):
        for k in range(p3, 2 * p3):
            V6[j * p + k, R:2 * R] = V3[j * p3 + (k - p3)]
    for k in range(p3, 2 * p3):
        for i in range(n):
            W6[k * n + i, R:2 * R] = W3[(k - p3) * n + i]

    U_list = U6.tolist()
    V_list = V6.tolist()
    W_list = W6.tolist()

    if _verify(U_list, V_list, W_list, n, m, p, tol=1e-6):
        return U_list, V_list, W_list
    return None


def construct():
    """
    Return {'U', 'V', 'W'} factor matrices for the 2x3x6 matrix multiplication tensor.
    Tries to find a rank-18 decomposition first; falls back to schoolbook rank-36.
    """
    n, m, p = DIMENSIONS

    # Try rank-18 via block structure
    result = _try_block_rank_18()
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 36
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
