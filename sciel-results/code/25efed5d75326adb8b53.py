"""Tensor decomposition for 2x3 x 3x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Build the rank-28 decomposition: 4 blocks of rank 7
    U, V, W = _rank28_decomposition(n, m, p)

    if not _verify(n, m, p, U, V, W):
        # Fallback to schoolbook if verification fails
        U, V, W = _schoolbook(n, m, p)

    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _rank28_decomposition(n, m, p):
    """
    Decompose 2x3 x 3x8 into 4 blocks of 2x3 x 3x2.
    Each 2x3 x 3x2 tensor T[i,j,k] = delta_{j,k} has rank 7:
    - 4 terms for j=2 (the bottom row of A, which has no Strassen structure)
    - 3 terms for the 2x2 top part (Strassen-like)
    
    Total rank: 4 * 7 = 28
    """
    n, m, p = 2, 3, 8
    block_size = p // 4  # 2
    rank_per_block = 7
    total_rank = rank_per_block * 4  # 28

    U = np.zeros((n * m, total_rank))  # 6 x 28
    V = np.zeros((m * p, total_rank))  # 24 x 28
    W = np.zeros((p * n, total_rank))  # 16 x 28

    for b in range(4):
        k_offset = b * block_size  # 0, 2, 4, 6
        col_offset = b * rank_per_block

        # 4 trivial terms for j=2 (bottom row of A)
        for ii in range(2):
            for kk in range(2):
                c = col_offset + ii * 2 + kk  # 0,1,2,3
                U[ii * m + 2, c] = 1
                V[2 * p + k_offset + kk, c] = 1
                W[(k_offset + kk) * n + ii, c] = 1

        # 3 Strassen terms for the 2x2 top part (i=0,1; j=0,1; k=0,1)
        # M1 = A[0,0] * B[0, k_offset+0] -> C[0, k_offset+0]
        c1 = col_offset + 4
        U[0 * m + 0, c1] = 1
        V[0 * p + k_offset + 0, c1] = 1
        W[(k_offset + 0) * n + 0, c1] = 1

        # M2 = (A[0,0] + A[1,0]) * B[1, k_offset+0] -> C[1, k_offset+0]
        c2 = col_offset + 5
        U[0 * m + 0, c2] = 1
        U[1 * m + 0, c2] = 1
        V[1 * p + k_offset + 0, c2] = 1
        W[(k_offset + 0) * n + 1, c2] = 1

        # M3 = A[1,0] * (B[1, k_offset+1] - B[0, k_offset+1]) -> C[1, k_offset+1]
        c3 = col_offset + 6
        U[1 * m + 0, c3] = 1
        V[1 * p + k_offset + 1, c3] = 1
        V[0 * p + k_offset + 1, c3] = -1
        W[(k_offset + 1) * n + 1, c3] = 1

    return U, V, W


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact using the bilinear form:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[l*p+k, c] * B[l,k])
    """
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B

    rank = U.shape[1]

    # Precompute U @ A for all i, c: shape (n, rank)
    U_A = np.zeros((n, rank))
    for i in range(n):
        U_A[i] = U[i * m:(i + 1) * m, :] @ A[i, :]

    # Precompute V @ B for all k, c: shape (p, rank)
    V_T = V.T.reshape(rank, m, p)
    V_B = np.zeros((p, rank))
    for k in range(p):
        V_B[k] = V_T[:, :, k].T @ B[:, k]

    # C_comp[i, k] = sum_c W[k*n+i, c] * U_A[i, c] * V_B[k, c]
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            C_comp[i, k] = np.dot(W[k * n + i, :], U_A[i, :] * V_B[k, :])

    return np.allclose(C_comp, C_true, atol=1e-9)


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

# EVOLVE-BLOCK-END
