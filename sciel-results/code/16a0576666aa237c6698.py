"""Rewritten program: block-based tensor decomposition for 4x5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]
_DEADLINE_SECONDS = 115  # safety margin before 120s kill

# EVOLVE-BLOCK-START


def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists, rank n*m*p."""
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


def _strassen_2x2():
    """Strassen 7-product decomposition for 2x2x2 matrix multiplication.
    Verified indices:
    U (A):
    0: [1, 0, 1, 0, 1, -1, 0]
    1: [0, 0, 0, 0, 1, 0, 1]
    2: [0, 1, 0, 1, 0, 0, -1]
    3: [1, 1, 0, 1, 0, 0, 0]
    V (B):
    0: [1, 1, 0, 0, 0, 1, 0]
    1: [0, 0, 1, 0, 0, 1, 0]
    2: [0, 0, 0, 1, 0, 0, 1]
    3: [1, 0, -1, -1, 1, 0, 1]
    W (C):
    0: [1, 0, 0, 1, -1, 0, 1]
    1: [0, 0, 1, 0, 1, 0, 0]
    2: [0, 1, 0, 1, 0, 0, 0]
    3: [1, -1, 1, 0, 0, 1, 0]
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, -1],
        [1, 1, 0, 1, 0, 0, 0]
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W


def _embed_subproblem(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_offset, j_offset, k_offset, n, m, p):
    """Embed a subproblem decomposition into the full (n,m,p) space."""
    rank_sub = len(U_sub[0])
    U_full = [[0] * rank_sub for _ in range(n * m)]
    V_full = [[0] * rank_sub for _ in range(m * p)]
    W_full = [[0] * rank_sub for _ in range(p * n)]

    for local_i in range(n_sub):
        for local_j in range(m_sub):
            full_i = i_offset + local_i
            full_j = j_offset + local_j
            local_idx = local_i * m_sub + local_j
            full_idx = full_i * m + full_j
            U_full[full_idx] = list(U_sub[local_idx])

    for local_j in range(m_sub):
        for local_k in range(p_sub):
            full_j = j_offset + local_j
            full_k = k_offset + local_k
            local_idx = local_j * p_sub + local_k
            full_idx = full_j * p + full_k
            V_full[full_idx] = list(V_sub[local_idx])

    for local_k in range(p_sub):
        for local_i in range(n_sub):
            full_k = k_offset + local_k
            full_i = i_offset + local_i
            local_idx = local_k * n_sub + local_i
            full_idx = full_k * n + full_i
            W_full[full_idx] = list(W_sub[local_idx])

    return U_full, V_full, W_full


def _concat(U1, V1, W1, U2, V2, W2, n, m, p):
    """Horizontally concatenate two decompositions (independent tensor terms)."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    U = [U1[i] + U2[i] for i in range(n * m)]
    V = [V1[i] + V2[i] for i in range(m * p)]
    W = [W1[i] + W2[i] for i in range(p * n)]
    return U, V, W


def _multi_concat(decomps, n, m, p):
    """Concatenate a list of (U,V,W) decompositions."""
    if len(decomps) == 0:
        U = [[0] * 0 for _ in range(n * m)]
        V = [[0] * 0 for _ in range(m * p)]
        W = [[0] * 0 for _ in range(p * n)]
        return U, V, W
    result = decomps[0]
    for d in decomps[1:]:
        result = _concat(result[0], result[1], result[2], d[0], d[1], d[2], n, m, p)
    return result


def _construct_4x5x5():
    """Build 4x5x5 decomposition via block partitioning.
    Rank 92 strategy:
    Split j (5) into 2, 1, 2.
    Block 1: 4x2 * 2x5. Split i (4) into 2,2. Split k (5) into 2,2,1.
      Each 2x2 * 2x2 uses Strassen (7).
      Each 2x2 * 2x1 uses Schoolbook (4).
      Cost per 2x2 row block: 7+7+4 = 18.
      Total for 4x2*2x5: 18+18 = 36.
    Block 2: 4x1 * 1x5. Schoolbook.
      Cost: 4*1*5 = 20.
    Block 3: 4x2 * 2x5. Same as Block 1.
      Cost: 36.
    Total: 36 + 20 + 36 = 92.
    """
    n, m, p = 4, 5, 5
    parts = []

    # --- Block 1: A[0:4, 0:2] * B[0:2, 0:5] ---
    # Top half A[0:2, 0:2]
    parts.append(_embed_subproblem(*_strassen_2x2(), 2, 2, 2, 0, 0, 0, n, m, p))
    parts.append(_embed_subproblem(*_strassen_2x2(), 2, 2, 2, 0, 0, 2, n, m, p))
    parts.append(_embed_subproblem(*_schoolbook(2, 2, 1), 2, 2, 1, 0, 0, 4, n, m, p))
    # Bottom half A[2:4, 0:2]
    parts.append(_embed_subproblem(*_strassen_2x2(), 2, 2, 2, 2, 0, 0, n, m, p))
    parts.append(_embed_subproblem(*_strassen_2x2(), 2, 2, 2, 2, 0, 2, n, m, p))
    parts.append(_embed_subproblem(*_schoolbook(2, 2, 1), 2, 2, 1, 2, 0, 4, n, m, p))

    # --- Block 2: A[0:4, 2:3] * B[2:3, 0:5] ---
    parts.append(_embed_subproblem(*_schoolbook(4, 1, 5), 4, 1, 5, 0, 2, 0, n, m, p))

    # --- Block 3: A[0:4, 3:5] * B[3:5, 0:5] ---
    # Top half A[0:2, 3:5]
    parts.append(_embed_subproblem(*_strassen_2x2(), 2, 2, 2, 0, 3, 0, n, m, p))
    parts.append(_embed_subproblem(*_strassen_2x2(), 2, 2, 2, 0, 3, 2, n, m, p))
    parts.append(_embed_subproblem(*_schoolbook(2, 2, 1), 2, 2, 1, 0, 3, 4, n, m, p))
    # Bottom half A[2:4, 3:5]
    parts.append(_embed_subproblem(*_strassen_2x2(), 2, 2, 2, 2, 3, 0, n, m, p))
    parts.append(_embed_subproblem(*_strassen_2x2(), 2, 2, 2, 2, 3, 2, n, m, p))
    parts.append(_embed_subproblem(*_schoolbook(2, 2, 1), 2, 2, 1, 2, 3, 4, n, m, p))

    U, V, W = _multi_concat(parts, n, m, p)
    return U, V, W


def _verify_np(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification using numpy."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3)
    return np.allclose(T, np.ones((n, m, p)), atol=tol)


def construct():
    """Construct the tensor decomposition for 4x5x5 matrix multiplication."""
    t0 = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5

    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)

    try:
        U, V, W = _construct_4x5x5()
        rank = len(U[0])
        if rank < n * m * p:
            if _verify_np(U, V, W, n, m, p):
                if time.time() - t0 < _DEADLINE_SECONDS - 5:
                    return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    return {"U": U_school, "V": V_school, "W": W_school}


# EVOLVE-BLOCK-END
