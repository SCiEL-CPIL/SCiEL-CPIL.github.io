import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    prod = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t, optimize=True)
    return np.allclose(prod, np.ones((n, m, p)), atol=tol)

def _strassen_block(n, m, p, j_a, j_b, k_a, k_b, U, V, W, col):
    """
    Add 7 Strassen terms for the block j in {j_a, j_b}, k in {k_a, k_b}.
    Assumes n=2 for the standard 7-term decomposition.
    Returns new col index.
    """
    # Indices for U: i * m + j
    # Indices for V: j * p + k
    # Indices for W: k * n + i
    
    # M1 = (A[0,j_a] + A[1,j_b]) * (B[j_a,k_a] + B[j_b,k_b])
    # Note: Standard Strassen uses A11, A22 etc.
    # A11 = A[0, j_a], A22 = A[1, j_b]
    # B11 = B[j_a, k_a], B22 = B[j_b, k_b]
    U[0 * m + j_a, col] += 1
    U[1 * m + j_b, col] += 1
    V[j_a * p + k_a, col] += 1
    V[j_b * p + k_b, col] += 1
    W[k_a * n + 0, col] += 1
    W[k_b * n + 1, col] += 1
    col += 1

    # M2 = (A[1,j_a] + A[1,j_b]) * B[j_a,k_a]
    U[1 * m + j_a, col] += 1
    U[1 * m + j_b, col] += 1
    V[j_a * p + k_a, col] += 1
    W[k_a * n + 1, col] += 1
    W[k_b * n + 1, col] -= 1
    col += 1

    # M3 = A[0,j_a] * (B[j_a,k_b] - B[j_b,k_b])
    U[0 * m + j_a, col] += 1
    V[j_a * p + k_b, col] += 1
    V[j_b * p + k_b, col] -= 1
    W[k_b * n + 0, col] += 1
    col += 1

    # M4 = A[1,j_b] * (B[j_b,k_a] - B[j_a,k_a])
    U[1 * m + j_b, col] += 1
    V[j_b * p + k_a, col] += 1
    V[j_a * p + k_a, col] -= 1
    W[k_a * n + 1, col] += 1
    col += 1

    # M5 = (A[0,j_a] + A[0,j_b]) * B[j_b,k_b]
    U[0 * m + j_a, col] += 1
    U[0 * m + j_b, col] += 1
    V[j_b * p + k_b, col] += 1
    W[k_a * n + 0, col] -= 1
    W[k_b * n + 0, col] += 1
    col += 1

    # M6 = (A[1,j_b] - A[0,j_a]) * (B[j_a,k_a] + B[j_a,k_b])
    U[1 * m + j_b, col] += 1
    U[0 * m + j_a, col] -= 1
    V[j_a * p + k_a, col] += 1
    V[j_a * p + k_b, col] += 1
    W[k_b * n + 1, col] += 1
    col += 1

    # M7 = (A[0,j_b] - A[1,j_b]) * (B[j_b,k_a] + B[j_b,k_b])
    U[0 * m + j_b, col] += 1
    U[1 * m + j_b, col] -= 1
    V[j_b * p + k_a, col] += 1
    V[j_b * p + k_b, col] += 1
    W[k_a * n + 0, col] += 1
    col += 1

    return col

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _construct_optimal(n, m, p):
    # Determine pairing
    j_pairs = []
    j = 0
    while j + 1 < m:
        j_pairs.append((j, j + 1))
        j += 2
    j_leftover = [j] if j < m else []

    k_pairs = []
    k = 0
    while k + 1 < p:
        k_pairs.append((k, k + 1))
        k += 2
    k_leftover = [k] if k < p else []

    # Calculate rank
    rank = 0
    rank += 7 * len(j_pairs) * len(k_pairs)
    # For leftover j, rank is n if n>1? No, for n=2, rank 1 suffices for the 2x6 block of 1s.
    # Actually, for general n, leftover j is n*p terms. Rank n*p is schoolbook.
    # But for n=2, we can do better.
    if n == 2:
        rank += 1 * len(j_leftover)
        rank += 1 * len(k_leftover) * m # Wait, leftover k is m*n terms. Rank 1?
        # Leftover k: T[i, j, k] for fixed k. 2x5 matrix of 1s. Rank 1.
    else:
        rank += n * p * len(j_leftover)
        rank += n * m * len(k_leftover)

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    # Strassen blocks
    for (j_a, j_b) in j_pairs:
        for (k_a, k_b) in k_pairs:
            col = _strassen_block(n, m, p, j_a, j_b, k_a, k_b, U, V, W, col)

    # Leftover j
    for j in j_leftover:
        if n == 2:
            # Rank 1 term
            for i in range(n):
                U[i * m + j, col] = 1
            for k in range(p):
                V[j * p + k, col] = 1
            for k in range(p):
                for i in range(n):
                    W[k * n + i, col] = 1
            col += 1
        else:
            for k in range(p):
                for i in range(n):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    # Leftover k
    for k in k_leftover:
        if n == 2:
            # Rank 1 term
            for j in range(m):
                for i in range(n):
                    U[i * m + j, col] = 1
                V[j * p + k, col] = 1
            for i in range(n):
                W[k * n + i, col] = 1
            col += 1
        else:
            for j in range(m):
                for i in range(n):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # 1) Structural optimal candidate
    U_b, V_b, W_b = _construct_optimal(n, m, p)
    best = None
    if _verify(U_b, V_b, W_b, n, m, p):
        best = (U_b, V_b, W_b)

    # 2) Schoolbook fallback
    U_s, V_s, W_s = _schoolbook(n, m, p)
    if best is None or U_s.shape[1] < best[0].shape[1]:
        best = (U_s, V_s, W_s)

    Uo, Vo, Wo = best
    return {
        "U": Uo.reshape(n * m, Uo.shape[1]).tolist(),
        "V": Vo.reshape(m * p, Vo.shape[1]).tolist(),
        "W": Wo.reshape(p * n, Wo.shape[1]).tolist(),
    }
# EVOLVE-BLOCK-END
