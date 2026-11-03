"""Program for matrix multiplication 2x3 by 3x5 using tensor code / Berlekamp-Massey approach."""
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _bm(sequence):
    """Berlekamp-Massey: returns minimal linear recurrence coefficients [c0, c1, ..., c_{L-1}]
    such that s[i] = sum_{j=1}^{L} c[j-1] * s[i-j] for i >= L."""
    if len(sequence) == 0:
        return []
    L = 0
    C = [0] * (len(sequence) + 1)
    C[0] = 1
    B = [0] * (len(sequence) + 1)
    B[0] = 1
    m = 1
    b = sequence[0]
    if b == 0:
        for idx in range(len(sequence)):
            if sequence[idx] != 0:
                b = sequence[idx]
                m = idx + 1
                break
    for n_idx in range(len(sequence)):
        # Compute discrepancy
        d = sequence[n_idx]
        for j in range(1, L + 1):
            d += C[j] * sequence[n_idx - j]
        if d == 0:
            m += 1
            continue
        T = C[:]
        coef = d / b if b != 0 else 0
        for j in range(m, len(C)):
            C[j] = C[j] - coef * B[j - m]
        if 2 * L <= n_idx:
            L = n_idx + 1 - L
            B = T
            b = d
            m = 1
        else:
            m += 1
    return C[1:L + 1]


def _rank_of_matrix(M):
    """Compute rank of a matrix using SVD (robust to floating point)."""
    if M.size == 0:
        return 0
    M = np.asarray(M, dtype=float)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    sv = np.linalg.svd(M, compute_uv=False)
    tol = max(M.shape) * np.finfo(float).eps * (sv[0] if len(sv) > 0 else 1.0)
    return int(np.sum(sv > tol))


def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 6
    V_dim = m * p  # 15
    W_dim = p * n  # 10

    # Build the matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # Flattened views:
    #   A: (i,j) x k  -> shape (6, 5)
    #   B: i x (j,k)  -> shape (2, 15)
    #   C: (i,k) x j  -> shape (10, 3)
    #
    # The tensor rank r must satisfy:
    #   r >= rank(A) = 5 (since A is all-ones 6x5, rank = 5)
    #   r >= rank(B) = 2 (since B is all-ones 2x15, rank = 1 actually)
    #   r >= rank(C) = 3 (since C is all-ones 10x3, rank = 1 actually)
    #
    # Wait, let me re-examine. The matmul tensor T[i,j,k] = 1 for all i,j,k.
    # A[i*m+j, k] = T[i,j,k] = 1 for all i,j,k -> A is all-ones 6x5 -> rank 1
    # B[i, j*p+k] = T[i,j,k] = 1 for all i,j,k -> B is all-ones 2x15 -> rank 1
    # C[k*n+i, j] = T[i,j,k] = 1 for all i,j,k -> C is all-ones 10x3 -> rank 1
    #
    # So the flattening rank lower bound is 1, which is useless.
    # The actual tensor rank is determined by the multilinear structure.
    #
    # For standard matmul_{n,m,p}, the rank is known to be n*m*p in general,
    # but for small cases it can be lower. For 2x3x5, let's try to find
    # a decomposition of rank < 30.
    #
    # Key insight: The tensor T[i,j,k] = 1 for all (i,j,k) in [0,n)x[0,m)x[0,p)
    # is actually the ALL-ONES tensor of shape (n, m, p)!
    #
    # The all-ones tensor of shape (n,m,p) has rank 1!
    # T = ones(n) ⊗ ones(m) ⊗ ones(p)
    #
    # So the rank-1 decomposition is:
    #   u = ones(n*m), v = ones(m*p), w = ones(p*n)
    #
    # Let's verify: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1*1*1 = 1 for all (i,j,k). ✓
    #
    # But wait - this is the tensor T[i,j,k] = 1 for ALL i,j,k, which is indeed
    # the matmul tensor since C[i,k] = sum_j A[i,j]*B[j,k] and the tensor encodes
    # the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.
    #
    # So the rank is 1!

    rank = 1
    U = [[1]] * U_dim
    V = [[1]] * V_dim
    W = [[1]] * W_dim

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
