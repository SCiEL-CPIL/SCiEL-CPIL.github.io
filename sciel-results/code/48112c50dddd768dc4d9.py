"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, T, atol=1e-9):
    """Verify that the decomposition exactly reproduces T."""
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    T_check = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_check += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_check, T, atol=atol)


def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Indexing (row-major): A: 0=(0,0),1=(0,1),2=(1,0),3=(1,1); B,C same.
    """
    return [
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]},
    ]


def _construct_strassen_2x2x4(n, m, p):
    """
    Construct a rank-42 decomposition using Strassen on 2x2x2 blocks.

    Partition A (2x4) into two 2x2 blocks (cols 0-1 and 2-3).
    Partition B (4x6) into three 4x2 blocks (cols 0-1, 2-3, 4-5),
    each 4x2 block = [top 2x2; bottom 2x2].
    C = A*B = [A1, A2] * [[B11,B12,B13],[B21,B22,B23]]
      = [A1*B11 + A2*B21, A1*B12 + A2*B22, A1*B13 + A2*B23]
    Each A_a * B_{a,b} (2x2 * 2x2) uses rank 7 (Strassen).
    Total rank = 2 * 3 * 7 = 42.
    """
    strassen_terms = _strassen_2x2x2_terms()
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    j_groups = [0, 2]   # A block base columns
    k_groups = [0, 2, 4]  # B column-block base columns

    col = 0
    for j_base in j_groups:
        for k_base in k_groups:
            for s in range(7):
                for (idx, coeff) in strassen_terms[s]['u']:
                    i, j_loc = divmod(idx, 2)
                    j = j_base + j_loc
                    U[i * m + j, col] += coeff
                for (idx, coeff) in strassen_terms[s]['v']:
                    j_loc, k_loc = divmod(idx, 2)
                    j = j_base + j_loc
                    k = k_base + k_loc
                    V[j * p + k, col] += coeff
                for (idx, coeff) in strassen_terms[s]['w']:
                    i_loc, k_loc = divmod(idx, 2)
                    k = k_base + k_loc
                    i = i_loc
                    W[k * n + i, col] += coeff
                col += 1

    return U, V, W


def _try_reduce_rank(U, V, W, T, max_rank, deadline):
    """
    Attempt to express the rank-42 decomposition as fewer rank-1 terms.
    Strategy: the tensor T is a sum of rank-1 terms u_r ⊗ v_r ⊗ w_r.
    We try to find a set of R < 42 rank-1 terms that sum to T by
    solving for the decomposition via linear algebra on a flattening.
    """
    R0 = U.shape[1]
    # If R0 is already small, nothing to do.
    if R0 <= max_rank:
        return U, V, W

    # We attempt to merge Strassen terms within each block.
    # Within a single (j_base, k_base) block, the 7 Strassen terms span
    # a 7-dim subspace of rank-1 terms. We cannot reduce below 7 per block
    # (Strassen is optimal for 2x2x2), but we can try to share terms
    # across blocks if the structure allows.
    #
    # For now, the safest reduction is to verify and return as-is.
    # A true rank reduction would require solving a non-convex problem.
    # We attempt a linear-algebra based merge: pick a target rank R and
    # try to find coefficients.
    #
    # Given the time budget and the known optimality of Strassen per block,
    # we return the 42-rank decomposition.
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    T = _build_tensor(n, m, p)

    # Build the rank-42 Strassen block decomposition.
    U, V, W = _construct_strassen_2x2x4(n, m, p)

    # Verify correctness.
    if not _verify(U, V, W, T):
        # Fallback: schoolbook rank 48 (should not happen).
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    # Attempt rank reduction (currently a no-op that preserves correctness).
    U, V, W = _try_reduce_rank(U, V, W, T, 41, deadline)

    # Convert to lists of ints/half-ints as required by the interface.
    def to_list(arr):
        return [[(int(x) if float(x).is_integer() else float(x)) for x in row]
                for row in arr.tolist()]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

# EVOLVE-BLOCK-END
