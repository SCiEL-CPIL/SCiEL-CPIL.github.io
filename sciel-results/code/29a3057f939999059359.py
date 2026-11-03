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


def _construct_rank36(n, m, p):
    """
    Construct a rank-36 decomposition exploiting n=2 structure.
    
    For 2x4 * 4x6, we compute C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, k in {0..5}.
    
    Key idea: For each pair of output columns (2t, 2t+1) with t=0,1,2,
    we use 6 terms that compute both C[0,2t], C[0,2t+1], C[1,2t], C[1,2t+1]
    by exploiting the 2-row structure.
    
    For columns 2t and 2t+1 of B, define:
      For each row i of A (i=0,1) and each column j of A (j=0..3):
        We need A[i,j] * B[j,2t] and A[i,j] * B[j,2t+1]
    
    Use the following 6 terms per column-pair (t=0,1,2):
    Term 1: U = [A[0,2t], A[0,2t+1], 0, 0], V = [B[0,2t], B[0,2t+1], B[1,2t], B[1,2t+1], 0, 0, 0, 0, 0, 0, 0, 0], W = [C[0,2t], 0, C[0,2t+1], 0]
    ... this doesn't work simply.
    
    Let me try a different approach: use 6 terms per column-pair where each term
    handles a specific (i, j) pattern.
    
    Actually, let's use a cleaner construction:
    For each t in {0,1,2} (column pair), for each s in {0,1} (row of A):
      Use 3 terms to compute C[s, 2t] and C[s, 2t+1]:
        Term a: U has A[s, 2t] at position (s*4+2t), V has B[0,2t],B[0,2t+1] at positions (0*6+2t),(0*6+2t+1), W has C[s,2t] at (2t*2+s), C[s,2t+1] at ((2t+1)*2+s)
        ...
    
    This is getting complex. Let me just try rank 42 and see if we can find
    linear dependencies via SVD.
    """
    # Fall back to the Strassen construction
    return _construct_strassen_2x2x4(n, m, p)


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
    """
    strassen_terms = _strassen_2x2x2_terms()
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    j_groups = [0, 2]
    k_groups = [0, 2, 4]

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
    Attempt to find a lower-rank decomposition by checking for linear
    dependencies in the rank-42 decomposition and attempting to find
    a rank-36 or rank-38 decomposition via structured search.
    """
    R0 = U.shape[1]
    if R0 <= max_rank:
        return U, V, W

    n, m, p = DIMENSIONS
    U_dim, V_dim, W_dim = n * m, m * p, p * n

    # Check for linear dependencies in the current decomposition
    T_flat = T.reshape(-1)
    term_matrix = np.zeros((T_flat.size, R0))
    for r in range(R0):
        term_matrix[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).reshape(-1)

    try:
        if time.time() < deadline - 20:
            sv = np.linalg.svd(term_matrix, compute_uv=False)
            tol = 1e-10
            effective_rank = int(np.sum(sv > tol))
            
            if effective_rank < R0:
                # There are linear dependencies. Try to find a basis of size effective_rank.
                # Use the left singular vectors to find a minimal spanning set.
                U_svd, _, Vt_svd = np.linalg.svd(term_matrix, full_matrices=False)
                # The columns of term_matrix that are linearly independent
                # We can keep the first effective_rank columns if they span
                # But we need to verify exactness
                U_red = U[:, :effective_rank]
                V_red = V[:, :effective_rank]
                W_red = W[:, :effective_rank]
                if _verify(U_red, V_red, W_red, T):
                    return U_red, V_red, W_red
    except:
        pass

    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 140.0

    T = _build_tensor(n, m, p)

    # Build the rank-42 Strassen block decomposition.
    U, V, W = _construct_strassen_2x2x4(n, m, p)

    # Verify correctness.
    if not _verify(U, V, W, T):
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

    # Attempt rank reduction.
    U, V, W = _try_reduce_rank(U, V, W, T, 41, deadline)

    def to_list(arr):
        return [[(int(x) if float(x).is_integer() else float(x)) for x in row]
                for row in arr.tolist()]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

# EVOLVE-BLOCK-END
