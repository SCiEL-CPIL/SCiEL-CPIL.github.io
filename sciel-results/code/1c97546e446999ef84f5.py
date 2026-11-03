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
    Attempt to find a lower-rank decomposition using algebraic structure.
    """
    R0 = U.shape[1]
    if R0 <= max_rank:
        return U, V, W

    n, m, p = DIMENSIONS
    U_dim, V_dim, W_dim = n*m, m*p, p*n

    # For 2x4x6, try to construct a more efficient decomposition.
    # Key insight: we can use the structure that n=2 to create
    # shared "row sum" and "row diff" terms across all B column groups.
    
    # Build a decomposition using:
    # For each pair of B columns (2j, 2j+1), j=0,1,2:
    #   s0 = A[0,2j] + A[0,2j+1], d0 = A[0,2j] - A[0,2j+1]
    #   s1 = A[1,2j] + A[1,2j+1], d1 = A[1,2j] - A[1,2j+1]
    #   sd = (A[0,2j]+A[1,2j]) + (A[0,2j+1]+A[1,2j+1])  [row sum]
    #   dd = (A[0,2j]-A[1,2j]) + (A[0,2j+1]-A[1,2j+1])  [row diff]
    # Then use Strassen-like combinations with B columns.
    #
    # Actually, let's try a direct construction with rank 36:
    # Use 6 groups of 6 terms each, where each group handles 2 B columns
    # and uses a 2x2x2 Strassen decomposition but with shared structure.
    
    # Let's try: for each of the 3 column pairs, use rank 7 (Strassen)
    # but try to merge the "common" parts.
    
    # Alternative: try to find if rank 40 or 41 works by checking
    # linear dependencies in the 42-term decomposition.
    
    # Flatten the tensor and check rank
    T_flat = T.reshape(-1)
    
    # Build the "term matrix" where each column is a flattened rank-1 term
    term_matrix = np.zeros((T_flat.size, R0))
    for r in range(R0):
        term_matrix[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).reshape(-1)
    
    # Check the rank of this matrix
    # If rank < R0, we have linear dependencies and can reduce
    
    # Use SVD to find the effective rank
    try:
        if time.time() < deadline - 10:
            sv = np.linalg.svd(term_matrix, compute_uv=False)
            # Find the smallest rank that still captures the tensor
            tol = 1e-10
            effective_rank = np.sum(sv > tol)
            
            if effective_rank < R0:
                # We can reduce! Use the top effective_rank components
                # But this gives a least-squares solution, not exact
                # For exact reduction, we need to find which terms to drop
                pass
    except:
        pass
    
    # For now, return the original if no clear reduction found
    # The 42-rank decomposition is correct and verified
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

    # Attempt rank reduction.
    U, V, W = _try_reduce_rank(U, V, W, T, 41, deadline)

    # Convert to lists of ints/half-ints as required by the interface.
    def to_list(arr):
        return [[(int(x) if float(x).is_integer() else float(x)) for x in row]
                for row in arr.tolist()]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

# EVOLVE-BLOCK-END
