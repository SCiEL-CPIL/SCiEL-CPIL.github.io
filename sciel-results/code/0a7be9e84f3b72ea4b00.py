"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

The multiplication tensor T has entries T[i,j,k] = 1 at positions (i*m+j, j*p+k, k*n+i)
for i in [0,n), j in [0,m), k in [0,p). The schoolbook rank is n*m*p = 63.

This program attempts to find a lower-rank exact decomposition using algebraic
search techniques, falling back to the schoolbook decomposition if no
improvement is found within the time budget.
"""

import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]
START_TIME = time.time()
TIME_LIMIT = 140  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - START_TIME


def _deadline_exceeded():
    return _elapsed() > TIME_LIMIT


def _schoolbook_decomposition(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of the multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute the decomposed tensor: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Compare against the true multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # True tensor value: 1 if (i,j,k) is a valid triple, else 0
                true_val = 1.0
                # Decomposed value: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                decomp_val = float(np.sum(
                    U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :]
                ))
                if abs(decomp_val - true_val) > 1e-10:
                    return False
                # Also check that non-multiplication entries are zero
                # (entries where the indices don't form a valid (i,j,k) triple)
    
    # Check that entries which should be zero are indeed zero
    # The tensor is zero except at positions (i*m+j, j*p+k, k*n+i) for valid i,j,k
    # We need to verify no spurious non-zero entries
    # This is expensive for large ranks, so we do a spot check
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        # For each nonzero pattern in u, v, w, check consistency
        nz_u = np.where(u_col != 0)[0]
        nz_v = np.where(v_col != 0)[0]
        nz_w = np.where(w_col != 0)[0]
        for a in nz_u:
            for b in nz_v:
                for c in nz_w:
                    # a = i*m+j, b = j'*p+k, c = k'*n+i'
                    # For this to contribute to the tensor, we need j == j' and k == k' and i == i'
                    # i = a // m, j = a % m
                    # j2 = b // p, k = b % p
                    # k2 = c // n, i2 = c % n
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (j == j2 and k == k2 and i == i2):
                        # This combination should not contribute
                        if abs(u_col[a] * v_col[b] * w_col[c]) > 1e-10:
                            return False
    return True


def _try_strassen_style(n, m, p):
    """
    Attempt to find a rank-reducing decomposition using algebraic methods.
    
    For the 3x3x3 case, Strassen achieves rank 7. We explore whether similar
    rank reductions are possible for 3x3x7 by exploiting the tensor structure.
    
    The tensor decomposes as a direct sum over j of 3 sub-tensors T_j, each
    representing 3x1x7 multiplication. If we can find rank-r decompositions
    for each T_j that share structure, we might reduce the total rank.
    
    However, the most promising approach is to use known low-rank identities
    and verify them.
    """
    # For now, return None to indicate no improvement found
    # A full algebraic search would be very complex
    return None


def _random_search_for_improvement(n, m, p, max_rank, max_iter=500):
    """
    Random search for a rank < max_rank decomposition.
    This is unlikely to succeed for large gaps but worth trying.
    """
    if _deadline_exceeded():
        return None
    
    for rank in range(max_rank - 1, 1, -1):
        if _deadline_exceeded():
            return None
        for attempt in range(max_iter):
            if _deadline_exceeded():
                return None
            # Random rank-1 term generation
            U = np.random.randint(-2, 3, size=(n * m, rank))
            V = np.random.randint(-2, 3, size=(m * p, rank))
            W = np.random.randint(-2, 3, size=(p * n, rank))
            
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None


# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Attempts to find a rank below the schoolbook rank of 63, falling back to
    the schoolbook decomposition if no improvement is found.
    
    Returns:
        dict with keys 'U', 'V', 'W' as lists of lists (integer entries).
        U: (n*m, rank) = (9, rank)
        V: (m*p, rank) = (21, rank)
        W: (p*n, rank) = (21, rank)
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 63
    
    # Start with the guaranteed-valid schoolbook decomposition
    best = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    # Attempt algebraic rank reduction
    if not _deadline_exceeded():
        improved = _try_strassen_style(n, m, p)
        if improved is not None:
            best = improved
            best_rank = len(improved["U"][0])
    
    # Attempt random search for further improvement (limited)
    if not _deadline_exceeded() and best_rank > 20:
        improved = _random_search_for_improvement(n, m, p, best_rank, max_iter=100)
        if improved is not None:
            best = improved
            best_rank = len(improved["U"][0])
    
    return best
# EVOLVE-BLOCK-END
