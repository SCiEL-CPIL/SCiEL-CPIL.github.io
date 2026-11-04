"""Rewritten program for matrix multiplication 3x4x7 tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _check_decomposition_fast(U, V, W, n, m, p):
    """Check if U,V,W (as lists of lists of ints) exactly decompose the matmul tensor.
    Returns True if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    return False
    return True

def _check_decomposition_numpy(U_np, V_np, W_np, n, m, p):
    """Fast numpy check. U_np: (n*m, rank), V_np: (m*p, rank), W_np: (p*n, rank).
    Returns True if all (i,j,k) sums equal 1.
    """
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                total = np.dot(U_np[u_idx, :], V_np[v_idx, :] * W_np[w_idx, :])
                if total != 1:
                    return False
    return True

def _try_random_integers(n, m, p, target_rank, deadline, seed_offset=0):
    """Try random integer entries in {-1, 0, 1} or small range."""
    np.random.seed(42 + seed_offset)
    for attempt in range(200):
        if time.time() > deadline:
            return None
        # Random entries from {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(n*m, target_rank))
        V = np.random.randint(-1, 2, size=(m*p, target_rank))
        W = np.random.randint(-1, 2, size=(p*n, target_rank))
        if _check_decomposition_numpy(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _try_random_small_ints(n, m, p, target_rank, deadline, seed_offset=0, lo=-2, hi=3):
    """Try random integer entries in [lo, hi]."""
    np.random.seed(100 + seed_offset)
    for attempt in range(100):
        if time.time() > deadline:
            return None
        U = np.random.randint(lo, hi, size=(n*m, target_rank))
        V = np.random.randint(lo, hi, size=(m*p, target_rank))
        W = np.random.randint(lo, hi, size=(p*n, target_rank))
        if _check_decomposition_numpy(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _try_structured_block(n, m, p, target_rank, deadline):
    """Try a structured approach: partition columns into blocks corresponding to
    sub-multiplications, with some overlap to reduce rank.
    
    Idea: The matmul tensor T[i,j,k] = 1 for all i,j,k.
    Schoolbook: one column per (i,j,k).
    
    Alternative: For each j, we have a sub-tensor T_i_k = 1 for all i,k.
    This is an outer product of ones vectors: T_i_k = 1_i * 1_k.
    So for fixed j, the sub-tensor over (i,k) has rank 1!
    
    This means: for each j (0..m-1), we can use ONE column that covers all (i,k) pairs
    for that j. That gives rank = m = 4.
    
    Wait, but the tensor is T[i,j,k] and the decomposition is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    For a single column c: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    If we set for column c_j (for each j):
    U[i*m+j, c_j] = 1 for all i
    V[j*p+k, c_j] = 1 for all k  
    W[k*n+i, c_j] = 1 for all i,k
    
    Then for a given (i,j,k):
    sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    = U[i*m+j, c_j]*V[j*p+k, c_j]*W[k*n+i, c_j]  (only c_j contributes for this j)
    = 1*1*1 = 1
    
    But wait - for a different j', column c_j' would have:
    U[i*m+j', c_j'] = 1, but U[i*m+j, c_j'] = 0 (since j != j')
    So only the column matching j contributes. 
    
    This gives rank = m = 4! Let me verify this is correct.
    
    Actually wait - I need to be more careful. Let me re-examine.
    
    Column c_j for j in range(m):
    - U[i*m + j, c_j] = 1 for all i in range(n)
    - U[i*m + j', c_j] = 0 for j' != j
    - V[j*p + k, c_j] = 1 for all k in range(p)
    - V[j'*p + k, c_j] = 0 for j' != j
    - W[k*n + i, c_j] = 1 for all i, k
    
    Then for tensor entry (i, j, k):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For c = c_j: U[i*m+j, c_j] = 1, V[j*p+k, c_j] = 1, W[k*n+i, c_j] = 1, product = 1
    For c = c_{j'} with j' != j: U[i*m+j, c_{j'}] = 0, so product = 0
    
    Total = 1. 
    
    So the rank is m = 4! This is a valid decomposition!
    """
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        c = j  # column index
        for i in range(n):
            U[i*m + j][c] = 1
        for k in range(p):
            V[j*p + k][c] = 1
        for i in range(n):
            for k in range(p):
                W[k*n + i][c] = 1
    
    # Verify
    if _check_decomposition_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def _try_structured_block_transposed(n, m, p, target_rank, deadline):
    """Alternative structured approach: group by i instead of j.
    For each i, use one column covering all (j,k).
    This gives rank = n = 3.
    
    Column c_i for i in range(n):
    - U[i*m + j, c_i] = 1 for all j
    - U[i'*m + j, c_i] = 0 for i' != i
    - V[j*p + k, c_i] = 1 for all j,k
    - W[k*n + i, c_i] = 1 for all k
    - W[k*n + i', c_i] = 0 for i' != i
    
    For tensor entry (i, j, k):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For c = c_i: U=1, V=1, W=1, product=1
    For c = c_{i'} with i' != i: U[i*m+j, c_{i'}] = 0, product = 0
    
    Total = 1. Rank = n = 3!
    """
    rank = n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        c = i
        for j in range(m):
            U[i*m + j][c] = 1
        for j in range(m):
            for k in range(p):
                V[j*p + k][c] = 1
        for k in range(p):
            W[k*n + i][c] = 1
    
    if _check_decomposition_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def _try_structured_block_k(n, m, p, target_rank, deadline):
    """Group by k: for each k, use one column covering all (i,j).
    This gives rank = p = 7.
    
    Column c_k for k in range(p):
    - U[i*m + j, c_k] = 1 for all i,j
    - V[j*p + k, c_k] = 1 for all j
    - V[j*p + k', c_k] = 0 for k' != k
    - W[k*n + i, c_k] = 1 for all i
    
    For tensor entry (i, j, k):
    For c = c_k: U=1, V=1, W=1, product=1
    For c = c_{k'} with k' != k: V[j*p+k, c_{k'}] = 0, product = 0
    
    Total = 1. Rank = p = 7.
    """
    rank = p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        c = k
        for i in range(n):
            for j in range(m):
                U[i*m + j][c] = 1
        for j in range(m):
            V[j*p + k][c] = 1
        for i in range(n):
            W[k*n + i][c] = 1
    
    if _check_decomposition_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def _try_combined_structured(n, m, p, deadline):
    """Try to combine the best structured decompositions.
    The minimum of the three structured approaches is min(n, m, p) = min(3, 4, 7) = 3.
    So rank 3 should work via the i-grouping approach.
    
    Let's verify rank 3 works and return it.
    """
    # Try rank = 3 (group by i)
    result = _try_structured_block_transposed(n, m, p, 3, deadline)
    if result is not None:
        return result
    
    # Try rank = 4 (group by j)
    result = _try_structured_block(n, m, p, 4, deadline)
    if result is not None:
        return result
    
    # Try rank = 7 (group by k)
    result = _try_structured_block_k(n, m, p, 7, deadline)
    if result is not None:
        return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 145.0  # 5s safety margin
    
    # Schoolbook fallback
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try structured decompositions (these should find very low rank solutions)
    structured = _try_combined_structured(n, m, p, DEADLINE)
    if structured is not None:
        rank = len(structured["U"][0])
        if rank < best_rank:
            best_rank = rank
            best_result = structured
    
    # Verify the structured result
    if best_rank < n * m * p:
        if _check_decomposition_fast(best_result["U"], best_result["V"], best_result["W"], n, m, p):
            return best_result
        else:
            # Shouldn't happen, but fallback
            best_result = schoolbook
            best_rank = n * m * p
    
    # If structured didn't work (shouldn't happen), try random search as backup
    if time.time() < DEADLINE - 10:
        for target_rank in range(best_rank, min(best_rank + 10, n*m*p)):
            if time.time() > DEADLINE - 5:
                break
            result = _try_random_integers(n, m, p, target_rank, DEADLINE - 5)
            if result is not None:
                best_rank = target_rank
                best_result = result
                break
    
    if time.time() > DEADLINE:
        return schoolbook
    
    return best_result

# EVOLVE-BLOCK-END
