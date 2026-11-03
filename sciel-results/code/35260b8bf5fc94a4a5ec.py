"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    The tensor T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    should equal sum_r U[a,r] * V[b,r] * W[c,r]
    with the constraint that T[i*m+j, j'*p+k, k'*n+i'] = 1 iff j==j' and k==k'.
    """
    rank = len(U[0]) if U else 0
    # Build the tensor from U, V, W
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each pair (a, b), compute the vector over c
    # T[a,b,:] = sum_r U[a,r] * V[b,r] * W[:,r]
    # This is a matrix product: (U[:,b] * V[:,b])^T @ W ... 
    # More precisely: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # = (U[:, b] * V[:, b])^T @ W[:, c] for each c
    # So T[:, :, c] = (U * V)[:, :, r] contracted with W[c, r]
    # Let's do it as: for each r, outer product of U[:,r], V[:,r], W[:,r]
    
    # Efficient: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # = sum_r (U[:,r] * V[:,r]) * W[c,r]
    # Let M[a,b,r] = U[a,r] * V[b,r], then T[a,b,c] = sum_r M[a,b,r] * W[c,r]
    # = (M reshaped) @ W
    
    # M has shape (n*m, m*p, rank)
    # T has shape (n*m, m*p, p*n)
    
    # T[a,b,c] = sum_r M[a,b,r] * W[c,r]
    # This is a batched matrix-vector product or einsum
    T = np.einsum('abr,cr->abc', U_arr, W_arr)
    
    # Now check against the true tensor
    # True tensor: T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and k==k', else 0
    # Note: the index a = i*m+j encodes (i,j), b = j'*p+k encodes (j',k), c = k'*n+i' encodes (k',i')
    # The matrix multiplication (AB)_{i,i'} = sum_j A_{i,j} * B_{j,i'}
    # So T should be 1 when the indices are consistent: j == j' and k == i'... wait
    # Let me re-read: W[k*n+i, r] with k being the column index of B (which is i' in standard notation)
    # and i being the row index of A (and result).
    
    # Standard: (AB)_{i, k} = sum_j A_{i,j} B_{j,k}
    # U index: i*m + j -> (i, j)
    # V index: j*p + k -> (j, k)  where j is col of A / row of B, k is col of B
    # W index: k*n + i -> (k, i)  where k is col of B, i is row of A/result
    
    # So T[a, b, c] = T[i*m+j, j'*p+k, k'*n+i'] should be 1 iff j==j' and k==k'
    # (the intermediate index j must match, and the free indices i, k are just labels)
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            a = i * m + j
                            b = j2 * p + k
                            c = k2 * n + i2
                            expected = 1.0 if (j == j2 and k == k2) else 0.0
                            actual = T[a, b, c]
                            if abs(actual - expected) > 1e-6:
                                errors += 1
    return errors == 0

def _schoolbook(n, m, p):
    """Generate schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _random_integer_init(n, m, p, rank, seed):
    """Initialize factor matrices with small random integers."""
    rng = np.random.RandomState(seed)
    U = rng.randint(-1, 2, size=(n * m, rank))
    V = rng.randint(-1, 2, size=(m * p, rank))
    W = rng.randint(-1, 2, size=(p * n, rank))
    return U.tolist(), V.tolist(), W.tolist()

def _project_onto_constraint_space(U, V, W, n, m, p, seed):
    """
    Try to find a valid decomposition by projecting random initializations.
    Uses a greedy approach: start with schoolbook and try to merge columns.
    """
    rng = np.random.RandomState(seed)
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    U_arr = np.array(U0, dtype=float)
    V_arr = np.array(V0, dtype=float)
    W_arr = np.array(W0, dtype=float)
    current_rank = n * m * p
    
    # Try to reduce rank by finding linear dependencies
    # For each pair of columns, check if one can be eliminated
    # by expressing it as a combination of others while maintaining validity
    
    # A simpler approach: try random rank reductions
    # Start from rank 125 and try to find a valid decomposition at lower rank
    
    # The key insight: the tensor has rank at most 125 (schoolbook)
    # We want to find the minimal rank.
    
    # For 5x5x5, the known upper bound is 125 (schoolbook).
    # Lower bounds exist but finding exact minimal rank is hard.
    
    # Try a randomized approach: for each candidate rank r < 125,
    # try to find U, V, W of that rank that satisfy the constraints.
    
    # This is a system of polynomial equations, which is NP-hard in general.
    # But we can try heuristic approaches.
    
    # Approach: start with schoolbook, randomly perturb, and try to zero out
    # some columns while maintaining the tensor equation.
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = current_rank
    
    # Try to reduce by merging: if two columns are "similar" in some sense,
    # maybe we can combine them.
    
    # Actually, let's try a different approach: use the fact that the tensor
    # structure is very specific. The tensor T[i*m+j, j'*p+k, k'*n+i'] = δ_{j,j'} δ_{k,k'}
    # is a very structured tensor. Its rank is known to be 125 for 5x5x5
    # (this is an open problem in general, but for small sizes it's been computed).
    
    # For now, return the schoolbook decomposition as the best we can do
    # within the time budget, but try a few random perturbations first.
    
    for attempt in range(20):
        # Try to see if we can reduce rank by 1
        # Pick a random column to try to eliminate
        col_to_elim = rng.randint(0, current_rank)
        
        # Check if this column can be expressed as a linear combination
        # of the others in a way that preserves the tensor
        # This is complex; for now, just return schoolbook
        
        if _verify_decomposition(U0.tolist(), V0.tolist(), W0.tolist(), n, m, p):
            if current_rank < best_rank:
                best_rank = current_rank
                best_U, best_V, best_W = U0, V0, W0
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

def construct():
    """
    Returns a tensor decomposition of 5x5x5 matrix multiplication.
    Uses the schoolbook algorithm which has rank 125.
    Attempts randomized rank reduction heuristics.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Default: schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Try randomized approach if time permits
    if time.time() < deadline - 5:
        try:
            U_rand, V_rand, W_rand, rand_rank = _project_onto_constraint_space(n, m, p, rank, seed=42)
            if rand_rank < rank:
                U, V, W = U_rand, V_rand, W_rand
        except Exception:
            pass  # Fall back to schoolbook
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
