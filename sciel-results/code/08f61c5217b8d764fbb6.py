"""
Rank-minimization via sparse integer linear algebra: 
Construct the rank-44 Strassen+Schoolbook decomposition, identify the 
linear dependence among the 44 rank-1 terms using SVD on the coefficient 
matrix, and solve for the exact integer combination that eliminates one 
term to achieve rank 43.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _multiplication_tensor(n, m, p):
    """Build the exact multiplication tensor T with T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _multiplication_tensor(n, m, p)
    rank = U.shape[1]
    result = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(result, T, atol=tol)

def _build_strassen_2x4x4():
    """
    Strassen decomposition for 2x4x4 matrix multiplication.
    Returns U (8, 28), V (16, 28), W (8, 28) where W uses index k*2+i.
    """
    rank = 28
    U = np.zeros((8, rank))
    V = np.zeros((16, rank))
    W = np.zeros((8, rank))

    col = 0
    for s in range(7):
        if s == 0:  # M1 = (A11+A22)(B11+B22)
            X = {0: [(0,0,1),(1,2,1)], 1: [(0,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1),(2,0,1)], (0,1): [(0,1,1),(2,1,1)],
                 (1,0): [(1,0,1),(3,0,1)], (1,1): [(1,1,1),(3,1,1)]}
            out = {0: [(0,0,1),(1,2,1)], 1: [(0,1,1),(1,3,1)]}
        elif s == 1:  # M2 = (A21+A22)B11
            X = {0: [(1,0,1),(1,2,1)], 1: [(1,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1)], (0,1): [(0,1,1)], (1,0): [(1,0,1)], (1,1): [(1,1,1)]}
            out = {0: [(1,0,1),(1,2,-1)], 1: [(1,1,1),(1,3,-1)]}
        elif s == 2:  # M3 = A11(B12-B22)
            X = {0: [(0,0,1)], 1: [(0,1,1)]}
            Y = {(0,0): [(0,2,1),(2,2,-1)], (0,1): [(0,3,1),(2,3,-1)],
                 (1,0): [(1,2,1),(3,2,-1)], (1,1): [(1,3,1),(3,3,-1)]}
            out = {0: [(0,2,1),(1,2,1)], 1: [(0,3,1),(1,3,1)]}
        elif s == 3:  # M4 = A22(B21-B11)
            X = {0: [(1,2,1)], 1: [(1,3,1)]}
            Y = {(0,0): [(2,0,1),(0,0,-1)], (0,1): [(2,1,1),(0,1,-1)],
                 (1,0): [(3,0,1),(1,0,-1)], (1,1): [(3,1,1),(1,1,-1)]}
            out = {0: [(0,0,1),(1,0,1)], 1: [(0,1,1),(1,1,1)]}
        elif s == 4:  # M5 = (A11+A12)B22
            X = {0: [(0,0,1),(0,2,1)], 1: [(0,1,1),(0,3,1)]}
            Y = {(0,0): [(2,2,1)], (0,1): [(2,3,1)], (1,0): [(3,2,1)], (1,1): [(3,3,1)]}
            out = {0: [(0,0,-1),(0,2,1)], 1: [(0,1,-1),(0,3,1)]}
        elif s == 5:  # M6 = (A21-A11)(B11+B12)
            X = {0: [(1,0,1),(0,0,-1)], 1: [(1,1,1),(0,1,-1)]}
            Y = {(0,0): [(0,0,1),(0,2,1)], (0,1): [(0,1,1),(0,3,1)],
                 (1,0): [(1,0,1),(1,2,1)], (1,1): [(1,1,1),(1,3,1)]}
            out = {0: [(1,2,1)], 1: [(1,3,1)]}
        elif s == 6:  # M7 = (A12-A22)(B21+B22)
            X = {0: [(0,2,1),(1,2,-1)], 1: [(0,3,1),(1,3,-1)]}
            Y = {(0,0): [(2,0,1),(2,2,1)], (0,1): [(2,1,1),(2,3,1)],
                 (1,0): [(3,0,1),(3,2,1)], (1,1): [(3,1,1),(3,3,1)]}
            out = {0: [(0,0,1),(1,2,1)], 1: [(0,1,1),(1,3,1)]}

        for j in range(2):
            for k in range(2):
                for (i, a, coeff) in X[j]:
                    U[i * 4 + a, col] += coeff
                for (b, c, coeff) in Y[(j,k)]:
                    V[b * 4 + c, col] += coeff
                for (i, k_out, coeff) in out[k]:
                    W[k_out * 2 + i, col] += coeff
                col += 1

    return U, V, W

def _build_schoolbook_1x4x4():
    """Schoolbook for 1x4x4: rank 16. U (4,16), V (16,16), W (4,16)."""
    rank = 16
    U = np.zeros((4, rank))
    V = np.zeros((16, rank))
    W = np.zeros((4, rank))
    for j in range(4):
        for k in range(4):
            U[j, j*4+k] = 1
            V[j*4+k, j*4+k] = 1
            W[k, j*4+k] = 1
    return U, V, W

def _combine_3x4x4():
    """
    Combine 2x4x4 (Strassen, rank 28) and 1x4x4 (schoolbook, rank 16) -> rank 44.
    Returns U (12,44), V (16,44), W (12,44) with W index k*3+i.
    """
    U2, V2, W2 = _build_strassen_2x4x4()
    U1, V1, W1 = _build_schoolbook_1x4x4()

    total_rank = 44
    U = np.zeros((12, total_rank))
    V = np.zeros((16, total_rank))
    W = np.zeros((12, total_rank))

    # 2x4x4 part (cols 0-27)
    U[0:8, 0:28] = U2
    V[0:16, 0:28] = V2
    for i in range(2):
        for k in range(4):
            W[k*3+i, 0:28] = W2[k*2+i, :]

    # 1x4x4 part (cols 28-43)
    U[8:12, 28:44] = U1
    V[0:16, 28:44] = V1
    for k in range(4):
        W[k*3+2, 28:44] = W1[k, :]

    return U, V, W

def _find_linear_dependence(U, V, W, T, n, m, p):
    """
    Find a linear dependence among the rank-1 terms of the decomposition.
    Returns (coefficients, index_to_eliminate) or (None, None).
    
    We form the matrix M of shape (n*m*m*p*p*n, rank) where each column c is
    the flattened tensor u_c ⊗ v_c ⊗ w_c. Then we find the null space.
    """
    rank = U.shape[1]
    
    # Build the matrix of flattened rank-1 tensors
    # Shape: (n*m*m*p*p*n, rank)
    M = np.zeros((n * m * m * p * p * n, rank))
    
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        # Tensor product u ⊗ v ⊗ w
        tensor = np.einsum('i,j,k->ijk', u, v, w)
        M[:, c] = tensor.flatten()
    
    # Find the null space of M
    # M has shape (N, rank) where N = n*m*m*p*p*n
    # We want to find x such that M @ x = 0
    # Use SVD: M = U_svd @ S @ Vt
    try:
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None
    
    # The null space is spanned by the right singular vectors corresponding
    # to zero (or near-zero) singular values.
    # Since rank <= min(N, rank), and we expect rank=44 to be dependent,
    # there should be at least one zero singular value.
    
    # Find indices where S is close to zero
    tol = 1e-10
    zero_indices = np.where(S < tol)[0]
    
    if len(zero_indices) == 0:
        # Check if the smallest singular value is very small
        if S[-1] < 1e-8:
            zero_indices = [len(S) - 1]
        else:
            return None, None
    
    # Take the last right singular vector (corresponding to smallest singular value)
    # This gives us a linear dependence x such that M @ x ≈ 0
    x = Vt[-1, :]
    
    # Find the index with the largest absolute coefficient to eliminate
    # We want to solve for x[eliminate] in terms of the others
    # x[eliminate] = -sum_{c != eliminate} x[c] * x[eliminate] / x[eliminate]
    # Actually, the dependence is sum_c x[c] * t_c = 0
    # So t_eliminate = -sum_{c != eliminate} x[c] * t_c / x[eliminate]
    
    # Choose the index to eliminate: the one with the largest |x[c]|
    # to minimize numerical error
    eliminate_idx = np.argmax(np.abs(x))
    
    if abs(x[eliminate_idx]) < 1e-12:
        return None, None
    
    # The coefficients for the remaining terms
    coeffs = x.copy()
    
    return coeffs, eliminate_idx

def _reduce_rank(U, V, W, coeffs, eliminate_idx, n, m, p):
    """
    Given a linear dependence sum_c coeffs[c] * t_c = 0, eliminate the term
    at eliminate_idx by expressing it in terms of the others.
    
    t_eliminate = -sum_{c != eliminate} coeffs[c] * t_c / coeffs[eliminate]
    
    This means we can remove column eliminate_idx and adjust the other columns.
    """
    rank = U.shape[1]
    new_rank = rank - 1
    
    # New factors without the eliminated column
    U_new = np.zeros((n * m, new_rank))
    V_new = np.zeros((m * p, new_rank))
    W_new = np.zeros((p * n, new_rank))
    
    # Map old indices to new indices
    old_to_new = {}
    new_idx = 0
    for c in range(rank):
        if c != eliminate_idx:
            old_to_new[c] = new_idx
            new_idx += 1
    
    # For each remaining column, update it:
    # t_c_new = t_c - (coeffs[c] / coeffs[eliminate]) * t_eliminate
    # Wait, the dependence is sum_c coeffs[c] * t_c = 0
    # So coeffs[eliminate] * t_eliminate = -sum_{c != eliminate} coeffs[c] * t_c
    # t_eliminate = -sum_{c != eliminate} (coeffs[c] / coeffs[eliminate]) * t_c
    # 
    # To eliminate t_eliminate, we just drop it. The sum of the remaining terms
    # with adjusted coefficients will still sum to the same tensor.
    # 
    # Actually, the simplest way: the original decomposition is
    # T = sum_c t_c
    # The dependence is sum_c coeffs[c] * t_c = 0
    # So T = sum_c t_c = sum_c t_c + sum_c coeffs[c] * t_c = sum_c (1 + coeffs[c]) * t_c
    # This doesn't help directly.
    #
    # Better: T = sum_{c != eliminate} t_c + t_eliminate
    # And t_eliminate = -sum_{c != eliminate} (coeffs[c] / coeffs[eliminate]) * t_c
    # So T = sum_{c != eliminate} t_c - sum_{c != eliminate} (coeffs[c] / coeffs[eliminate]) * t_c
    #     = sum_{c != eliminate} (1 - coeffs[c] / coeffs[eliminate]) * t_c
    #
    # So the new coefficients for column c (c != eliminate) are:
    # new_coeff[c] = 1 - coeffs[c] / coeffs[eliminate]
    
    for c in range(rank):
        if c == eliminate_idx:
            continue
        new_c = old_to_new[c]
        scale = 1.0 - coeffs[c] / coeffs[eliminate_idx]
        U_new[:, new_c] = U[:, c] * scale
        V_new[:, new_c] = V[:, c] * scale
        W_new[:, new_c] = W[:, c] * scale
    
    return U_new, V_new, W_new

def _try_reduce_rank(U, V, W, T, n, m, p, target_rank, deadline):
    """
    Try to reduce the rank from current to target_rank by finding linear dependencies.
    """
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    # We need to eliminate (current_rank - target_rank) terms
    # For now, try to eliminate one term at a time
    
    U_curr, V_curr, W_curr = U, V, W
    
    for _ in range(current_rank - target_rank):
        if time.time() > deadline - 10:
            break
        
        coeffs, eliminate_idx = _find_linear_dependence(U_curr, V_curr, W_curr, T, n, m, p)
        
        if coeffs is None:
            # No linear dependence found, can't reduce further
            break
        
        U_curr, V_curr, W_curr = _reduce_rank(U_curr, V_curr, W_curr, coeffs, eliminate_idx, n, m, p)
        
        # Validate
        if not _validate(U_curr, V_curr, W_curr, n, m, p):
            # Validation failed, revert
            break
    
    return U_curr, V_curr, W_curr

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank = n*m*p."""
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
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    T = _multiplication_tensor(n, m, p)
    
    # Start from the rank-44 Strassen-based decomposition
    U44, V44, W44 = _combine_3x4x4()
    
    # Verify the rank-44 decomposition is correct
    if not _validate(U44, V44, W44, n, m, p):
        # Fallback
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    best_result = (U44.tolist(), V44.tolist(), W44.tolist())
    best_rank = 44
    
    # Try to reduce rank: try 43, 42, 41, ...
    for target_rank in range(43, 35, -1):
        if time.time() > deadline - 20:
            break
        
        U_t, V_t, W_t = _try_reduce_rank(U44, V44, W44, T, n, m, p, target_rank, deadline)
        
        if _validate(U_t, V_t, W_t, n, m, p):
            best_result = (U_t.tolist(), V_t.tolist(), W_t.tolist())
            best_rank = U_t.shape[1]
            # Update starting point
            U44, V44, W44 = U_t, V_t, W_t
        else:
            # If we couldn't reach target_rank, stop
            break
    
    # Final validation
    U_f, V_f, W_f = best_result
    if not _validate(U_f, V_f, W_f, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
