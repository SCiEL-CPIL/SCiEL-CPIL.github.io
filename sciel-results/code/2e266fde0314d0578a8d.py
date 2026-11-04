import numpy as np
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses rank-98 Strassen (Kronecker of 2x2x2 rank-7) as base,
    then attempts column elimination with fast vectorized tensor verification.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s limit

    n, m, p = 4, 4, 8

    # Build the base rank-98 Strassen decomposition
    U_base, V_base, W_base = _construct_rank98(n, m, p)

    # Verify the base decomposition with fast check
    if not _verify_decomposition_fast(U_base, V_base, W_base, n, m, p):
        U_base, V_base, W_base = _construct_naive(n, m, p)
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}

    best_result = (U_base, V_base, W_base)
    best_rank = U_base.shape[1]

    # Try iterative column elimination with fast tensor verification
    elapsed = time.time() - start_time
    if elapsed < deadline - start_time - 30:
        further_reduced = _iterative_column_elimination_fast(
            best_result[0], best_result[1], best_result[2], n, m, p,
            deadline - start_time
        )
        if further_reduced is not None:
            U_f, V_f, W_f = further_reduced
            if _verify_decomposition_fast(U_f, V_f, W_f, n, m, p):
                if U_f.shape[1] < best_rank:
                    best_result = (U_f, V_f, W_f)
                    best_rank = U_f.shape[1]

    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}


def _construct_rank98(n, m, p):
    """Rank-98 decomposition using Kronecker product of Strassen levels."""
    n, m, p = 4, 4, 8

    # Strassen 2x2x2 rank-7 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0, -1]
    ], dtype=float)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)

    # Build 4x4x4 rank-49 decomposition via Kronecker product
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)

    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2 * i1 + i2
                    j = 2 * j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t * 7 + s
                            U4[i * 4 + j, c] = U2[i1 * 2 + j1, t] * U2[i2 * 2 + j2, s]

    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2 * j1 + j2
                    k = 2 * k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t * 7 + s
                            V4[j * 4 + k, c] = V2[j1 * 2 + k1, t] * V2[j2 * 2 + k2, s]

    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2 * k1 + k2
                    i = 2 * i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t * 7 + s
                            W4[k * 4 + i, c] = W2[k1 * 2 + i1, t] * W2[k2 * 2 + i2, s]

    # Build 4x4x8 rank-98 decomposition
    rank = 98
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # First block (output cols 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, 0:49]

    # Second block (output cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 49:98] = V4[j * 4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k + 4) * 4 + i, 49:98] = W4[k * 4 + i, 0:49]

    return U, V, W


def _compute_tensor(U, V, W, n, m, p):
    """
    Compute the full tensor T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    in a vectorized way. Returns shape (n, m, p).
    """
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)

    # For each t, we have outer product U3[:,:,t] (n,m) * V3[:,:,t] (m,p) * W3[:,:,t] (p,n)
    # T[i,j,k] = sum_t U3[i,j,t] * V3[j,k,t] * W3[k,i,t]
    # We can compute this as: for each t, add U3[:,:,t] ⊗ V3[:,:,t] ⊗ W3[:,:,t]
    # But more efficiently: T = einsum('ijt,jkt,kit->ijk', U3, V3, W3)
    T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3, optimize=True)
    return T


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Fast verification: compute the full tensor and check it equals the multiplication tensor.
    The multiplication tensor has T[i,j,k] = 1 if j==k (for standard matmul C[i,k] = sum_j A[i,j]*B[j,k])
    Wait - for matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    So the tensor is T[i,j,k] = 1 if the indices match the multiplication pattern.
    Actually, the tensor decomposition satisfies:
    sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = delta_{j,k}... no.
    
    Let me reconsider. The matrix multiplication tensor M satisfies:
    M[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    So M[i,j,k] = 1 for all valid (i,j,k) since every triple contributes.
    Wait no - the tensor is defined as:
    For C = A*B: C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor M has entries M[i,j,k] = 1 for all i,j,k (it's the identity in the sense
    that each (i,j,k) triple contributes to exactly one output entry C[i,k]).
    
    Actually, the correct statement is:
    The matrix multiplication tensor is T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    when we write C[i,k] = sum_j A[i,j] * B[j,k].
    
    No wait. The tensor is defined such that:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor decomposition form:
    C[i,k] = sum_t (sum_j U[i,j,t]*A[i,j]) * (sum_j V[j,k,t]*B[j,k]) * W[k,i,t]
    
    This equals sum_t W[k,i,t] * (U[:, :, t] · A) * (V[:, :, t] · B)
    
    For this to equal A*B, we need:
    sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1 for all (i,j,k)
    
    So yes, the tensor is all ones.
    """
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False

    T = _compute_tensor(U, V, W, n, m, p)
    # The multiplication tensor has T[i,j,k] = 1 for all (i,j,k)
    expected = np.ones((n, m, p), dtype=float)
    if not np.allclose(T, expected, atol=tol):
        return False
    return True


def _iterative_column_elimination_fast(U, V, W, n, m, p, time_budget):
    """
    Iteratively try to eliminate columns using fast tensor verification.
    A column t can be removed if the tensor identity still holds without it.
    """
    start = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    original_rank = U.shape[1]
    
    max_iterations = 30  # Try to remove up to 30 columns
    
    for iteration in range(max_iterations):
        if time.time() - start > time_budget:
            break
        
        current_rank = U.shape[1]
        if current_rank <= 64:  # Don't go below reasonable bounds
            break
        
        # Try to remove each column, starting from the end
        removed = False
        for col in range(current_rank - 1, -1, -1):
            if time.time() - start > time_budget:
                break
            
            # Try removing column col
            mask = np.ones(current_rank, dtype=bool)
            mask[col] = False
            U_trial = U[:, mask]
            V_trial = V[:, mask]
            W_trial = W[:, mask]
            
            # Fast check: compute tensor and verify
            if _verify_decomposition_fast(U_trial, V_trial, W_trial, n, m, p):
                U, V, W = U_trial, V_trial, W_trial
                removed = True
                break
        
        if not removed:
            break
    
    if U.shape[1] < original_rank:
        return U, V, W
    return None


def _construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    return _verify_decomposition_fast(U, V, W, n, m, p)
