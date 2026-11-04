import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook rank-100 decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is exact for matrix multiplication."""
    rank = U.shape[1]
    # Build the full tensor T[i, j, j2, k] = sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k*n+i, r]
    # This should equal delta[j, j2] for all i, j, j2, k
    T = np.zeros((n, m, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        # Reshape u_col to (n, m), v_col to (m, p), w_col to (p, n)
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        # Outer product: T[i, j, j2, k] += u_mat[i, j] * v_mat[j2, k] * w_mat[k, i]
        # Note: w_mat[k, i] corresponds to W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        T[i, j, j2, k] += u_mat[i, j] * v_mat[j2, k] * w_mat[k, i]
    # Check T[i, j, j2, k] == delta[j, j2]
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    expected = 1.0 if j == j2 else 0.0
                    if abs(T[i, j, j2, k] - expected) > tol:
                        return False
    return True

def _round_to_half_integer(x, tol=1e-8):
    """Round x to the nearest half-integer if within tol, otherwise return None."""
    # Half-integers are k/2 for integer k
    k = round(2 * x)
    candidate = k / 2.0
    if abs(x - candidate) < tol:
        return candidate
    return None

def _is_half_integer_matrix(M, tol=1e-8):
    """Check if all entries in M are half-integers within tol."""
    for val in M.flatten():
        if _round_to_half_integer(val, tol) is None:
            return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_iter=1000, time_limit=120):
    """
    Try to reduce the rank of the decomposition by finding linear dependencies.
    Uses a greedy approach: try to remove columns that are linear combinations of others.
    """
    start_time = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to reduce rank by checking if any column can be expressed as a combination of others
    # This is a simplified approach; a full SVD-based approach would be better but more complex
    
    rank = U_np.shape[1]
    for iter_num in range(max_iter):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Try to remove one column at a time
        for r in range(rank):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Try removing column r
            U_test = np.delete(U_np, r, axis=1)
            V_test = np.delete(V_np, r, axis=1)
            W_test = np.delete(W_np, r, axis=1)
            
            new_rank = U_test.shape[1]
            if new_rank >= best_rank:
                continue
            
            # Verify the decomposition
            if _verify_decomposition(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test.copy(), V_test.copy(), W_test.copy()
                best_rank = new_rank
                U_np, V_np, W_np = U_test.copy(), V_test.copy(), W_test.copy()
                rank = new_rank
                break
        
        if rank == best_rank:
            # No more reductions possible in this pass
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Constructs the rank-minimal tensor decomposition for matrix multiplication 
    of dimensions 4x5 * 5x5.
    
    Starts with the schoolbook algorithm (rank 100) and attempts to reduce the rank
    by finding and removing linearly dependent terms.
    """
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = U.shape[1]
    
    # Verify schoolbook is correct
    assert _verify_decomposition(U, V, W, n, m, p), "Schoolbook decomposition failed verification"
    
    # Try to reduce rank
    U_red, V_red, W_red, reduced_rank = _try_reduce_rank(U, V, W, n, m, p, max_iter=500, time_limit=140)
    
    if reduced_rank < best_rank:
        U, V, W = U_red, V_red, W_red
        best_rank = reduced_rank
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
