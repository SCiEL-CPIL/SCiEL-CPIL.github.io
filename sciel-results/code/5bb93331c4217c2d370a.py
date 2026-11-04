import numpy as np
import time

def construct():
    """
    Rank-reduced Strassen-based decomposition for 4x4 * 4x8 matrix multiplication.
    
    Builds a rank-98 decomposition via Kronecker product of Strassen 2x2x2,
    then searches for linearly dependent columns to reduce the rank.
    """
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    
    n, m, p = 4, 4, 8
    
    # Build the base rank-98 Strassen decomposition
    result = construct_rank98(n, m, p)
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to naive if verification fails
        return construct_naive(n, m, p)
    
    elapsed = time.time() - start_time
    if elapsed > 80:
        # Not enough time for reduction, return base
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linearly dependent columns
    rank_reduced = _reduce_rank(U, V, W, n, m, p, deadline - start_time)
    
    if rank_reduced is not None:
        U_r, V_r, W_r = rank_reduced
        if _verify_decomposition(U_r, V_r, W_r, n, m, p):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Return the base decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_rank98(n, m, p):
    """Rank-98 decomposition using Kronecker product of Strassen levels."""
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    # Build 4x4x8 rank-98 decomposition
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First block (output cols 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second block (output cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Test with random matrices
    rng = np.random.default_rng(42)
    for trial in range(5):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # Compute using decomposition
        # M_t = sum_{i,j} U[i,j,t] * A[i,j] * (sum_{k} V[j,k,t] * B[j,k])
        # Actually: M_t = (U[:,t]^T * A_flat) * (V[:,t]^T * B_flat) where A_flat is row-major
        # C[i,k] = sum_t W[k,i,t] * M_t
        
        A_flat = A.flatten()  # n*m
        B_flat = B.flatten()  # m*p
        
        M = np.zeros(rank)
        for t in range(rank):
            u_val = np.dot(U[:, t], A_flat)
            v_val = np.dot(V[:, t], B_flat)
            M[t] = u_val * v_val
        
        C_computed = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                C_computed[i, k] = np.dot(W[k*n+i, :], M)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _reduce_rank(U, V, W, n, m, p, time_budget):
    """
    Try to reduce the rank by finding linearly dependent columns.
    
    A column t is redundant if there exist columns t1, t2, ..., tk and scalars
    such that the tensor contribution of column t equals the sum of contributions
    of the other columns.
    
    We search for pairs of columns that are linearly dependent in a specific sense.
    """
    rank = U.shape[1]
    elapsed_start = time.time()
    
    # Strategy: check if any column can be expressed as a linear combination
    # of others. For a bilinear form, column t is redundant if:
    # U[:,t] = sum_s alpha_s * U[:,s] AND V[:,t] = sum_s beta_s * V[:,s] 
    # AND W[:,t] = sum_s (alpha_s * beta_s) * W[:,s]
    # This is a special case. More generally, we need:
    # For all i,j,k: sum_t W[k,i,t]*U[i,j,t]*V[j,k,t] is invariant under removing t
    # if the contribution of t is exactly cancelled by adjusting others.
    
    # Simpler approach: find columns where U[:,t], V[:,t], W[:,t] are all
    # linearly dependent on subsets of other columns.
    
    # First, let's try to find duplicate or negated column triples
    # Two columns t1, t2 are "equivalent" if U[:,t1]=c*U[:,t2], V[:,t1]=c*V[:,t2], W[:,t1]=c*W[:,t2]
    # In that case, we can combine them (but this doesn't reduce rank unless we merge).
    
    # Better: find if column t is a linear combination of a small subset of other columns
    # such that the product structure is preserved.
    
    # Let's try: for each column t, check if it can be removed by adjusting
    # a few other columns. This is expensive, so we'll do a limited search.
    
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    kept = list(range(rank))
    
    # Try removing columns one at a time, checking if the decomposition still works
    # after compensating with nearby columns.
    
    # Actually, a more systematic approach: 
    # The tensor T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # If we remove column t, we need T to be unchanged.
    # This means U[:,t]*V[:,t]*W[:,t] (as a tensor) must be in the span of
    # {U[:,s]*V[:,s]*W[:,s] for s != t}.
    
    # Let's build the tensor for each column and check linear dependence.
    
    def col_tensor(U_c, V_c, W_c, n, m, p):
        """Compute the rank-1 tensor for one column."""
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
        return T.flatten()
    
    # Build all column tensors
    col_tensors = []
    for t in range(rank):
        col_tensors.append(col_tensor(U_work[:, t], V_work[:, t], W_work[:, t], n, m, p))
    
    col_tensors = np.array(col_tensors)  # shape (rank, n*m*p)
    
    # Find linearly dependent columns using QR with column pivoting
    # or by checking rank of subsets
    
    # Use a greedy approach: iterate through columns, keep a column only if
    # it adds rank to the set of kept column tensors.
    
    kept_indices = []
    kept_tensors = []
    
    for t in range(rank):
        if time.time() - elapsed_start > time_budget * 0.8:
            break
        
        if not kept_tensors:
            kept_indices.append(t)
            kept_tensors.append(col_tensors[t])
        else:
            # Check if col_tensors[t] is in the span of kept_tensors
            mat = np.array(kept_tensors)  # (k, n*m*p)
            # We want to solve mat.T @ coeffs = col_tensors[t]
            # So A = mat.T (n*m*p, k), b = col_tensors[t] (n*m*p,)
            A = mat.T
            b = col_tensors[t]
            try:
                x, residuals, rank_mat, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                if residuals.size > 0 and residuals[0] < 1e-10:
                    # Linearly dependent, skip
                    pass
                else:
                    kept_indices.append(t)
                    kept_tensors.append(col_tensors[t])
            except np.linalg.LinAlgError:
                # If lstsq fails, keep the column to be safe
                kept_indices.append(t)
                kept_tensors.append(col_tensors[t])
    
    if len(kept_indices) < rank:
        # We found some redundant columns
        U_reduced = U_work[:, kept_indices]
        V_reduced = V_work[:, kept_indices]
        W_reduced = W_work[:, kept_indices]
        
        # Verify
        if _verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p):
            return U_reduced, V_reduced, W_reduced
    
    return None
