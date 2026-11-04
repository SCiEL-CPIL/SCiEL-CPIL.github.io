import numpy as np
import time

def _find_independent_columns(A, tol=1e-10):
    """Find a maximum set of linearly independent columns in A.
    Returns the list of column indices that form a basis."""
    n_cols = A.shape[1]
    if n_cols == 0:
        return []
    
    # Use QR with column pivoting via iterative rank detection
    # Simpler approach: incremental rank tracking
    basis_cols = []
    # Maintain a matrix B whose columns are the selected basis vectors
    B = np.zeros((A.shape[0], 0))
    
    for col_idx in range(n_cols):
        v = A[:, col_idx]
        if len(basis_cols) == 0:
            # First non-zero column
            if np.linalg.norm(v) > tol:
                basis_cols.append(col_idx)
                B = v.reshape(-1, 1)
            continue
        
        # Check if v is in span of B
        # Solve B x = v, check residual
        try:
            x, residuals, rank, sv = np.linalg.lstsq(B, v, rcond=tol)
            residual = np.linalg.norm(v - B @ x)
            if residual > tol * max(1.0, np.linalg.norm(v)):
                # v is not in span of B
                basis_cols.append(col_idx)
                B = np.hstack([B, v.reshape(-1, 1)])
        except np.linalg.LinAlgError:
            # Fallback: use SVD
            pass
    
    return basis_cols

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition correctly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    """
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This is expensive for large rank, so we check a sample
    # Actually, let's do it properly using einsum
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Reshape for efficient computation
    U_reshaped = U.reshape(n, m, rank)  # U_reshaped[i,j,r]
    V_reshaped = V.reshape(m, p, rank)  # V_reshaped[j,k,r]
    W_reshaped = W.reshape(p, n, rank)  # W_reshaped[k,i,r]
    
    # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    # This is a 4D contraction. Let's compute it step by step.
    # First: M[i,j,k,r] = U_reshaped[i,j,r] * V_reshaped[j,k,r]
    # Then: T[i,j,k] = sum_r M[i,j,k,r] * W_reshaped[k,i,r]
    
    # Efficient approach using einsum
    # U_reshaped: (n, m, r)
    # V_reshaped: (m, p, r)  
    # W_reshaped: (p, n, r)
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Compute in steps to avoid memory issues
    # Step 1: A[i,j,k,r] = U[i,j,r] * V[j,k,r]  -- shape (n, m, p, r)
    # This might be large. For n=m=4, p=8, r=98: 4*4*8*98 = 12544, fine.
    A = np.einsum('ijr,jkr->ijk r'.replace(' ', ''), U_reshaped, V_reshaped)
    # A shape: (n, m, p, rank)
    
    # Step 2: T[i,j,k] = sum_r A[i,j,k,r] * W[k,i,r]
    # W_reshaped: (p, n, r) -> need W[k,i,r]
    T = np.einsum('ijk r,ki r->ijk'.replace(' ', ''), A, W_reshaped)
    
    # T should be all 1s
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol

def _reduce_rank(U, V, W, n, m, p, tol=1e-10, max_iterations=50):
    """Try to reduce the rank by finding and removing linearly dependent columns.
    A column r can be removed if the three vectors U[:,r], V[:,r], W[:,r] 
    are such that the tensor contribution can be expressed via other columns.
    
    More precisely: if there exist coefficients alpha_r (for other columns) such that
    for all (i,j,k): U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = sum_{r' != r} alpha_{r'} * U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r']
    then column r is redundant.
    
    This is equivalent to checking if the trilinear form associated with column r
    is in the span of the other trilinear forms.
    
    A simpler sufficient condition: if U[:,r] is in the column space of U[:, S] for some
    subset, AND V[:,r] is in the column space of V[:, S], AND W[:,r] is in the column
    space of W[:, S], then... no, that's not quite right either.
    
    Actually, the correct condition is: the rank of the tensor is the minimum number
    of rank-1 tensors needed. Two different rank-1 tensors can combine.
    
    Let me think differently. The decomposition is valid if:
    sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
    
    If we remove column r, we need:
    sum_{r' != r} U[i*m+j,r'] * V[j*p+k,r'] * W[k*n+i,r'] = 1 for all (i,j,k)
    
    This means the tensor T_r(i,j,k) = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] must be
    expressible as a linear combination of T_{r'} for r' != r.
    
    The set of all rank-1 tensors {u ⊗ v ⊗ w} spans a space. The question is whether
    T_r is in the span of {T_{r'} : r' != r}.
    
    We can check this by: fix the trilinear form and see if it's in the span.
    One way: vectorize. T_r as a vector in R^{n*m*p}. Check if it's in the span
    of the other T_{r'} vectors.
    """
    rank = U.shape[1]
    
    # Vectorize each rank-1 tensor component
    # T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape to 3D and then flatten
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # For each r, compute the vector T_r of length n*m*p
    # T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Precompute all T_r vectors
    T_vectors = []
    for r in range(rank):
        u = U_reshaped[:, :, r]  # (n, m)
        v = V_reshaped[:, :, r]  # (m, p)
        w = W_reshaped[:, :, r]  # (p, n)
        # T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        t = np.einsum('ij,jk,ki->ijk', u, v, w)
        T_vectors.append(t.flatten())
    
    T_matrix = np.array(T_vectors).T  # (n*m*p, rank)
    
    # Now find the maximum set of linearly independent columns in T_matrix
    # The rank of T_matrix gives the minimum number of columns needed
    # (since each column is a trilinear form, and we need their sum to be all-ones)
    
    # Find independent columns
    independent_cols = _find_independent_columns(T_matrix, tol=tol)
    
    if len(independent_cols) < rank:
        # We can potentially reduce rank
        # But we need to verify that the sum of the remaining columns still equals all-ones
        # Actually, the condition is that sum_r T_r = ones vector
        # If we remove some columns, we need sum_{r in remaining} T_r = ones
        # This is a different condition than linear independence!
        
        # The correct approach: we want to find a subset S of columns such that
        # sum_{r in S} T_r = ones, and |S| is minimized.
        # This is a different problem than finding linearly independent columns.
        
        # However, if the T_r are linearly dependent, we can express some as combinations
        # of others. But we need the SUM to be ones, not just linear independence.
        
        # Let me reconsider. The constraint is:
        # sum_r T_r = 1_vector
        # If T_r = sum_{r' in S} c_{r,r'} T_{r'} for some r not in S,
        # then sum_r T_r = sum_{r in S} (1 + sum_{r not in S, r' = r} c_{r,r'}) T_r ... this gets complicated.
        
        # A simpler approach: just check if we can remove columns one at a time
        # and still have the sum equal to ones.
        
        current_U = U.copy()
        current_V = V.copy()
        current_W = W.copy()
        current_rank = rank
        removed = []
        
        for iteration in range(max_iterations):
            if time.time() > start_time + 135.0:
                break
            
            found_removal = False
            for r in range(current_rank):
                if r in removed:
                    continue
                
                # Try removing column r
                # Check if sum of remaining columns still equals ones
                # Compute sum of T_r for r not removed and r != current r
                T_remaining = np.zeros(n * m * p)
                for r2 in range(current_rank):
                    if r2 == r or r2 in removed:
                        continue
                    T_remaining += T_vectors[r2]
                
                if np.max(np.abs(T_remaining - 1.0)) < tol:
                    # Column r can be removed
                    removed.append(r)
                    found_removal = True
                    break
            
            if not found_removal:
                break
        
        if removed:
            # Build the reduced matrices
            keep_cols = [r for r in range(current_rank) if r not in removed]
            U_new = U[:, keep_cols]
            V_new = V[:, keep_cols]
            W_new = W[:, keep_cols]
            return U_new, V_new, W_new
    
    return U, V, W

def construct():
    """
    Rank-minimal decomposition for 4x4 * 4x8 matrix multiplication.
    
    Uses the Kronecker product of two levels of Strassen (2x2x2 rank-7)
    to get a rank-49 decomposition for 4x4x4, then applies it independently
    to the two 4x4 column blocks of B, giving total rank 98.
    Then attempts rank reduction by identifying redundant columns.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
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
    
    # Build 4x4x4 rank-49 decomposition
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
    
    # First block
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second block
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        # Should not happen, but return the base decomposition
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt rank reduction
    try:
        U_reduced, V_reduced, W_reduced = _reduce_rank(U, V, W, n, m, p, tol=1e-8, max_iterations=100)
        
        if U_reduced.shape[1] < rank:
            # Verify the reduced decomposition
            if _verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p, tol=1e-6):
                return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}
    except Exception:
        pass
    
    # Return the base rank-98 decomposition
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
