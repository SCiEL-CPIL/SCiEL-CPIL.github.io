import numpy as np
import time

def construct():
    """
    Rank-reduced decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Strassen-based construction then SVD rank reduction.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    # Build the base rank-98 Strassen decomposition
    U, V, W = _construct_rank98(n, m, p)
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _construct_naive(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    elapsed = time.time() - start_time
    if elapsed > 120:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank via SVD-based dependency resolution
    rank_reduced = _reduce_rank_svd(U, V, W, n, m, p, deadline - start_time)
    
    if rank_reduced is not None:
        U_r, V_r, W_r = rank_reduced
        if _verify_decomposition(U_r, V_r, W_r, n, m, p):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Return the base decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _construct_rank98(n, m, p):
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
    
    return U, V, W


def _construct_naive(n, m, p):
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
                
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Test with random matrices
    rng = np.random.default_rng(42)
    for trial in range(3):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
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


def _reduce_rank_svd(U, V, W, n, m, p, time_budget):
    """
    SVD-based rank reduction for matrix multiplication tensor.
    
    Key insight: For matmul tensor T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t],
    if the column tensors (flattened over (i,j,k)) are linearly dependent, we can find
    a smaller generating set. However, simple scaling only works when the dependencies
    have a special structure. We use a more robust approach:
    
    1. Compute the true rank of the tensor via SVD of the flattened tensor
    2. Find a basis of independent column tensors
    3. For each non-basis column, express it as a linear combination of basis columns
    4. The reduction is valid only if we can absorb the coefficients into the factor matrices
       in a way that preserves the matmul structure.
    
    A simpler but effective approach: try removing columns one at a time and checking
    if the remaining columns still span the same tensor space (with appropriate scaling).
    """
    rank = U.shape[1]
    dim = n * m * p
    elapsed_start = time.time()
    
    # Build column tensors
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_flat_idx = i_idx * m + j_idx
    v_flat_idx = j_idx * p + k_idx
    w_flat_idx = k_idx * n + i_idx
    
    U_vals = U[u_flat_idx, :]  # (dim, rank)
    V_vals = V[v_flat_idx, :]  # (dim, rank)
    W_vals = W[w_flat_idx, :]  # (dim, rank)
    
    # col_tensors[t, idx] = U_vals[idx, t] * V_vals[idx, t] * W_vals[idx, t]
    col_tensors = (U_vals * V_vals * W_vals).T  # (rank, dim)
    
    if time.time() - elapsed_start > time_budget * 0.2:
        return None
    
    # SVD to find the rank of the space spanned by column tensors
    try:
        U_svd, s, Vt = np.linalg.svd(col_tensors, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # Determine numerical rank
    tol = 1e-10
    num_rank = int(np.sum(s > tol))
    
    if num_rank >= rank:
        return None
    
    # Try to find a valid reduction by iteratively removing columns
    # and checking if we can rescale to maintain the decomposition
    
    # First, try the simple approach: find basis columns and scale
    coords = U_svd[:, :num_rank] * s[:num_rank]  # (rank, num_rank)
    
    if time.time() - elapsed_start > time_budget * 0.3:
        return None
    
    # Find basis using QR with pivoting
    coords_T = coords.T  # (num_rank, rank)
    Q, R, piv = np.linalg.qr(coords_T, mode='complete', pivoting=True)
    basis_indices = sorted(piv[:num_rank])
    
    if len(basis_indices) != num_rank:
        return None
    
    # For the reduction to work with simple scaling, we need:
    # sum_t alpha_{t,b} * (u_t, v_t, w_t) = (scaled_u_b, v_b, w_b) for each basis b
    # This requires that for each non-basis t and basis b:
    #   alpha_{t,b} * u_t = scaled factor * u_b  (element-wise proportional)
    #   alpha_{t,b} * v_t = v_b  (element-wise)
    #   alpha_{t,b} * w_t = w_b  (element-wise)
    # which is very restrictive.
    
    # Alternative: try a greedy removal approach
    # Start with all columns, try removing each one and see if the tensor can be
    # reconstructed with the remaining columns by adjusting scalings.
    
    best_result = None
    current_rank = rank
    active = list(range(rank))
    
    max_attempts = min(rank, 50)
    attempts = 0
    
    for attempt in range(max_attempts):
        if time.time() - elapsed_start > time_budget * 0.9:
            break
        
        # Try removing each column in active
        removed = False
        for idx in range(len(active)):
            if time.time() - elapsed_start > time_budget * 0.9:
                break
            
            test_active = active[:idx] + active[idx+1:]
            if len(test_active) < num_rank:
                continue
            
            # Try to find scalings for test_active that reproduce the tensor
            U_test = U[:, test_active]
            V_test = V[:, test_active]
            W_test = W[:, test_active]
            
            # The tensor from test_active: sum_{t in test_active} U[:,t] * V[:,t] * W[:,t]
            # We want this to equal the original tensor
            # Build the linear system: find scalings s_t such that
            # sum_{t in test_active} s_t * (u_t * v_t * w_t) = original tensor
            
            # Compute tensor for test_active columns
            col_tensors_test = np.array([
                U_vals[:, t] * V_vals[:, t] * W_vals[:, t] 
                for t in test_active
            ]).T  # (len(test_active), dim)
            
            # Original tensor
            col_tensors_orig = col_tensors[active, :]  # (current_rank, dim)
            # Actually we need the full original tensor, not just active columns
            # The original tensor is the sum over ALL original columns
            
            # Let me reconsider: the original tensor T = sum_{t=0}^{rank-1} u_t ⊗ v_t ⊗ w_t
            # If we remove column r, we need sum_{t != r} s_t * u_t ⊗ v_t ⊗ w_t = T
            # This is a linear system in s_t
            
            # Build the system: for each (i,j,k), 
            # sum_{t in test_active} s_t * U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = T[i,j,k]
            
            # T[i,j,k] = sum_{t=0}^{rank-1} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
            
            # This is a large linear system. Let's use least squares.
            
            # Flatten the tensor space
            # For each (i,j,k), we have a linear equation in s_t
            # A[s_idx, t_idx] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
            # b[s_idx] = sum_{all_t} U[i*m+j, all_t] * V[j*p+k, all_t] * W[k*n+i, all_t]
            
            # Build A matrix (dim x len(test_active))
            A_mat = np.array([
                U_vals[:, t] * V_vals[:, t] * W_vals[:, t]
                for t in test_active
            ]).T  # (dim, len(test_active))
            
            # Build b vector (dim,)
            b_vec = np.array([
                np.sum(U_vals[:, t] * V_vals[:, t] * W_vals[:, t])
                for t in range(rank)
            ])
            # Wait, that's not right. b should be the original tensor value at each (i,j,k)
            # T[i,j,k] = sum_{t=0}^{rank-1} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
            
            b_vec = np.zeros(dim)
            for t in range(rank):
                b_vec += U_vals[:, t] * V_vals[:, t] * W_vals[:, t]
            
            # Solve A_mat @ s = b_vec
            try:
                s_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
            except np.linalg.LinAlgError:
                continue
            
            # Check if the solution is valid (small residual)
            residual = np.linalg.norm(A_mat @ s_sol - b_vec)
            if residual > 1e-8 * max(1, np.linalg.norm(b_vec)):
                continue
            
            # Check that scalings are reasonable (not too large)
            if np.any(np.abs(s_sol) > 1e6):
                continue
            
            # Build the reduced decomposition
            U_red = U[:, test_active].copy() * s_sol[np.newaxis, :]
            V_red = V[:, test_active].copy()
            W_red = W[:, test_active].copy()
            
            if _verify_decomposition(U_red, V_red, W_red, n, m, p):
                if best_result is None or len(test_active) < best_result[0]:
                    best_result = (len(test_active), U_red, V_red, W_red)
                    active = test_active
                    removed = True
                    break
        
        if not removed:
            break
    
    if best_result is not None:
        _, U_r, V_r, W_r = best_result
        return U_r, V_r, W_r
    
    return None


def _greedy_basis_selection(coords, num_rank):
    """Greedy selection of linearly independent rows from coords."""
    rank = coords.shape[0]
    selected = []
    
    for t in range(rank):
        if len(selected) >= num_rank:
            break
        if not selected:
            selected.append(t)
        else:
            A = coords[selected, :]
            b = coords[t, :]
            try:
                x, residuals, _, _ = np.linalg.lstsq(A.T, b, rcond=1e-12)
                if residuals.size > 0 and residuals[0] < 1e-10:
                    continue
            except np.linalg.LinAlgError:
                continue
            selected.append(t)
    
    if len(selected) < num_rank:
        return None
    return selected[:num_rank]
