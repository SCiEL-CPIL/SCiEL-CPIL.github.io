import numpy as np
import time

def construct():
    """
    Rank-reduced Strassen-based decomposition for 4x4 * 4x8 matrix multiplication.
    
    Builds a rank-98 decomposition via Kronecker product of Strassen 2x2x2,
    then uses SVD-based rank detection to find the minimal generating set
    and resolve all linear dependencies in one batched solve.
    """
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    # Build the base rank-98 Strassen decomposition
    result = construct_rank98(n, m, p)
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        return construct_naive(n, m, p)
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Not enough time for reduction, return base
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank via SVD-based dependency resolution
    rank_reduced = _reduce_rank_svd(U, V, W, n, m, p, deadline - start_time)
    
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
    SVD-based rank reduction: build all column tensors, find the rank of their span,
    identify a minimal generating set, express all others as linear combinations,
    and apply scaling corrections in one batched operation.
    """
    rank = U.shape[1]
    dim = n * m * p
    elapsed_start = time.time()
    
    # Build column tensors: col_tensors[t, idx] where idx = i*m*p + j*p + k
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
    
    if time.time() - elapsed_start > time_budget * 0.5:
        return None
    
    # SVD to find the rank of the space spanned by column tensors
    # col_tensors is (rank, dim). We want the row space rank.
    # SVD: col_tensors = U_svd @ diag(s) @ Vt
    try:
        U_svd, s, Vt = np.linalg.svd(col_tensors, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # Determine numerical rank
    tol = 1e-10
    num_rank = np.sum(s > tol)
    
    if num_rank >= rank:
        # No reduction possible
        return None
    
    # The row space is spanned by the first num_rank right singular vectors (rows of Vt)
    # We need to find which num_rank columns of col_tensors form a basis.
    # Strategy: use the SVD to express each column tensor in the basis of left singular vectors.
    # col_tensors @ Vt[:num_rank].T gives coordinates in the row space basis.
    
    # Actually, let's think differently. The columns of col_tensors.T (i.e., rows of col_tensors)
    # are the tensors. We want to find a subset of rows that spans all others.
    
    # Use the SVD: col_tensors = U_svd @ diag(s) @ Vt
    # The row space is spanned by the first num_rank rows of Vt (transposed).
    # Each row t of col_tensors can be written as sum_{a=1}^{num_rank} col_tensors[t,a'] * Vt[a']
    # where a' indexes into the left singular vectors.
    
    # Better approach: project each row onto the row space basis.
    # The row space basis vectors are Vt[:num_rank, :].T (shape dim x num_rank)
    # Coordinates of row t in this basis: col_tensors[t] @ Vt[:num_rank].T
    # But actually, since col_tensors = U_svd @ diag(s) @ Vt,
    # row t of col_tensors = U_svd[t, :num_rank] @ diag(s[:num_rank]) @ Vt[:num_rank, :]
    # So the coordinates of row t in the basis {Vt[0,:], ..., Vt[num_rank-1,:]} are:
    # coords[t, a] = U_svd[t, a] * s[a] for a in range(num_rank)
    
    coords = U_svd[:, :num_rank] * s[:num_rank]  # (rank, num_rank)
    
    if time.time() - elapsed_start > time_budget * 0.6:
        return None
    
    # Now find a subset of rows that forms a basis for the row space.
    # We need num_rank rows whose coordinate vectors are linearly independent.
    # Use QR with column pivoting on coords.T (shape num_rank x rank)
    # to find which rank columns are pivot columns.
    
    # Actually, we want to find num_rank rows of coords that are linearly independent.
    # This is equivalent to finding num_rank columns of coords.T that are linearly independent.
    
    coords_T = coords.T  # (num_rank, rank)
    
    # Use QR with column pivoting
    Q, R, piv = np.linalg.qr(coords_T, mode='complete', pivoting=True)
    # piv gives the column permutation. The first num_rank columns of R (in pivoted order)
    # should have non-zero diagonal entries if the rank is exactly num_rank.
    
    # The pivot columns in the original ordering are piv[:num_rank]
    basis_indices = sorted(piv[:num_rank])
    
    if len(basis_indices) != num_rank:
        return None
    
    # Verify that these basis rows are indeed linearly independent
    basis_coords = coords[basis_indices, :]  # (num_rank, num_rank)
    det_val = np.linalg.det(basis_coords)
    if abs(det_val) < 1e-10:
        # Try a different approach: greedy selection
        basis_indices = _greedy_basis_selection(coords, num_rank)
        if basis_indices is None:
            return None
        basis_coords = coords[basis_indices, :]
        if abs(np.linalg.det(basis_coords)) < 1e-10:
            return None
    
    if time.time() - elapsed_start > time_budget * 0.7:
        return None
    
    # Now express all non-basis rows as linear combinations of basis rows.
    # For row t not in basis: coords[t, :] = sum_{b in basis} alpha[t, b] * coords[b, :]
    # So alpha[t, :] = coords[t, :] @ inv(basis_coords)
    
    inv_basis = np.linalg.inv(basis_coords)  # (num_rank, num_rank)
    
    # For each row t, compute its coordinates in the basis
    # alpha[t, b] for b in basis_indices
    all_alpha = coords @ inv_basis  # (rank, num_rank)
    
    # Now, the scaling: if we remove row t and express it as sum of basis rows,
    # we need to scale the basis rows to compensate.
    # The tensor decomposition requires:
    #   sum_t U[:,t] * V[:,t] * W[:,t] (as tensor) = the multiplication tensor
    # If T_t = sum_b alpha[t,b] * T_b, then we can remove t and add alpha[t,b] to the scale of b.
    # But we also need to account for the fact that the basis rows themselves may need scaling.
    
    # The correct approach:
    # Let S_b be the scale factor for basis row b.
    # We want: sum_{b in basis} S_b * T_b = sum_{all t} T_t
    # Since T_t = sum_b alpha[t,b] * T_b for all t (including basis rows where alpha is identity),
    # sum_{all t} T_t = sum_{all t} sum_b alpha[t,b] * T_b = sum_b (sum_t alpha[t,b]) * T_b
    # So S_b = sum_t alpha[t,b]
    
    # alpha is (rank, num_rank) where alpha[t, b_idx] is the coefficient of basis row b_idx for row t.
    # S[b_idx] = sum over all t of alpha[t, b_idx]
    
    S = all_alpha.sum(axis=0)  # (num_rank,)
    
    # Check that S is non-zero
    if np.any(np.abs(S) < 1e-12):
        return None
    
    # Build the reduced decomposition
    U_reduced = U[:, basis_indices].copy()
    V_reduced = V[:, basis_indices].copy()
    W_reduced = W[:, basis_indices].copy()
    
    # Apply scaling to U (only one factor needs scaling)
    U_reduced *= S[np.newaxis, :]
    
    if time.time() - elapsed_start > time_budget * 0.85:
        return None
    
    return U_reduced, V_reduced, W_reduced


def _greedy_basis_selection(coords, num_rank):
    """Greedy selection of linearly independent rows from coords."""
    rank = coords.shape[0]
    selected = []
    
    for t in range(rank):
        if len(selected) >= num_rank:
            break
        # Check if coords[t] is in the span of already selected rows
        if not selected:
            selected.append(t)
        else:
            A = coords[selected, :]  # (k, num_rank)
            b = coords[t, :]
            try:
                x, residuals, _, _ = np.linalg.lstsq(A.T, b, rcond=1e-12)
                if residuals.size > 0 and residuals[0] < 1e-10:
                    continue  # dependent, skip
            except np.linalg.LinAlgError:
                continue
            selected.append(t)
    
    if len(selected) < num_rank:
        return None
    return selected[:num_rank]
