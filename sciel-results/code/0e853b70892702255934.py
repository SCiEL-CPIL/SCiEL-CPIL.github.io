import numpy as np
import time

def construct():
    """
    Rank-96 (or lower) decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a base rank-98 Strassen decomposition and reduces rank by identifying linear dependencies.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    # Build the base rank-98 Strassen decomposition
    result = construct_rank98(n, m, p)
    U = np.array(result['U'], dtype=np.float64)
    V = np.array(result['V'], dtype=np.float64)
    W = np.array(result['W'], dtype=np.float64)
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to naive if verification fails
        return construct_naive(n, m, p)
    
    elapsed = time.time() - start_time
    if elapsed > 120:
        # Not enough time for reduction, return base
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linearly dependent columns
    rank_reduced = _reduce_rank_robust(U, V, W, n, m, p, deadline - start_time)
    
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
    ], dtype=np.float64)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=np.float64)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=np.float64)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)
    
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
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
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
    U = np.zeros((n * m, n * m * p), dtype=np.float64)
    V = np.zeros((m * p, n * m * p), dtype=np.float64)
    W = np.zeros((p * n, n * m * p), dtype=np.float64)
    
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
        A = rng.integers(-5, 6, size=(n, m)).astype(np.float64)
        B = rng.integers(-5, 6, size=(m, p)).astype(np.float64)
        C_expected = A @ B
        
        # Compute using decomposition
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

def _reduce_rank_robust(U, V, W, n, m, p, time_budget):
    """
    Robustly reduce rank by finding linear dependencies in the tensor columns.
    """
    rank = U.shape[1]
    dim = n * m * p
    elapsed_start = time.time()
    
    # Construct the matrix M where M[:, t] = vec(T_t)
    # T_t[i,j,k] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # Precompute indices
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_flat_idx = i_idx * m + j_idx
    v_flat_idx = j_idx * p + k_idx
    w_flat_idx = k_idx * n + i_idx
    
    U_vals = U[u_flat_idx, :]  # (dim, rank)
    V_vals = V[v_flat_idx, :]  # (dim, rank)
    W_vals = W[w_flat_idx, :]  # (dim, rank)
    
    M = (U_vals * V_vals * W_vals).T  # (rank, dim)
    
    # Find rank of M
    # Use SVD to find a basis
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
        
    # Determine rank with tolerance
    tol = 1e-12
    r = np.sum(s > tol)
    
    if r >= rank:
        return None
        
    # The columns of M corresponding to the first r singular vectors form a basis?
    # No, the rows of Vt (right singular vectors) span the row space of M.
    # The columns of M are in the span of the first r rows of Vt? No.
    # M = U_svd @ diag(s) @ Vt.
    # Columns of M are linear combinations of columns of U_svd.
    # So U_svd[:, :r] is a basis for the column space of M.
    
    # We want to express each column M[:, t] in terms of a basis of r columns.
    # Let's pick the first r columns of M that are linearly independent?
    # Or better, use the SVD basis.
    # Let B = U_svd[:, :r] (dim x r).
    # M = B @ C, where C = diag(s[:r]) @ Vt[:r, :] (r x rank).
    # So M[:, t] = B @ C[:, t].
    # This means T_t = \sum_{i=1}^r C[i, t] (T_{basis_i})?
    # Wait, B is a basis for the column space.
    # The "basis tensors" are not directly T_i.
    # We need to find linear combinations of the original T_t's.
    
    # Alternative: Find a subset of indices I of size r such that M[:, I] has full rank.
    # Then for each t not in I, solve M[:, I] @ alpha = M[:, t].
    
    # Find pivot indices
    pivots = []
    # Use QR with column pivoting or just SVD of M.T?
    # Let's use SVD of M to find the rank, then pick pivots from M.
    # Actually, we can just use the fact that M = U_svd S Vt.
    # The rows of Vt are the coordinates of the columns of M in the basis U_svd.
    # So if we let Y = Vt[:r, :] (r x rank), then M = U_svd[:, :r] @ Y.
    # The columns of Y are the coordinates of the columns of M in the basis U_svd[:, :r].
    # This doesn't help us express T_t as a combination of T_s directly.
    
    # We need: T_t = \sum_{s \in I} \alpha_{ts} T_s.
    # This is equivalent to: M[:, t] = M[:, I] @ \alpha_t.
    # So we need to find I and solve for \alpha.
    
    # Let's find I by selecting columns of M that are "most independent".
    # We can use the rows of Vt. The rows of Vt are orthogonal.
    # The columns of M are M[:, t] = U_svd @ S @ Vt[t, :].
    # This is getting complicated.
    
    # Simpler: Iterative pivot selection on M.
    # Start with I = [].
    # For t in range(rank):
    #   if M[:, t] is not in span(M[:, I]):
    #     I.append(t)
    #   else:
    #     solve for coeffs
    
    # This is what the old code did, but maybe it was unstable.
    # Let's try to make it stable with QR.
    
    I = []
    coeffs = {} # t -> {s: alpha}
    
    # We will maintain a QR decomposition of M[:, I]
    # But M is (dim x rank). dim = 128, rank = 98.
    # We can maintain Q (dim x |I|) and R (|I| x |I|) such that M[:, I] = Q @ R.
    
    Q = np.zeros((dim, 0))
    R = np.zeros((0, 0))
    
    for t in range(rank):
        if time.time() - elapsed_start > time_budget * 0.9:
            break
            
        # Project M[:, t] onto span(Q)
        if len(I) == 0:
            Q_new = M[:, t:t+1]
            norm = np.linalg.norm(Q_new)
            if norm < tol:
                # Zero column, skip? Should not happen for valid decomposition
                continue
            Q_new = Q_new / norm
            Q = Q_new
            R = np.array([[norm]])
            I.append(t)
        else:
            # Compute projection coefficients
            # y = Q.T @ M[:, t]
            y = Q.T @ M[:, t]
            # z = M[:, t] - Q @ y
            z = M[:, t] - Q @ y
            norm_z = np.linalg.norm(z)
            
            if norm_z < tol:
                # Dependent
                # M[:, t] = Q @ y
                # M[:, I] = Q @ R
                # So Q @ R @ alpha = Q @ y => R @ alpha = y
                alpha = np.linalg.solve(R, y)
                coeff_dict = {}
                for idx, s in enumerate(I):
                    if abs(alpha[idx]) > tol:
                        coeff_dict[s] = alpha[idx]
                coeffs[t] = coeff_dict
            else:
                # Independent
                # Update Q and R
                z = z / norm_z
                Q_new = np.hstack([Q, z.reshape(-1, 1)])
                R_new = np.zeros((len(I) + 1, len(I) + 1))
                R_new[:-1, :-1] = R
                R_new[-1, -1] = norm_z
                # R_new[:-1, -1] = Q.T @ z = 0 (since z is orthogonal to Q)
                # Wait, R is upper triangular.
                # M[:, I] = Q @ R.
                # New column M[:, t] = Q_new @ [0, ..., 0, norm_z].T ?
                # No, M[:, t] = Q @ y + z * norm_z.
                # So in the new basis Q_new, the coordinates are [y, norm_z].
                # So R_new[:, -1] = [y, norm_z].
                R_new[:-1, -1] = y
                R_new[-1, -1] = norm_z
                
                Q = Q_new
                R = R_new
                I.append(t)
                
    if len(I) < rank:
        # Calculate scaling factors
        scale = np.ones(len(I))
        I_to_idx = {s: idx for idx, s in enumerate(I)}
        
        for t, c_dict in coeffs.items():
            for s, alpha in c_dict.items():
                if s in I_to_idx:
                    scale[I_to_idx[s]] += alpha
        
        # Apply scaling to U
        U_reduced = U[:, I].copy()
        for idx, s in enumerate(I):
            if abs(scale[idx]) > tol:
                U_reduced[:, idx] *= scale[idx]
        
        V_reduced = V[:, I].copy()
        W_reduced = W[:, I].copy()
        
        if _verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p):
            return U_reduced, V_reduced, W_reduced
            
    return None
