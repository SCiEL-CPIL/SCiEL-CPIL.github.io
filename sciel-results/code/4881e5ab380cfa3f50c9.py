import numpy as np
import time

def construct():
    """
    Rank-reduced decomposition for 4x4 * 4x8 matrix multiplication.
    Starts with rank-98 Kronecker-Strassen decomposition, then attempts
    column elimination to reduce rank while preserving exact decomposition.
    """
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    # Build the base rank-98 Strassen decomposition
    U_base, V_base, W_base = _construct_rank98(n, m, p)
    
    # Verify the base decomposition
    if not _verify_decomposition(U_base, V_base, W_base, n, m, p):
        U_base, V_base, W_base = _construct_naive(n, m, p)
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    
    best_result = (U_base, V_base, W_base)
    best_rank = U_base.shape[1]
    
    # Try column elimination to reduce rank
    elapsed = time.time() - start_time
    if elapsed < deadline - start_time - 20:
        reduced = _column_elimination(U_base, V_base, W_base, n, m, p, deadline - start_time)
        if reduced is not None:
            U_r, V_r, W_r = reduced
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                if U_r.shape[1] < best_rank:
                    best_result = (U_r, V_r, W_r)
                    best_rank = U_r.shape[1]
    
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


def _column_elimination(U, V, W, n, m, p, time_budget):
    """
    Try to eliminate columns from the decomposition.
    For each column t, check if it can be expressed as a linear combination
    of other columns such that the tensor equation still holds.
    
    The key insight: if column t can be written as sum_s alpha_s * column_s,
    then we need U[:,t] = sum_s alpha_s * U[:,s], V[:,t] = sum_s alpha_s * V[:,s],
    and W[:,t] = sum_s alpha_s * W[:,s] simultaneously. This is a strong constraint.
    
    A more practical approach: try removing one column at a time and see if the
    remaining columns can still span the tensor.
    """
    start = time.time()
    rank = U.shape[1]
    
    # Precompute the tensor values for each column
    # For each column t and each (i,j,k), the contribution is U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
    # The total tensor value at (i,j,k) is sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    
    # Build the full tensor as a 3D array
    tensor = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for t in range(rank):
                    total += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                tensor[i, j, k] = total
    
    # Try eliminating columns one at a time
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank
    
    # Try each column for elimination
    for t_remove in range(rank):
        if time.time() - start > time_budget:
            break
        
        if current_rank <= 1:
            break
        
        # Check if we can eliminate column t_remove
        # We need to find coefficients alpha_s (for s != t_remove) such that
        # for all (i,j,k): sum_{s != t_remove} alpha_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
        #   = U[i*m+j,t_remove]*V[j*p+k,t_remove]*W[k*n+i,t_remove]
        
        # This is a linear system in alpha_s
        # Build the system: for each (i,j,k), we have an equation
        # sum_{s != t_remove} alpha_s * (U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]) = U[i*m+j,t_remove]*V[j*p+k,t_remove]*W[k*n+i,t_remove]
        
        # Number of unknowns: current_rank - 1
        # Number of equations: n*m*p = 128
        
        num_unknowns = current_rank - 1
        num_eqs = n * m * p
        
        # Build coefficient matrix A and right-hand side b
        # A[(i,j,k), s_idx] = U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] where s_idx maps to s != t_remove
        # b[(i,j,k)] = U[i*m+j,t_remove]*V[j*p+k,t_remove]*W[k*n+i,t_remove]
        
        # Create index mapping for remaining columns
        remaining_cols = [s for s in range(current_rank) if s != t_remove]
        
        # Build A matrix
        A = np.zeros((num_eqs, num_unknowns), dtype=float)
        b = np.zeros(num_eqs, dtype=float)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    b[eq_idx] = current_U[i*m+j, t_remove] * current_V[j*p+k, t_remove] * current_W[k*n+i, t_remove]
                    for s_idx, s in enumerate(remaining_cols):
                        A[eq_idx, s_idx] = current_U[i*m+j, s] * current_V[j*p+k, s] * current_W[k*n+i, s]
                    eq_idx += 1
        
        # Solve the linear system
        try:
            alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-12)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is valid (residual is small)
        if residuals.size > 0 and residuals[0] > 1e-9:
            continue
        
        # Verify the solution
        A_alpha = A @ alpha
        if not np.allclose(A_alpha, b, atol=1e-9):
            continue
        
        # Eliminate the column
        new_U = np.zeros((n*m, current_rank - 1), dtype=float)
        new_V = np.zeros((m*p, current_rank - 1), dtype=float)
        new_W = np.zeros((p*n, current_rank - 1), dtype=float)
        
        for s_idx, s in enumerate(remaining_cols):
            new_U[:, s_idx] = current_U[:, s]
            new_V[:, s_idx] = current_V[:, s]
            new_W[:, s_idx] = current_W[:, s]
        
        # Scale the columns by alpha
        for s_idx in range(num_unknowns):
            if abs(alpha[s_idx]) > 1e-15:
                new_U[:, s_idx] *= alpha[s_idx]
                new_V[:, s_idx] *= alpha[s_idx]
                new_W[:, s_idx] *= alpha[s_idx]
        
        # Verify the new decomposition
        if _verify_decomposition(new_U, new_V, new_W, n, m, p):
            current_U = new_U
            current_V = new_V
            current_W = new_W
            current_rank -= 1
            # Restart the elimination loop with the new rank
            t_remove = 0
            continue
    
    if current_rank < rank:
        return current_U, current_V, current_W
    
    return None


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
