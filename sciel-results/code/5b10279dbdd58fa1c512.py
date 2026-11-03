"""
Matrix multiplication 4,4,5 rank minimization.
Strategy: Start with rank-65 (Strassen 4x4x4 + schoolbook 4x4x1), then use
ALS with rounding to half-integers to search for lower-rank exact decompositions.
"""
import numpy as np
import time
import copy

DIMENSIONS = [4, 4, 5]


def _strassen_2x2_terms():
    """
    Returns the 7 rank-1 terms for 2x2x2 Strassen.
    """
    terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (1,1): -1}, {(0,0): -1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,1): 1})
    ]
    return terms


def _build_4x4x4_strassen():
    """
    Constructs 4x4x4 decomposition using two levels of Strassen. Rank 49.
    """
    n = 4
    base_terms = _strassen_2x2_terms()
    rank = 49
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    for t1 in range(7):
        U1, V1, W1 = base_terms[t1]
        for t2 in range(7):
            U2, V2, W2 = base_terms[t2]
            idx = t1 * 7 + t2
            
            for (i1, j1), c1 in U1.items():
                for (i2, j2), c2 in U2.items():
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    U[i*n + j, idx] += c1 * c2
            
            for (j1, k1), c1 in V1.items():
                for (j2, k2), c2 in V2.items():
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    V[j*n + k, idx] += c1 * c2
            
            for (k1, i1), c1 in W1.items():
                for (k2, i2), c2 in W2.items():
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    W[k*n + i, idx] += c1 * c2
                    
    return U, V, W


def _build_4x4x5_seed():
    """
    Builds the seed rank-65 decomposition: Strassen 4x4x4 (rank 49) + schoolbook 4x4x1 (rank 16).
    """
    n, m, p = 4, 4, 5
    U444, V444, W444 = _build_4x4x4_strassen()
    rank_444 = 49
    rank_441 = n * m
    total_rank = rank_444 + rank_441
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    U[:, :rank_444] = U444
    V[:m*4, :rank_444] = V444
    W[:4*n, :rank_444] = W444
    
    for i in range(n):
        for j in range(m):
            r = rank_444 + i * m + j
            U[i*m + j, r] = 1.0
            V[j*p + 4, r] = 1.0
            W[4*n + i, r] = 1.0
            
    return U, V, W


def _verify(U, V, W, n, m, p, atol=1e-9):
    """
    Verify sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=atol)


def _round_to_half_integer(x):
    """Round a scalar to the nearest half-integer."""
    return np.round(2 * x) / 2.0


def _round_matrix_to_half_int(M):
    """Round all entries of a matrix to nearest half-integer."""
    return np.round(2 * M) / 2.0


def _als_decompose(U0, V0, W0, n, m, p, max_iters=200, tol=1e-10, deadline=None):
    """
    Alternating least squares to refine a tensor decomposition.
    Given initial factors U0, V0, W0 of shape (n*m, R), (m*p, R), (p*n, R),
    iteratively solve for each factor given the others, then round to half-integers.
    Returns refined (U, V, W) or None if it fails to converge to exact.
    """
    R = U0.shape[1]
    U = U0.astype(float).copy()
    V = V0.astype(float).copy()
    W = W0.astype(float).copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return None
        
        # Solve for U: for each (i,j), U[i*m+j, :] should satisfy
        # sum_k V[j*p+k, r] W[k*n+i, r] U[i*m+j, r] = 1 for all k
        # This is: for fixed (i,j), U[i*m+j, :] @ (V[j,:,r] * W[:,i,r]) = ones(p)
        # Actually: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
        # Define M_u[i,j,k,r] = V[j*p+k,r] * W[k*n+i,r]
        # Then sum_r M_u[i,j,k,r] * U[i*m+j,r] = 1 for all k
        # For each (i,j): solve (p x R) system
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build A matrix: A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                A = V[j*p:(j+1)*p, :] * W[:p, i:i+1].T  # shape (p, R)
                b = np.ones(p)
                # Solve A @ x = b for x = U[i*m+j, :]
                try:
                    x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    return None
                U_new[i*m+j, :] = x
        
        # Solve for V: for each (j,k), V[j*p+k, :] should satisfy
        # sum_i U[i*m+j, r] W[k*n+i, r] V[j*p+k, r] = 1 for all i
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                A = U[:, j*m:(j+1)*m].T * W[k*n:(k+1)*n, :].T  # Wait, need to be careful
                # U[i*m+j, r] for i=0..n-1, r=0..R-1: shape (n, R)
                # W[k*n+i, r] for i=0..n-1, r=0..R-1: shape (n, R)
                A = U[:, j] .reshape(n, 1) * W[k*n:(k+1)*n, :]  # shape (n, R)
                # Actually U[:, j] doesn't work. U is (n*m, R). U[i*m+j, r]
                U_col = U.reshape(n, m, R)[:, j, :]  # shape (n, R)
                W_row = W.reshape(p, n, R)[k, :, :]  # shape (n, R)
                A = U_col * W_row  # shape (n, R)
                b = np.ones(n)
                try:
                    x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    return None
                V_new[j*p+k, :] = x
        
        # Solve for W: for each (k,i), W[k*n+i, :] should satisfy
        # sum_j U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all j
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                U_row = U.reshape(n, m, R)[i, :, :]  # shape (m, R)
                V_col = V.reshape(m, p, R)[:, k, :]  # shape (m, R)
                A = U_row * V_col  # shape (m, R)
                b = np.ones(m)
                try:
                    x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    return None
                W_new[k*n+i, :] = x
        
        U = U_new
        V = V_new
        W = W_new
        
        # Check if we've converged to exact
        if _verify(U, V, W, n, m, p, atol=1e-10):
            # Round to half-integers
            U_r = _round_matrix_to_half_int(U)
            V_r = _round_matrix_to_half_int(V)
            W_r = _round_matrix_to_half_int(W)
            if _verify(U_r, V_r, W_r, n, m, p, atol=1e-9):
                return U_r, V_r, W_r
            # Try without rounding first
            return U, V, W
        
        # Check convergence
        if it > 0:
            diff = np.max(np.abs(U - U_prev)) + np.max(np.abs(V - V_prev)) + np.max(np.abs(W - W_prev))
            if diff < tol:
                break
        
        U_prev = U.copy()
        V_prev = V.copy()
        W_prev = W.copy()
    
    # Final check
    if _verify(U, V, W, n, m, p, atol=1e-10):
        U_r = _round_matrix_to_half_int(U)
        V_r = _round_matrix_to_half_int(V)
        W_r = _round_matrix_to_half_int(W)
        if _verify(U_r, V_r, W_r, n, m, p, atol=1e-9):
            return U_r, V_r, W_r
        return U, V, W
    
    return None


def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce the rank from current R to target_rank by zeroing out columns
    and re-solving via ALS.
    """
    R = U.shape[1]
    if target_rank >= R:
        return None
    
    # Try to find a subset of columns of size target_rank that can span the tensor
    # Strategy: iteratively remove columns that are most redundant
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    current_rank = R
    while current_rank > target_rank:
        if time.time() > deadline:
            return None
        
        # Try removing each column and see if we can still decompose
        best_col = None
        best_result = None
        best_err = np.inf
        
        for col in range(current_rank):
            if time.time() > deadline:
                break
            
            # Remove column col
            mask = [i for i in range(current_rank) if i != col]
            U_sub = U_curr[:, mask]
            V_sub = V_curr[:, mask]
            W_sub = W_curr[:, mask]
            
            # Try ALS to see if this sub-decomposition can work
            result = _als_decompose(U_sub, V_sub, W_sub, n, m, p, max_iters=50, deadline=deadline)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify(U_r, V_r, W_r, n, m, p, atol=1e-9):
                    return U_r, V_r, W_r
        
        # If no single column removal works, try a different approach
        # Use ALS on the current factors and check
        result = _als_decompose(U_curr, V_curr, W_curr, n, m, p, max_iters=100, deadline=deadline)
        if result is not None:
            U_curr, V_curr, W_curr = result
            if _verify(U_curr, V_curr, W_curr, n, m, p, atol=1e-9):
                # We have a valid decomposition at current_rank, but we want target_rank
                # For now, return what we have
                pass
        
        # If we can't reduce, break
        break
    
    return None


def _schoolbook_decomposition(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    return U, V, W


def construct():
    """
    Returns the factor matrices for the 4x4x5 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with the seed rank-65 decomposition
    U, V, W = _build_4x4x5_seed()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]  # 65
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = U.shape[1]
    
    # Try ALS refinement on the seed
    if time.time() < deadline - 10:
        result = _als_decompose(U, V, W, n, m, p, max_iters=300, deadline=deadline - 10)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p, atol=1e-9):
                r = U_r.shape[1]
                if r < best_rank:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = r
    
    # Try to reduce rank by trying different initializations
    # Strategy: start with random half-integer initializations at various ranks
    for target_rank in range(best_rank - 1, 48, -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations at this rank
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random half-integers
            np.random.seed(42 + trial)
            U_rand = np.round(2 * np.random.randn(n*m, target_rank) * 0.5) / 2.0
            V_rand = np.round(2 * np.random.randn(m*p, target_rank) * 0.5) / 2.0
            W_rand = np.round(2 * np.random.randn(p*n, target_rank) * 0.5) / 2.0
            
            # Scale to make it more likely to work
            # Actually, let's use a structured initialization
            # Use the first target_rank columns of the seed
            if target_rank <= best_rank:
                U_init = best_U[:, :target_rank].copy()
                V_init = best_V[:, :target_rank].copy()
                W_init = best_W[:, :target_rank].copy()
            else:
                continue
            
            result = _als_decompose(U_init, V_init, W_init, n, m, p, max_iters=200, deadline=deadline - 5)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify(U_r, V_r, W_r, n, m, p, atol=1e-9):
                    r = U_r.shape[1]
                    if r < best_rank:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = r
                        break  # Found a better rank, move to next target
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
