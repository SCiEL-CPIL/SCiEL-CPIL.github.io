import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_half_int(U, V, W, n, m, p, tol=1e-9):
    """Verify with half-integer entries (values must be k/2 for integer k)."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all entries are half-integers
    for arr in [U_arr, V_arr, W_arr]:
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-6):
            return False
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_int(arr):
    """Snap values to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _als_decomposition(n, m, p, rank, max_iters=200, tol=1e-12, seed=0):
    """
    Perform ALS to find a rank decomposition.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    """
    np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                V_j = V_3d[j, :, :]  # (p, rank)
                W_i = W_3d[:, i, :]  # (p, rank)
                M = V_j * W_i  # (p, rank)
                target = np.ones(p)
                try:
                    u, _, _, _ = np.linalg.lstsq(M, target, rcond=None)
                except np.linalg.LinAlgError:
                    u = np.zeros(rank)
                U[i * m + j, :] = u
                U_3d[i, j, :] = u
        
        # Update V
        for j in range(m):
            for k in range(p):
                # M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                U_j = U_3d[:, j, :]  # (n, rank)
                W_k = W_3d[k, :, :]  # (n, rank)
                M = U_j * W_k  # (n, rank)
                target = np.ones(n)
                try:
                    v, _, _, _ = np.linalg.lstsq(M, target, rcond=None)
                except np.linalg.LinAlgError:
                    v = np.zeros(rank)
                V[j * p + k, :] = v
                V_3d[j, k, :] = v
        
        # Update W
        for k in range(p):
            for i in range(n):
                # M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                U_i = U_3d[i, :, :]  # (m, rank)
                V_k = V_3d[:, k, :]  # (m, rank)
                M = U_i * V_k  # (m, rank)
                target = np.ones(m)
                try:
                    w, _, _, _ = np.linalg.lstsq(M, target, rcond=None)
                except np.linalg.LinAlgError:
                    w = np.zeros(rank)
                W[k * n + i, :] = w
                W_3d[k, i, :] = w
        
        # Calculate error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                    err += (1.0 - val) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < tol:
            break
    
    # Try to snap to half-integers
    U_snap = _snap_to_half_int(best_U)
    V_snap = _snap_to_half_int(best_V)
    W_snap = _snap_to_half_int(best_W)
    
    U_list = [[float(x) for x in row] for row in U_snap]
    V_list = [[float(x) for x in row] for row in V_snap]
    W_list = [[float(x) for x in row] for row in W_snap]
    
    if _verify_half_int(U_list, V_list, W_list, n, m, p):
        return (U_list, V_list, W_list), best_err
    
    # Also try integer snapping
    U_int = np.round(best_U)
    V_int = np.round(best_V)
    W_int = np.round(best_W)
    
    U_list_int = [[float(x) for x in row] for row in U_int]
    V_list_int = [[float(x) for x in row] for row in V_int]
    W_list_int = [[float(x) for x in row] for row in W_int]
    
    if _verify_half_int(U_list_int, V_list_int, W_list_int, n, m, p):
        return (U_list_int, V_list_int, W_list_int), best_err
    
    return None, best_err

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _structured_decomposition(n, m, p):
    """
    Try a structured decomposition using the fact that for matrix multiplication,
    we can use a Kronecker-like structure.
    
    For n x m times m x p, one approach is to use:
    rank = n * p + m * (n + p - 1) or similar patterns.
    
    Actually, let's try a simpler structured approach:
    Use the identity that A[i,j] * B[j,k] can be decomposed using
    intermediate sums.
    """
    # Try a decomposition based on summing over j with shared terms
    # This is essentially the schoolbook, but let's see if we can do better
    # with a different pattern.
    
    # For now, return schoolbook as the structured baseline
    return _schoolbook_decomposition(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Known lower bound: max(n*m, m*p, n*p) = max(20, 25, 20) = 25
    # Try ranks from lower bound up to schoolbook
    # We want the MINIMAL rank, so iterate from low to high
    
    # First, try a few known structured patterns
    # For 4x5 * 5x5, known results suggest rank can be below 100
    
    # Try ALS for various ranks
    # Focus on ranks that are likely to work: 25-50
    for rank in range(25, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for restart in range(3):
            if time.time() > deadline:
                break
                
            seed = rank * 1000 + restart * 7 + 42
            result, err = _als_decomposition(n, m, p, rank, max_iters=150, tol=1e-11, seed=seed)
            
            if result is not None:
                U_list, V_list, W_list = result
                # Final verification
                if _verify(U_list, V_list, W_list, n, m, p):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # Found a solution for this rank; since we go from low to high,
            # this is the minimal rank we've found so far
            # But we should continue to see if we can find an even lower one?
            # No, we're iterating from 25 upward, so the first one we find is minimal.
            return best
    
    return best
