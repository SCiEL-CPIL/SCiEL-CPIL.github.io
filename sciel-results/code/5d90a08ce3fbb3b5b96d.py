import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _als_decomposition(n, m, p, rank, max_iters=100, tol=1e-10):
    """
    Perform ALS to find a rank decomposition.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    """
    # Initialize randomly
    np.random.seed(0)
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U, V, W
    
    for it in range(max_iters):
        # Update U
        # For each i, j: minimize sum_k (1 - sum_c U[i,j,c] V[j,k,c] W[k,i,c])^2
        for i in range(n):
            for j in range(m):
                # Design matrix M for this (i,j)
                # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Target is 1 for all k
                target = np.ones(p)
                # Solve M @ u = target
                try:
                    u, residuals, rank_m, s = np.linalg.lstsq(M, target, rcond=None)
                except np.linalg.LinAlgError:
                    u = np.zeros(rank)
                U[i * m + j, :] = u
        
        # Update V
        # For each j, k: minimize sum_i (1 - sum_c U[i,j,c] V[j,k,c] W[k,i,c])^2
        for j in range(m):
            for k in range(p):
                # Design matrix M for this (j,k)
                # M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                # Target is 1 for all i
                target = np.ones(n)
                try:
                    v, residuals, rank_m, s = np.linalg.lstsq(M, target, rcond=None)
                except np.linalg.LinAlgError:
                    v = np.zeros(rank)
                V[j * p + k, :] = v
        
        # Update W
        # For each k, i: minimize sum_j (1 - sum_c U[i,j,c] V[j,k,c] W[k,i,c])^2
        for k in range(p):
            for i in range(n):
                # Design matrix M for this (k,i)
                # M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                # Target is 1 for all j
                target = np.ones(m)
                try:
                    w, residuals, rank_m, s = np.linalg.lstsq(M, target, rcond=None)
                except np.linalg.LinAlgError:
                    w = np.zeros(rank)
                W[k * n + i, :] = w
        
        # Calculate error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (1.0 - val) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < tol:
            break
            
    # Try to snap to integers/half-integers
    # Check if best_U, best_V, best_W are close to integers
    # If so, return the snapped version
    def snap_and_verify(U, V, W):
        U_snap = np.round(U)
        V_snap = np.round(V)
        W_snap = np.round(W)
        
        # Verify exactness
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_snap[i * m + j, :], V_snap[j * p + k, :] * W_snap[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return None
        return U_snap, V_snap, W_snap

    snapped = snap_and_verify(best_U, best_V, best_W)
    if snapped:
        return snapped, best_err
    
    # If snapping fails, maybe try half-integers?
    # Or just return the float solution if the error is 0? 
    # The task requires "integer/half-integer entries".
    # If the float solution has very small error but isn't integer, it's not valid.
    # We should only return if we find an exact integer solution.
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # safety margin
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try ALS for lower ranks
    # Lower bound is 25. Try up to 99.
    # We can try ranks in increasing order.
    # To save time, we can try a few random restarts for each rank.
    
    for rank in range(25, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for restart in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(rank * 100 + restart)
            # Call ALS
            result, err = _als_decomposition(n, m, p, rank, max_iters=50, tol=1e-12)
            
            if result:
                U, V, W = result
                # Convert to lists of numbers
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Final verification
                if _verify(U_list, V_list, W_list, n, m, p):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    break # Found a solution for this rank, try lower? No, we go up.
                    # Actually we want the MINIMAL rank. So if we find rank 30, we should check 25-29 first.
                    # Since we iterate rank from 25 upwards, the first one we find is the minimal.
                    return best
                    
    return best
