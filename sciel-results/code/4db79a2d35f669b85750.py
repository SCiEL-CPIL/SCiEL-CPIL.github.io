import time
import numpy as np
import itertools

DIMENSIONS = [3, 4, 6]

def _snap_to_half_integers(mat):
    """Snap values to nearest half-integer and return the snapped matrix."""
    return np.round(mat * 2) / 2

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                error += (val - 1.0) ** 2
    return np.sqrt(error) < tol

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Vectorized verification - returns the max absolute error."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _als_step_vectorized(U, V, W, n, m, p, rank):
    """Perform one full ALS cycle with vectorized operations where possible."""
    U_new = np.zeros_like(U)
    V_new = np.zeros_like(V)
    W_new = np.zeros_like(W)
    
    # Update U: for each (i,j), solve A @ U[i*m+j,:] = ones(p)
    # A[k,:] = V[j*p+k,:] * W[k*n+i,:]
    for i in range(n):
        for j in range(m):
            A = V[j*p:(j+1)*p, :] * W[0::p, :]  # W[k*n+i,:] for k=0..p-1
            # W[k*n+i] for fixed i: indices are i, i+n, i+2n, ..., i+(p-1)*n
            A = np.zeros((p, rank))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            try:
                U_new[i * m + j, :], _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
            except:
                U_new[i * m + j, :] = U[i * m + j, :]
    
    # Update V: for each (j,k), solve A @ V[j*p+k,:] = ones(n)
    # A[i,:] = U[i*m+j,:] * W[k*n+i,:]
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, rank))
            for i in range(n):
                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
            try:
                V_new[j * p + k, :], _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
            except:
                V_new[j * p + k, :] = V[j * p + k, :]
    
    # Update W: for each (k,i), solve A @ W[k*n+i,:] = ones(m)
    # A[j,:] = U[i*m+j,:] * V[j*p+k,:]
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, rank))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            try:
                W_new[k * n + i, :], _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
            except:
                W_new[k * n + i, :] = W[k * n + i, :]
    
    return U_new, V_new, W_new

def _try_rank_search(n, m, p, rank, num_restarts=30, timeout=20):
    """
    Attempt to find a rank decomposition using alternating least squares.
    """
    start_time = time.time()
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Initialize with random small integers
        U = np.random.randint(-1, 2, size=(n * m, rank)).astype(float)
        V = np.random.randint(-1, 2, size=(m * p, rank)).astype(float)
        W = np.random.randint(-1, 2, size=(p * n, rank)).astype(float)
        
        # Ensure some non-zero entries
        if np.all(U == 0): U[0, 0] = 1
        if np.all(V == 0): V[0, 0] = 1
        if np.all(W == 0): W[0, 0] = 1
        
        for iteration in range(50):
            if time.time() - start_time > timeout:
                break
            
            U, V, W = _als_step_vectorized(U, V, W, n, m, p, rank)
            
            # Check error
            err = _verify_decomposition_fast(U, V, W, n, m, p)
            if err < 1e-8:
                # Try to snap to half-integers
                U_snap = _snap_to_half_integers(U)
                V_snap = _snap_to_half_integers(V)
                W_snap = _snap_to_half_integers(W)
                if _verify_decomposition_fast(U_snap, V_snap, W_snap, n, m, p) < 1e-7:
                    return U_snap, V_snap, W_snap
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # If we found a good solution but couldn't snap, try more iterations
        if best_error < 1e-6 and best_U is not None:
            U, V, W = best_U, best_V, best_W
            for _ in range(20):
                if time.time() - start_time > timeout:
                    break
                U, V, W = _als_step_vectorized(U, V, W, n, m, p, rank)
                err = _verify_decomposition_fast(U, V, W, n, m, p)
                if err < 1e-8:
                    U_snap = _snap_to_half_integers(U)
                    V_snap = _snap_to_half_integers(V)
                    W_snap = _snap_to_half_integers(W)
                    if _verify_decomposition_fast(U_snap, V_snap, W_snap, n, m, p) < 1e-7:
                        return U_snap, V_snap, W_snap
    
    return None

def _structured_init(n, m, p, rank, seed=42):
    """Create structured initializations exploiting block structure."""
    rng = np.random.RandomState(seed)
    
    # For 3x4x6, we can try patterns where each block (i,j) or (j,k) has structure
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strategy 1: Group by i (rows of A)
    # Each U[i*m+j, :] should be similar for fixed i
    for i in range(n):
        base = rng.randint(-1, 2, size=(rank,)).astype(float)
        for j in range(m):
            U[i * m + j, :] = base + rng.randn(rank) * 0.1
    
    # Strategy 2: Group by j (columns of A / rows of B)  
    for j in range(m):
        base = rng.randint(-1, 2, size=(rank,)).astype(float)
        for k in range(p):
            V[j * p + k, :] = base + rng.randn(rank) * 0.1
    
    # Strategy 3: Group by k (columns of B)
    for k in range(p):
        base = rng.randint(-1, 2, size=(rank,)).astype(float)
        for i in range(n):
            W[k * n + i, :] = base + rng.randn(rank) * 0.1
    
    return U, V, W

def _try_structured_search(n, m, p, rank, num_structured=10, timeout=15):
    """Try structured initializations."""
    start_time = time.time()
    
    for s in range(num_structured):
        if time.time() - start_time > timeout:
            break
        
        U, V, W = _structured_init(n, m, p, rank, seed=s)
        
        for iteration in range(40):
            if time.time() - start_time > timeout:
                break
            U, V, W = _als_step_vectorized(U, V, W, n, m, p, rank)
            err = _verify_decomposition_fast(U, V, W, n, m, p)
            if err < 1e-8:
                U_snap = _snap_to_half_integers(U)
                V_snap = _snap_to_half_integers(V)
                W_snap = _snap_to_half_integers(W)
                if _verify_decomposition_fast(U_snap, V_snap, W_snap, n, m, p) < 1e-7:
                    return U_snap, V_snap, W_snap
    
    return None

def _construct_schoolbook(n, m, p):
    """Fallback schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin (150s limit, leave 20s buffer)
    
    # Lower bound is max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Schoolbook rank is 72
    min_rank = max(n * m, m * p, p * n)
    schoolbook_rank = n * m * p
    
    # Try ranks from low to high
    for r in range(min_rank, min(min_rank + 20, schoolbook_rank + 1)):
        if time.time() > deadline - 10:
            break
        
        # Allocate time based on rank
        time_budget = 15 if r <= 30 else 8
        
        # First try structured search
        result = _try_structured_search(n, m, p, r, num_structured=8, timeout=time_budget // 2)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Then try random restarts
        remaining = min(time_budget - time_budget // 2, deadline - time.time() - 5)
        if remaining > 2:
            result = _try_rank_search(n, m, p, r, num_restarts=20, timeout=remaining)
            if result is not None:
                U, V, W = result
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
