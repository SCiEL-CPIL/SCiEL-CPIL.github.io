import numpy as np
import time

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS
SCHOOLBOOK_RANK = n * m * p

def _schoolbook():
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

def _verify(U, V, W):
    """
    Verify if U, V, W form a valid decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    rank = len(W[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Construct the tensor T[i,j,k]
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # We can compute this by iterating over c or using matrix operations.
    # For each c, the term is U_c V_c W_c (outer product structure? No, it's element-wise product of vectors).
    # Actually, for a fixed c, the contribution to T[i,j,k] is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        # Extract vectors for this c
        # u_vec[i*m+j] = U[i*m+j, c]
        # v_vec[j*p+k] = V[j*p+k, c]
        # w_vec[k*n+i] = W[k*n+i, c]
        
        # Reshape U[:, c] to (n, m)
        u_mat = U[:, c].reshape((n, m))
        # Reshape V[:, c] to (m, p)
        v_mat = V[:, c].reshape((m, p))
        # Reshape W[:, c] to (p, n)
        w_mat = W[:, c].reshape((p, n))
        
        # The term for T[i,j,k] is u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # We need to sum this over c.
        # We can add the contribution of this c to T.
        # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        # Efficient way:
        # For each i, j:
        #   u_val = u_mat[i,j]
        #   If u_val != 0:
        #     For each k:
        #       T[i,j,k] += u_val * v_mat[j,k] * w_mat[k,i]
        
        for i in range(n):
            for j in range(m):
                u_val = u_mat[i, j]
                if u_val != 0:
                    # v_mat[j, :] is a vector of length p
                    # w_mat[:, i] is a vector of length p
                    # w_mat[k, i] for k in 0..p-1
                    w_vec = w_mat[:, i] # length p
                    # product = u_val * (v_mat[j, :] * w_vec)
                    T[i, j, :] += u_val * (v_mat[j, :] * w_vec)
                    
    # Check if T is all ones
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _try_rank(r, time_limit=10):
    """
    Try to find a decomposition of rank r using alternating least squares.
    Returns dict if found, None otherwise.
    """
    start_time = time.time()
    
    # Number of attempts
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize U, V, W randomly
        # Use small random values to avoid explosion
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Alternating Least Squares
        # We want to minimize sum (sum_c U_c V_c W_c - 1)^2
        # This is not convex, but ALS can find local minima.
        
        for it in range(20): # Max iterations per attempt
            if time.time() - start_time > time_limit:
                break
                
            # 1. Update U
            # For each i, j:
            #   Minimize sum_k (sum_c U[i*m+j, c] * K_c[i,j,k] - 1)^2
            #   where K_c[i,j,k] = V[j*p+k, c] * W[k*n+i, c]
            #   This is a linear system: A * U_vec = b
            #   A is p x r, b is p x 1 (all ones)
            #   U_vec is r x 1
            
            for i in range(n):
                for j in range(m):
                    # Build A for this i, j
                    A = np.zeros((p, r))
                    for k in range(p):
                        # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        # Vectorized:
                        # V_row = V[j*p + np.arange(p), :]
                        # W_col = W[np.arange(p)*n + i, :]
                        # A = V_row * W_col
                        v_row = V[j*p + np.arange(p), :]
                        w_col = W[np.arange(p)*n + i, :]
                        A[k, :] = v_row[k, :] * w_col[k, :]
                    
                    b = np.ones(p)
                    # Solve A * U_vec = b
                    # Use least squares
                    U_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m + j, :] = U_vec
            
            # 2. Update V
            # For each j, k:
            #   Minimize sum_i (sum_c L_c[i,j,k] * V[j*p+k, c] - 1)^2
            #   where L_c[i,j,k] = U[i*m+j, c] * W[k*n+i, c]
            #   A is n x r, b is n x 1
            
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        u_row = U[i*m + j, :]
                        w_row = W[k*n + i, :]
                        A[i, :] = u_row * w_row
                    b = np.ones(n)
                    V_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p + k, :] = V_vec
            
            # 3. Update W
            # For each k, i:
            #   Minimize sum_j (sum_c M_c[i,j,k] * W[k*n+i, c] - 1)^2
            #   where M_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c]
            #   A is m x r, b is m x 1
            
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        u_row = U[i*m + j, :]
                        v_row = V[j*p + k, :]
                        A[j, :] = u_row * v_row
                    b = np.ones(m)
                    W_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n + i, :] = W_vec
            
            # Check residual
            # Calculate T
            T = np.zeros((n, m, p))
            for c in range(r):
                u_mat = U[:, c].reshape((n, m))
                v_mat = V[:, c].reshape((m, p))
                w_mat = W[:, c].reshape((p, n))
                for i in range(n):
                    for j in range(m):
                        u_val = u_mat[i, j]
                        if u_val != 0:
                            w_vec = w_mat[:, i]
                            T[i, j, :] += u_val * (v_mat[j, :] * w_vec)
            
            residual = np.sum((T - 1.0)**2)
            if residual < 1e-6:
                # Try to snap to integers/half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
            # Optional: If residual is not decreasing, break
            # For simplicity, we just continue
            
    return None

def construct():
    # Deadline guard
    start_time = time.time()
    DEADLINE = 110  # 10s margin
    
    # Try to find a better rank
    # Start from 59 down to 36 (Strassen bound)
    # We limit the search to a few ranks to stay within time
    for r in range(SCHOOLBOOK_RANK - 1, 35, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Allocate time based on rank
        # Lower ranks are harder, give them more time? Or just fixed small time.
        time_budget = 5.0
        
        result = _try_rank(r, time_limit=time_budget)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook()
