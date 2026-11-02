import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    np.random.seed(42)
    rng = random.Random(42)
    
    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition."""
        r = len(U[0])
        U_arr = np.array(U, dtype=np.float64).reshape(n, m, r)
        V_arr = np.array(V, dtype=np.float64).reshape(m, p, r)
        W_arr = np.array(W, dtype=np.float64).reshape(p, n, r)
        
        Ue = U_arr.reshape(n, m, 1, r)
        Ve = V_arr.reshape(1, m, p, r)
        We = W_arr.T.reshape(n, 1, p, r)
        
        T = np.sum(Ue * Ve * We, axis=3)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    def als_decompose(target_rank, max_iters=500, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        r = target_rank
        
        # Initialize with random values
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Solve for U: for each (i,j), U[i*m+j,:] should satisfy
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is: U[i*m+j,:] @ (V[j*p+k,:] * W[k*n+i,:]) = 1
            # For fixed (i,j), we have a linear system in U[i*m+j,:]
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Build the linear system: A @ u = b
                    # where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # and b[k] = 1
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    # Solve A @ u = b using least squares
                    u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i * m + j, :] = u
            
            # Solve for V: for each (j,k), V[j*p+k,:] should satisfy
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j * p + k, :] = v
            
            # Solve for W: for each (k,i), W[k*n+i,:] should satisfy
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k * n + i, :] = w
            
            # Check convergence
            change = (np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W))
            U, V, W = U_new, V_new, W_new
            
            if change < tol:
                break
        
        # Verify
        if verify(U.tolist(), V.tolist(), W.tolist()):
            return U, V, W, True
        return U, V, W, False
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Try ranks from lower bound to schoolbook
    lower_bound = max(n * p, m * p, n * m)
    best_result = None
    best_rank = n * m * p + 1
    
    for target_rank in range(lower_bound, n * m * p + 1):
        if time.time() > deadline - 10:
            break
        
        success = False
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(restart * 1000 + target_rank)
            U, V, W, ok = als_decompose(target_rank, max_iters=300)
            
            if ok:
                success = True
                best_result = (U, V, W)
                best_rank = target_rank
                break
        
        if success:
            break
    
    if best_result is None:
        # Fallback to schoolbook
        U, V, W, rank = schoolbook()
        best_result = (U, V, W)
        best_rank = rank
    
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
