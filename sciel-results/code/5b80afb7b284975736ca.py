import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    def verify(U, V, W, R):
        """Check if decomposition is exact with half-integer entries"""
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W, R):
        """Vectorized verification"""
        # Build the full tensor reconstruction
        # T_recon[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Reshape to use einsum
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T_recon, 1.0, atol=1e-9)
    
    # Try ranks from low to high
    for R in range(20, min(64, n*m*p + 1)):
        if time.time() > deadline:
            break
        
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + R * 100 + attempt)
            
            # Initialize
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Alternating minimization
            for it in range(30):
                if time.time() > deadline:
                    break
                
                # Solve for U: for each (i,j), find U[i*m+j,:] such that
                # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                # This is a linear system: A_U @ U[i*m+j,:] = ones(p)
                # where A_U[k, r] = V[j*p+k, r] * W[k*n+i, r]
                
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # A_U is (p, R)
                        A_U = V[j*p:(j+1)*p, :] * W[np.arange(p)*n + i, :]
                        # Solve A_U @ x = ones(p)
                        x, _, _, _ = np.linalg.lstsq(A_U, np.ones(p), rcond=None)
                        U_new[i*m+j, :] = x
                U = U_new
                
                # Solve for V: for each (j,k), find V[j*p+k,:] such that
                # sum_r (U[i*m+j,r] * W[k*n+i,r]) * V[j*p+k,r] = 1 for all i
                # A_V[i, r] = U[i*m+j, r] * W[k*n+i, r]
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        A_V = U[np.arange(n)*m + j, :] * W[k*np.arange(n) + np.arange(n), :]
                        # Wait, W[k*n+i, r] for i in range(n)
                        # W indices: k*n + i for i=0..n-1
                        W_slice = W[k*n:(k+1)*n, :]
                        A_V = U[np.arange(n)*m + j, :] * W_slice
                        x, _, _, _ = np.linalg.lstsq(A_V, np.ones(n), rcond=None)
                        V_new[j*p+k, :] = x
                V = V_new
                
                # Solve for W: for each (k,i), find W[k*n+i,:] such that
                # sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1 for all j
                # A_W[j, r] = U[i*m+j, r] * V[j*p+k, r]
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        A_W = U[np.arange(m)*1 + i, :] * V[np.arange(m)*p + k, :]
                        # U[i*m+j, r] for j=0..m-1: indices i*m + j
                        U_slice = U[i*m:(i+1)*m, :]
                        V_slice = V[np.arange(m)*p + k, :]  # V[j*p+k] for j=0..m-1
                        A_W = U_slice * V_slice
                        x, _, _, _ = np.linalg.lstsq(A_W, np.ones(m), rcond=None)
                        W_new[k*n+i, :] = x
                W = W_new
            
            # Check if we have a good solution
            if verify_fast(U, V, W, R):
                # Round to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                if verify_fast(U_int, V_int, W_int, R):
                    best_rank = R
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    break
            elif verify_fast(U, V, W, R):
                # Try rounding
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                if verify_fast(U_int, V_int, W_int, R):
                    best_rank = R
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    break
        
        if best_rank < n * m * p:
            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
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
