import time
import numpy as np
import sys

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n, m, p):
    """
    Check the full decomposition identity for M_{n,m,p}.
    """
    r = U.shape[1]
    
    # Check support entries (should be 1)
    # T[(i,j), (j',k), (k',i')] = 1 if i=i', j=j', k=k'
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
                s = float(np.dot(np.multiply(U[i*m+j], V[j*p+k]), W[k*n+i]))
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Check non-support entries (should be 0)
    for idx_u in range(n*m):
        i = idx_u // m
        j = idx_u % m
        for idx_v in range(m*p):
            j_p = idx_v // p
            k = idx_v % p
            for idx_w in range(p*n):
                k_p = idx_w // n
                i_p = idx_w % n
                
                is_support = (j == j_p and k == k_p and i == i_p)
                
                if not is_support:
                    s = float(np.dot(np.multiply(U[idx_u], V[idx_v]), W[idx_w]))
                    if abs(s) > 1e-9:
                        return False
                        
    return True

def _solve_als_233(rank, iterations=200, restarts=20, deadline=None):
    """
    Try to find a rank decomposition for M_{2,3,3} using ALS.
    """
    n, m, p = 2, 3, 3
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for restart in range(restarts):
        if deadline and time.time() > deadline:
            break
            
        # Initialize
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        for it in range(iterations):
            if deadline and time.time() > deadline:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    b = np.ones(p)
                    for k in range(p):
                        A[k, :] = V[j*p + k] * W[k*n + i]
                    U[i*m + j], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    b = np.ones(n)
                    for i in range(n):
                        A[i, :] = U[i*m + j] * W[k*n + i]
                    V[j*p + k], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m + j] * V[j*p + k]
                    W[k*n + i], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Calculate error
        err = 0.0
        for idx_u in range(n*m):
            i = idx_u // m
            j = idx_u % m
            for idx_v in range(m*p):
                j_p = idx_v // p
                k = idx_v % p
                for idx_w in range(p*n):
                    k_p = idx_w // n
                    i_p = idx_w % n
                    s = float(np.dot(np.multiply(U[idx_u], V[idx_v]), W[idx_w]))
                    is_support = (j == j_p and k == k_p and i == i_p)
                    target = 1.0 if is_support else 0.0
                    err += (s - target) ** 2
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if best_err < 1e-10:
                return best_U, best_V, best_W, best_err
                
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140  # Safety margin
    
    # We know M_{2,3,6} is two copies of M_{2,3,3}
    # So we search for M_{2,3,3} with rank R, then use 2R for M_{2,3,6}
    
    # Try ranks for M_{2,3,3}: 12, 13, 14, 15
    # If we find one, we build the 2,3,6 solution
    
    found = False
    U_233 = V_233 = W_233 = None
    R_233 = 0
    
    for R in [12, 13, 14, 15]:
        if time.time() > deadline:
            break
        U_233, V_233, W_233, err = _solve_als_233(R, iterations=100, restarts=10, deadline=deadline)
        if U_233 is not None and err < 1e-9:
            R_233 = R
            found = True
            break
    
    if found:
        # Build M_{2,3,6} from M_{2,3,3}
        # U_236 shape (12, 2*R_233)
        # V_236 shape (18, 2*R_233)
        # W_236 shape (12, 2*R_233)
        
        U_236 = np.zeros((n*m, 2*R_233))
        V_236 = np.zeros((m*p, 2*R_233))
        W_236 = np.zeros((p*n, 2*R_233))
        
        # First half (k=0,1,2)
        U_236[:, 0:R_233] = U_233
        for j in range(m):
            for k in range(3):
                V_236[j*p + k, 0:R_233] = V_233[j*3 + k]
        for k in range(3):
            for i in range(n):
                W_236[k*n + i, 0:R_233] = W_233[k*n + i]
                
        # Second half (k=3,4,5)
        U_236[:, R_233:2*R_233] = U_233
        for j in range(m):
            for k in range(3, 6):
                V_236[j*p + k, R_233:2*R_233] = V_233[j*3 + (k-3)]
        for k in range(3, 6):
            for i in range(n):
                W_236[k*n + i, R_233:2*R_233] = W_233[(k-3)*n + i]
        
        # Verify
        if _verify(U_236, V_236, W_236, n, m, p):
            return {"U": U_236.tolist(), "V": V_236.tolist(), "W": W_236.tolist()}
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
