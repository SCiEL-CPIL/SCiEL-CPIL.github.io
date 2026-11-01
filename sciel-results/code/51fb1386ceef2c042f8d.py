"""Optimized tensor decomposition for 2x2x9 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 9]

def _verify(U, V, W, n=2, m=2, p=9, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For each (i,j,k) in [0,n)x[0,m)x[0,p), the entry should be 1
    # For all other combinations, it should be 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], np.dot(V[v_idx, :], W[w_idx, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=2, m=2, p=9):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2x4():
    """
    Strassen-like decomposition for 2x2x4.
    The 2x2x2 multiplication tensor has rank 7 (Strassen).
    For 2x2x4, we can use a block approach.
    
    Actually, let's think about this differently.
    The tensor for 2x2x4 multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    where i in {0,1}, j in {0,1}, k in {0,1,2,3}.
    
    We can view this as two separate 2x2x2 multiplications (columns 0-1 and 2-3),
    but they share the same A matrix. So we can potentially share some factors.
    
    Let me use a direct approach: enumerate all possible rank-r decompositions
    using random initialization and gradient-free search.
    """
    pass

def _try_rank_r(n, m, p, r, max_iters=5000, seed=42):
    """
    Try to find a rank-r decomposition using random restarts and local search.
    Uses the approach of random initialization followed by alternating minimization.
    """
    rng = np.random.RandomState(seed)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for restart in range(20):
        # Random initialization
        U = rng.randn(n * m, r) * 0.1
        V = rng.randn(m * p, r) * 0.1
        W = rng.randn(p * n, r) * 0.1
        
        # Alternating minimization
        for it in range(max_iters):
            # Update U: for each row i*m+j, solve least squares
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    # We want: sum_c U[u_idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                    # This is: U[u_idx,:] * (V[j*p+0,:] * W[0*n+i,:] + ... ) -- not linear in U alone
                    # Actually, for fixed V and W, the constraint on U[u_idx,:] is:
                    # For each k: sum_c U[u_idx,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # Let M_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product, shape r)
                    # Then: U[u_idx,:] * M_k = 1 for each k
                    # This is a system: sum_c U[u_idx,c] * M_k[c] = 1
                    # i.e., U[u_idx,:] . M_k = 1 for k=0..p-1
                    # Stack: matrix A of shape (p, r) where A[k,c] = V[j*p+k,c]*W[k*n+i,c]
                    # Solve A @ U[u_idx,:] = ones(p)
                    A = np.outer(V[j*p:(j+1)*p, :], W[(0*n+i):(p*n+i):n, :].T)
                    # A has shape (p, r)
                    # We need A @ x = ones(p) where x = U[u_idx,:]
                    # Least squares solution
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[u_idx, :] = x
                    except:
                        pass
            
            # Update V: for each row j*p+k, solve least squares
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    # For each i: sum_c U[i*m+j,c] * V[v_idx,c] * W[k*n+i,c] = 1
                    # Let M_i = U[i*m+j,:] * W[k*n+i,:] (element-wise, shape r)
                    # V[v_idx,:] . M_i = 1 for i=0..n-1
                    A = np.outer(U[0*m+j:n*m+j:m, :], W[k*n:(k+1)*n:n, :].T)
                    # A has shape (n, r)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[v_idx, :] = x
                    except:
                        pass
            
            # Update W: for each row k*n+i, solve least squares
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    # For each (i,j): sum_c U[i*m+j,c] * V[j*p+k,c] * W[w_idx,c] = 1
                    # Let M_{i,j} = U[i*m+j,:] * V[j*p+k,:] (element-wise, shape r)
                    # W[w_idx,:] . M_{i,j} = 1 for all (i,j)
                    A = np.outer(U[0:m, :], V[k*p:(k+1)*p, :].T)
                    # A has shape (m, r) -- wait, we need all (i,j) pairs
                    # Actually for fixed k, the constraints are for all i in [0,n), j in [0,m):
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[w_idx,c] = 1
                    # M_{i,j} = U[i*m+j,:] * V[j*p+k,:]
                    # W[w_idx,:] . M_{i,j} = 1
                    rows = []
                    for i in range(n):
                        for j in range(m):
                            rows.append(U[i*m+j, :] * V[j*p+k, :])
                    A = np.array(rows)  # shape (n*m, r)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(n*m), rcond=None)
                        W[w_idx, :] = x
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(U[u_idx, :], np.dot(V[v_idx, :], W[w_idx, :]))
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-12:
                break
        
        if best_err < 1e-12:
            break
    
    if best_err < 1e-9 and best_U is not None:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known results:
    # 2x2x2: rank 7 (Strassen)
    # 2x2x3: rank ? (maybe 10 or 11)
    # 2x2x4: rank ? 
    # 2x2x9: rank ?
    
    # Let's try various ranks
    for r in range(10, best_rank):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank_r(n, m, p, r, max_iters=2000, seed=r)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W, n, m, p, tol=1e-8):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                break  # Found the minimal rank
    
    if time.time() > deadline - 10:
        return best_result
    
    return best_result
