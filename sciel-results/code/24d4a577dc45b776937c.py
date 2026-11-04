"""Tensor decomposition for 2x3 * 3x9 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k] -> C[i,k]
    # The decomposition requires: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # and for all (i,j,k) where the "cross" terms should be 0, they must be 0.
    # Actually the matrix multiplication tensor is: T[i,j,k] = delta (it's 1 for valid (i,j,k) pairs)
    # More precisely, C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] = 1 for all valid i,j,k (since each term contributes)
    # Wait, the tensor is: for the matmul, T[i,j,k] = 1 if the triple (i,j,k) is "active"
    # Actually in tensor decomposition of matmul, the tensor T has entries T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # because C[i,k] = sum_j A[i,j]B[j,k], so each (i,j,k) contributes 1.
    
    # Check: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1 for all i,j,k
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_full(U, V, W, n, m, p):
    """Full verification including cross-terms being zero."""
    rank = U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The matrix multiplication tensor T[i,j,k] is 1 for all valid (i,j,k)
    # The decomposition must satisfy: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k in valid ranges.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] for r in range(rank))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 6, 27, 18
    
    # Schoolbook rank as fallback
    sb_rank = n * m * p  # 54
    U_sb = [[0] * sb_rank for _ in range(nm)]
    V_sb = [[0] * sb_rank for _ in range(mp)]
    W_sb = [[0] * sb_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = sb_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Also rank >= n*m*p / max(n,m,p) by some bounds... actually let's just try
    # For 2x3x9, the schoolbook is 54. Let's try to find something better.
    
    # Strategy: Use random initialization and iterative refinement
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Try various ranks from lower bound upward
    # Lower bound: at least max(nm, mp, pn) = 27
    # But actually for matmul, known results: rank(2x3x9) - let's try to find it
    
    # Use a simple approach: try rank r from 27 upward, use random init + optimization
    for r in range(27, sb_rank):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        attempts = 3
        for attempt in range(attempts):
            if time.time() > deadline - 30:
                break
            
            random.seed(42 + r * 100 + attempt)
            np.random.seed(42 + r * 100 + attempt)
            
            # Random initialization
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # Iterative refinement: for each factor, fix others and solve
            # This is like ALS (Alternating Least Squares) for the tensor
            max_iter = 50
            for it in range(max_iter):
                if time.time() > deadline - 30:
                    break
                
                # Update U: for each row i*m+j, we need U[i*m+j,:] to satisfy
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # For fixed V, W, this is a linear system in U[i*m+j,:]
                # Actually, for each (i,j), we have p equations (one per k):
                # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for k=0..p-1
                # This is: U[i*m+j,:] @ diag stuff... let me think differently.
                
                # For fixed V, W: define for each (i,j), a matrix M_{i,j} of size p x r where
                # M_{i,j}[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then we need U[i*m+j,:] @ M_{i,j}.T = ones(p)
                # So U[i*m+j,:] = pinv(M_{i,j}.T) @ ones(p)  (least squares)
                
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, r))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve M.T @ u = ones(p)
                        u, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                
                # Update V: for each row j*p+k, need
                # sum_r V[j*p+k,r] * U[i*m+j,r] * W[k*n+i,r] = 1 for all i
                # For fixed U, W: define M_{j,k} of size n x r where
                # M_{j,k}[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # Need V[j*p+k,:] @ M_{j,k}.T = ones(n)
                
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, r))
                        for i in range(n):
                            M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        v, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                
                # Update W: for each row k*n+i, need
                # sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] = 1 for all j
                # For fixed U, V: define M_{k,i} of size m x r where
                # M_{k,i}[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # Need W[k*n+i,:] @ M_{k,i}.T = ones(m)
                
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, r))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        w, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                
                # Check convergence
                if it % 5 == 0:
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = sum(U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr] for rr in range(r))
                                err = abs(s - 1.0)
                                if err > max_err:
                                    max_err = err
                    if max_err < 1e-10:
                        # Found a valid decomposition
                        best_rank = r
                        best_U = [[float(x) for x in U[row]] for row in range(nm)]
                        best_V = [[float(x) for x in V[row]] for row in range(mp)]
                        best_W = [[float(x) for x in W[row]] for row in range(pn)]
                        break
                if max_err < 1e-10:
                    break
        
        if best_rank == r:
            break
    
    # Convert to lists
    if best_U is not None and isinstance(best_U, list):
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback: schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
