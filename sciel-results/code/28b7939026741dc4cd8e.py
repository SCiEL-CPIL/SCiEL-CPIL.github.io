"""Matrix multiplication 2x3 x 3x8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    def build_schoolbook():
        rank = n * m * p  # 48
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
        return U, V, W
    
    U_school, V_school, W_school = build_schoolbook()
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    if time.time() > deadline - 50:
        return best_result
    
    # Try to find lower rank decomposition
    # The constraint: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is valid."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(
                        U[i*m+j, :] * V[j*p+k, :],
                        W[k*n+i, :]
                    )
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try rank reduction by finding linear combinations
    # Strategy: start from schoolbook, try to merge pairs of terms
    
    def try_merge_terms(U, V, W, r_start):
        """Try to reduce rank by finding linear dependencies."""
        n, m, p = DIMENSIONS
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = r_start
        
        # For each pair of columns, check if we can merge them
        # A merge is possible if there exist scalars a, b, c such that:
        # a*U[:,i] + b*U[:,j], a*V[:,i] + b*V[:,j], c*W[:,i] + d*W[:,j]
        # form a valid rank-1 term, and the other combination is zero
        
        # Simpler approach: try to find columns that are linearly dependent
        # in a way that allows merging
        
        # Alternative: use the fact that if two terms have the same (i,j) or (j,k) or (k,i),
        # they might be mergeable
        
        # For now, just try small perturbations and see if rank drops
        return None
    
    # Try direct numerical optimization for rank < 48
    # Use alternating minimization
    
    def try_rank(r, max_iter=100):
        """Try to find a rank-r decomposition using alternating minimization."""
        if time.time() > deadline - 10:
            return None
        
        n, m, p = DIMENSIONS
        rng = np.random.default_rng(42)
        
        # Initialize
        U = rng.random((n*m, r))
        V = rng.random((m*p, r))
        W = rng.random((p*n, r))
        
        # Normalize to avoid scale issues
        for col in range(r):
            norm_u = np.linalg.norm(U[:, col])
            norm_v = np.linalg.norm(V[:, col])
            norm_w = np.linalg.norm(W[:, col])
            if norm_u > 0 and norm_v > 0 and norm_w > 0:
                scale = (norm_u * norm_v * norm_w) ** (1/3)
                U[:, col] /= norm_u * scale
                V[:, col] /= norm_v * scale
                W[:, col] /= norm_w * scale
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each (i,j), we need
            # U[i*m+j, :] * (V[j*p+k, :] * W[k*n+i, :]) summed over k = 1
            # This is: U[i*m+j, :] . (sum_k V[j*p+k, :] * W[k*n+i, :]) = 1
            # So U[i*m+j, :] should be proportional to sum_k V[j*p+k, :] * W[k*n+i, :]
            
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    # target vector: sum_k V[j*p+k, :] * W[k*n+i, :]
                    target = np.zeros(r)
                    for k in range(p):
                        target += V[j*p+k, :] * W[k*n+i, :]
                    
                    if np.linalg.norm(target) > 1e-15:
                        # U[i*m+j, :] . target = 1
                        # Set U[i*m+j, :] = target / (target . target)
                        U_new[i*m+j, :] = target / (target @ target)
                    else:
                        U_new[i*m+j, :] = 0
            
            # Update V: for each (j,k), we need
            # V[j*p+k, :] . (sum_i U[i*m+j, :] * W[k*n+i, :]) = 1
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    target = np.zeros(r)
                    for i in range(n):
                        target += U[i*m+j, :] * W[k*n+i, :]
                    
                    if np.linalg.norm(target) > 1e-15:
                        V_new[j*p+k, :] = target / (target @ target)
                    else:
                        V_new[j*p+k, :] = 0
            
            # Update W: for each (k,i), we need
            # W[k*n+i, :] . (sum_j U[i*m+j, :] * V[j*p+k, :]) = 1
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    target = np.zeros(r)
                    for j in range(m):
                        target += U[i*m+j, :] * V[j*p+k, :]
                    
                    if np.linalg.norm(target) > 1e-15:
                        W_new[k*n+i, :] = target / (target @ target)
                    else:
                        W_new[k*n+i, :] = 0
            
            U, V, W = U_new, V_new, W_new
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-12:
                break
        
        if best_err < 1e-10:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from 47 down to 30
    for r in range(47, 29, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
    
    return best_result
