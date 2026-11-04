"""Rewritten program using structured search for lower-rank decompositions.
Uses alternating projections with integer/half-integer rounding to find
rank-minimal exact tensor decompositions for 3x4x6 matrix multiplication.
"""
import numpy as np
from itertools import product
import random
import sys
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Try to find a lower rank decomposition using alternating projections
    # with integer/half-integer rounding
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from schoolbook down to a lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But we know rank >= max(n*m, m*p, p*n) is not tight for matmul
    # A better lower bound: the tensor rank of matmul(n,m,p) >= max(n*m, m*p, p*n)
    # For 3x4x6, known lower bounds suggest rank >= 30 or so
    
    # Start from a few below schoolbook and try to find valid decompositions
    # We'll use a greedy approach: start with rank r, initialize randomly,
    # then use alternating least squares with rounding
    
    for target_rank in range(schoolbook_rank - 1, 23, -1):
        if time.time() > 115:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > 115:
                break
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Normalize initial matrices
            for r in range(target_rank):
                u_norm = np.linalg.norm(U[:, r])
                v_norm = np.linalg.norm(V[:, r])
                w_norm = np.linalg.norm(W[:, r])
                if u_norm > 0:
                    U[:, r] /= u_norm
                if v_norm > 0:
                    V[:, r] /= v_norm
                if w_norm > 0:
                    W[:, r] /= w_norm
            
            # Alternating least squares with rounding
            for iteration in range(200):
                if time.time() > 115:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # For fixed i,j: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
                # This is: U[i*m+j, :] * diag(V[j*p+k,:]) * W[k*n+i,:]... 
                # Actually, for fixed i,j: the equation is
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k in [0,p)
                
                # Let's define for each (i,j): A_{k,r} = V[j*p+k, r] * W[k*n+i, r]
                # Then we need: sum_r U[i*m+j, r] * A_{k,r} = 1 for all k
                # This is a linear system: A @ U[i*m+j, :]^T = ones(p)
                
                for i in range(n):
                    for j in range(m):
                        # Build matrix A of shape (p, target_rank)
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            for r in range(target_rank):
                                A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                        
                        # Solve A @ u = ones(p) in least squares sense
                        try:
                            u, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        except:
                            continue
                        
                        U[i * m + j, :] = u
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                # For fixed j,k: sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for all i
                # Let B_{i,r} = U[i*m+j, r] * W[k*n+i, r]
                # Then: B @ V[j*p+k, :]^T = ones(n)
                
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, target_rank))
                        for i in range(n):
                            for r in range(target_rank):
                                B[i, r] = U[i * m + j, r] * W[k * n + i, r]
                        
                        try:
                            v, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        except:
                            continue
                        
                        V[j * p + k, :] = v
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                # For fixed k,i: sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for all j
                # Let C_{j,r} = U[i*m+j, r] * V[j*p+k, r]
                # Then: C @ W[k*n+i, :]^T = ones(m)
                
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, target_rank))
                        for j in range(m):
                            for r in range(target_rank):
                                C[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        
                        try:
                            w, residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        except:
                            continue
                        
                        W[k * n + i, :] = w
                
                # Round to nearest half-integer
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Check if the decomposition is exact
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            error += abs(val - 1.0)
                
                if error < 1e-6:
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
            
            if best_rank < schoolbook_rank:
                break
        
        if best_rank < schoolbook_rank:
            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: return schoolbook decomposition
    rank = schoolbook_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
