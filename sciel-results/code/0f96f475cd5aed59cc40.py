"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s
    
    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Start with schoolbook rank 36
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Try to find lower rank decompositions using numerical optimization
    # The tensor is all-ones, so we're looking for a rank-r decomposition
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W correctly decompose the all-ones tensor."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try ranks from 22 upward (known lower bound considerations)
    # For 3x3x4, try ranks 22, 23, 24, ...
    
    np.random.seed(42)
    
    for target_rank in range(22, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        max_trials = 50 if target_rank <= 25 else 20
        if time.time() > deadline - 15:
            max_trials = 5
            
        found = False
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Simple iterative refinement
            # Use alternating minimization style
            for iteration in range(200):
                if time.time() > deadline - 3:
                    break
                    
                # Update U: for each (i,j), minimize sum_k (T[i,j,k] - sum_r U[i,j,r]V[j,k,r]W[k,i,r])^2
                # This is a least squares problem
                for i in range(n):
                    for j in range(m):
                        # Residual for this (i,j)
                        res = np.zeros(p)
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            res[k] = 1.0 - val
                        
                        # Solve: res = U[i*m+j,:] @ (V[j*p+p-1:j*p,:] * W[...])^T
                        # Actually, for fixed i,j: res[k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        # Let M[k,r] = V[j*p+k,r] * W[k*n+i,r], then res = U[i*m+j,:] @ M.T
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            for r in range(target_rank):
                                M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        
                        # U[i*m+j,:] @ M.T = res
                        # U[i*m+j,:] = res @ (M.T @ M)^{-1} @ M.T (least squares)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(M.T, res, rcond=None)
                            U[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V similarly
                for j in range(m):
                    for k in range(p):
                        res = np.zeros(n)
                        for i in range(n):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            res[i] = 1.0 - val
                        
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            for r in range(target_rank):
                                M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        
                        try:
                            sol, _, _, _ = np.linalg.lstsq(M.T, res, rcond=None)
                            V[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W similarly
                for k in range(p):
                    for i in range(n):
                        res = np.zeros(m)
                        for j in range(m):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            res[j] = 1.0 - val
                        
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            for r in range(target_rank):
                                M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        
                        try:
                            sol, _, _, _ = np.linalg.lstsq(M.T, res, rcond=None)
                            W[k*n+i, :] = sol
                        except:
                            pass
                
                # Check convergence
                if verify_decomposition(U, V, W, tol=1e-5):
                    found = True
                    break
            
            if found:
                # Polish with higher precision
                for polish_iter in range(100):
                    if time.time() > deadline - 2:
                        break
                    for i in range(n):
                        for j in range(m):
                            res = np.zeros(p)
                            for k in range(p):
                                val = 0.0
                                for r in range(target_rank):
                                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                res[k] = 1.0 - val
                            M = np.zeros((p, target_rank))
                            for k in range(p):
                                for r in range(target_rank):
                                    M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                            try:
                                sol, _, _, _ = np.linalg.lstsq(M.T, res, rcond=None)
                                U[i*m+j, :] = sol
                            except:
                                pass
                    for j in range(m):
                        for k in range(p):
                            res = np.zeros(n)
                            for i in range(n):
                                val = 0.0
                                for r in range(target_rank):
                                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                res[i] = 1.0 - val
                            M = np.zeros((n, target_rank))
                            for i in range(n):
                                for r in range(target_rank):
                                    M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                            try:
                                sol, _, _, _ = np.linalg.lstsq(M.T, res, rcond=None)
                                V[j*p+k, :] = sol
                            except:
                                pass
                    for k in range(p):
                        for i in range(n):
                            res = np.zeros(m)
                            for j in range(m):
                                val = 0.0
                                for r in range(target_rank):
                                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                res[j] = 1.0 - val
                            M = np.zeros((m, target_rank))
                            for j in range(m):
                                for r in range(target_rank):
                                    M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                            try:
                                sol, _, _, _ = np.linalg.lstsq(M.T, res, rcond=None)
                                W[k*n+i, :] = sol
                            except:
                                pass
                    
                    if verify_decomposition(U, V, W, tol=1e-8):
                        break
                
                if verify_decomposition(U, V, W, tol=1e-7):
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
        
        if found:
            break
    
    # Convert to lists with integer/half-integer rounding if close
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Try to round to nearest 0.5
                rounded = round(val * 2) / 2.0
                if abs(val - rounded) < 1e-6:
                    new_row.append(rounded)
                else:
                    new_row.append(float(val))
            result.append(new_row)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
