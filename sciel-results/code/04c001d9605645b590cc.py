"""Program for matrix multiplication tensor rank minimization (3x5 by 5x6)."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook decomposition as fallback
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    for idx, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
        U[i * m + j, idx] = 1
        V[j * p + k, idx] = 1
        W[k * n + i, idx] = 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank decompositions using random initialization + refinement
    # The tensor T[i,j,k] = 1 for all i,j,k (matrix multiplication tensor)
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Strategy: Try ranks from lower bounds up to schoolbook
    # Lower bound for 3x5x6 matrix multiplication: max(n*m, m*p, n*p) = max(15, 30, 18) = 30
    # But we can potentially do better with structured decompositions
    
    # Try various ranks
    for target_rank in range(30, rank_sb + 1):
        if time.time() > deadline - 5:
            break
        
        # Try random initialization with refinement
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.RandomState(attempt * 12345 + target_rank)
            
            # Initialize with random values
            U_r = rng.randn(n * m, target_rank)
            V_r = rng.randn(m * p, target_rank)
            W_r = rng.randn(p * n, target_rank)
            
            # Use iterative refinement (similar to ALS for tensor decomposition)
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # T[i,j,:] contracted with V and W
                        # U[i*m+j, c] * sum_k V[j*p+k, c] * W[k*n+i, c] = 1
                        # For each c: U[i*m+j, c] = 1 / sum_k V[j*p+k, c] * W[k*n+i, c]
                        denom = np.zeros(target_rank)
                        for k in range(p):
                            denom += V_r[j * p + k, :] * W_r[k * n + i, :]
                        # Avoid division by zero
                        mask = np.abs(denom) > 1e-10
                        U_r[i * m + j, mask] = 1.0 / denom[mask]
                        U_r[i * m + j, ~mask] = 0.0
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        denom = np.zeros(target_rank)
                        for i in range(n):
                            denom += U_r[i * m + j, :] * W_r[k * n + i, :]
                        mask = np.abs(denom) > 1e-10
                        V_r[j * p + k, mask] = 1.0 / denom[mask]
                        V_r[j * p + k, ~mask] = 0.0
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        denom = np.zeros(target_rank)
                        for j in range(m):
                            denom += U_r[i * m + j, :] * V_r[j * p + k, :]
                        mask = np.abs(denom) > 1e-10
                        W_r[k * n + i, mask] = 1.0 / denom[mask]
                        W_r[k * n + i, ~mask] = 0.0
            
            # Check if we got a good solution
            # Verify the decomposition
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(target_rank):
                            val += U_r[i * m + j, c] * V_r[j * p + k, c] * W_r[k * n + i, c]
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-6:
                # Try to round to half-integers
                U_rounded = np.round(U_r * 2) / 2
                V_rounded = np.round(V_r * 2) / 2
                W_rounded = np.round(W_r * 2) / 2
                
                # Verify rounded solution
                max_err_r = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(target_rank):
                                val += U_rounded[i * m + j, c] * V_rounded[j * p + k, c] * W_rounded[k * n + i, c]
                            max_err_r = max(max_err_r, abs(val - 1))
                
                if max_err_r < 1e-9:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded.copy(), V_rounded.copy(), W_rounded.copy()
                    break  # Found a valid decomposition at this rank
            elif max_err < 0.1:
                # Close but not quite - try more iterations
                for iteration in range(500):
                    if time.time() > deadline - 5:
                        break
                    
                    for i in range(n):
                        for j in range(m):
                            denom = np.zeros(target_rank)
                            for k in range(p):
                                denom += V_r[j * p + k, :] * W_r[k * n + i, :]
                            mask = np.abs(denom) > 1e-10
                            U_r[i * m + j, mask] = 1.0 / denom[mask]
                    
                    for j in range(m):
                        for k in range(p):
                            denom = np.zeros(target_rank)
                            for i in range(n):
                                denom += U_r[i * m + j, :] * W_r[k * n + i, :]
                            mask = np.abs(denom) > 1e-10
                            V_r[j * p + k, mask] = 1.0 / denom[mask]
                    
                    for k in range(p):
                        for i in range(n):
                            denom = np.zeros(target_rank)
                            for j in range(m):
                                denom += U_r[i * m + j, :] * V_r[j * p + k, :]
                            mask = np.abs(denom) > 1e-10
                            W_r[k * n + i, mask] = 1.0 / denom[mask]
                
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(target_rank):
                                val += U_r[i * m + j, c] * V_r[j * p + k, c] * W_r[k * n + i, c]
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-6:
                    U_rounded = np.round(U_r * 2) / 2
                    V_rounded = np.round(V_r * 2) / 2
                    W_rounded = np.round(W_r * 2) / 2
                    
                    max_err_r = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for c in range(target_rank):
                                    val += U_rounded[i * m + j, c] * V_rounded[j * p + k, c] * W_rounded[k * n + i, c]
                                max_err_r = max(max_err_r, abs(val - 1))
                    
                    if max_err_r < 1e-9:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_rounded.copy(), V_rounded.copy(), W_rounded.copy()
                        break
    
    # Convert to lists
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
