"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 if (i,j,k) corresponds to a valid
    # (input, output) triple, i.e., always 1 for the multiplication tensor.
    # Actually, the multiplication tensor has T[i,j,k] = 1 for all i,j,k where
    # the k-th output depends on A[i,j]*B[j,k].
    # The tensor entries are: for each (i,j,k), T[i,j,k] = 1 (since C[i,k] += A[i,j]*B[j,k])
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _als_decompose(n, m, p, rank, max_iter=2000, tol=1e-12, seed=None):
    """ALS decomposition of the matrix multiplication tensor."""
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    for iteration in range(max_iter):
        # Update U: for each (i,j), U[i*m+j, :] = (V[j*p+0:p, :]^T * V[j*p+0:p, :]^T ... )
        # Actually: for fixed (i,j), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        # This is a linear system in U[i*m+j, :]
        # For fixed (i,j): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
        # Let M_k,c = V[j*p+k,c] * W[k*n+i,c]
        # Then M @ U[i*m+j, :]^T = 1 (vector of ones of length p)
        
        U_new = U.copy()
        for i in range(n):
            for j in range(m):
                # Build matrix: p x rank
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve M @ x = ones(p)
                try:
                    x, residuals, rank_sv, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                except:
                    continue
                U_new[i*m+j, :] = x
        U = U_new
        
        # Update V similarly
        V_new = V.copy()
        for j in range(m):
            for k in range(p):
                # For fixed (j,k): sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1 for all i
                # Let M_i,c = U[i*m+j,c] * W[k*n+i,c]
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, residuals, rank_sv, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                except:
                    continue
                V_new[j*p+k, :] = x
        V = V_new
        
        # Update W similarly
        W_new = W.copy()
        for k in range(p):
            for i in range(n):
                # For fixed (k,i): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all j
                # Let M_j,c = U[i*m+j,c] * V[j*p+k,c]
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, residuals, rank_sv, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except:
                    continue
                W_new[k*n+i, :] = x
        W = W_new
        
        # Check convergence
        if iteration % 10 == 0:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < tol:
                break
    
    # Check if solution is valid
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err < 1e-8:
        # Convert to lists
        U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
        V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
        W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
        return U_list, V_list, W_list, max_err
    return None, None, None, max_err

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start from the schoolbook rank and try to reduce
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # Also rank >= max(n+p-1, m) etc. For 2x3x5, the Strassen-like bounds give rank >= 15
    # Actually the trivial bound is rank >= max(n*m, m*p, p*n) = 15
    
    # Try ranks from the lower bound up to the schoolbook rank
    min_rank = max(n*m, m*p, p*n)
    
    for rank in range(min_rank, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U, V, W, err = _als_decompose(n, m, p, rank, max_iter=500, tol=1e-10, seed=rank*1000+attempt)
            
            if U is not None:
                # Verify with exact arithmetic using fractions
                U_frac = [[Fraction(U[i][c]).limit_denominator(1000000) for c in range(rank)] for i in range(n*m)]
                V_frac = [[Fraction(V[i][c]).limit_denominator(1000000) for c in range(rank)] for i in range(m*p)]
                W_frac = [[Fraction(W[i][c]).limit_denominator(1000000) for c in range(rank)] for i in range(p*n)]
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for c in range(rank):
                                s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                            if s != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = rank
                    best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                    # If we found a solution at the minimum possible rank, we're done
                    if rank == min_rank:
                        return best_result
                    break  # Move to next rank
    
    return best_result
# EVOLVE-BLOCK-END
