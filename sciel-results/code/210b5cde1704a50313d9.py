"""Tensor decomposition search for 2x3 * 3x10 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]
n, m, p = DIMENSIONS
N = n * m  # 6
M = m * p  # 30
P = p * n  # 20

# The matrix multiplication tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
# We need U (N x r), V (M x r), W (P x r) such that
# sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
# where the tensor index is (i,j,k) and the W index is k*n+i.

# Rank lower bound: max(N, M, P) = 30
# Known: rank(2x3x10) is at most 60 (schoolbook), and at least 30.
# The true rank is unknown but likely between 30 and 60.
# Let's try to find a decomposition of rank < 60.

def verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    r = U.shape[1]
    # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    return True, None

def try_rank(r, max_iters=2000, seed=42):
    """Try to find a rank-r decomposition using iterative refinement."""
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize with random entries
    U = np.random.randn(N, r)
    V = np.random.randn(M, r)
    W = np.random.randn(P, r)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Update U: for each (i,j), U[i*m+j, :] should satisfy
        # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c] for all k) = 1
        # This is a linear system. Let's form the system properly.
        # For fixed (i,j), we need: U[i*m+j,:] @ diag(V[j*p+k,:]*W[k*n+i,:]) summed over k = 1
        # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        # This is: sum_k [sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]] = ... no
        # It's: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
        
        # So for fixed (i,j), define M_k[c] = V[j*p+k,c] * W[k*n+i,c]
        # Then we need: U[i*m+j,:] @ M_k = 1 for all k
        # This is a system: sum_c U[c] * M_k[c] = 1 for k=0..p-1
        # That's p equations in r unknowns.
        
        # Similarly for V: for fixed (j,k), sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1 for all i
        # And for W: for fixed (k,i), sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all j
        
        # Let's use a simple alternating least squares approach.
        # Update U:
        for i in range(n):
            for j in range(m):
                # System: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                # Solve least squares
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i*m+j, :] = sol
        
        # Update V:
        for j in range(m):
            for k in range(p):
                # System: for each i, sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j*p+k, :] = sol
        
        # Update W:
        for k in range(p):
            for i in range(n):
                # System: for each j, sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k*n+i, :] = sol
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
    
    return best_err, best_U, best_V, best_W

def construct():
    deadline = time.time() + 140  # 140s limit, 10s safety margin
    
    # First, return the schoolbook decomposition as a fallback
    rank_sb = n * m * p  # 60
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # The theoretical lower bound is max(N, M, P) = 30
    # Let's try ranks from 30 to 59
    
    best_result = None
    best_rank = rank_sb
    
    for r in range(30, rank_sb):  # Try ranks 30 to 59
        if time.time() > deadline - 30:
            break
        
        # Try a few seeds
        for seed in [42, 123, 456, 789, 101]:
            if time.time() > deadline - 20:
                break
            
            err, U, V, W = try_rank(r, max_iters=500, seed=seed)
            
            if err < 1e-9:
                # Found a valid decomposition
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
            
            if best_result is None or r < best_rank:
                best_rank = r
                best_result = (U, V, W)
    
    # Return schoolbook as fallback
    return {"U": U_sb, "V": V_sb, "W": W_sb}
