"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x5 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

def _validate(U, V, W, n, m, p, tol=1e-8):
    """Check if the factor matrices exactly decompose the matrix multiplication tensor."""
    rank = U[0].__len__() if U else 0
    # Build the multiplication tensor T[i,j,k] = 1 for all i,j,k
    # The decomposition requires: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # We check this for all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Keep a safety margin before 150s limit
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # The multiplication tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # Strategy: Try to find lower-rank decompositions using structured random initialization
    # and iterative refinement. The key insight is that for small matrices, the border rank
    # might be less than the rank, or there might be structured low-rank decompositions.
    
    # For 2x3 x 3x5, the tensor has 2*3*5 = 30 entries, all equal to 1.
    # The schoolbook rank is 30. We try to find lower rank.
    
    # Known lower bounds: The rank of matrix multiplication tensor M_{n,m,p} is at least
    # max(n*m, m*p, p*n) = max(6, 15, 10) = 15.
    # Also, the rank is at least n*m*p / max(n,m,p) = 30/5 = 6 (weak).
    # A better lower bound: rank >= n*m*p / (n+m+p-2) for some cases, but let's just search.
    
    # Try ranks from 15 upward (the trivial lower bound is max(n*m, m*p, p*n) = 15)
    # Actually, the lower bound is more nuanced. Let's try a few candidate ranks.
    
    # For this specific problem, let's try a structured approach:
    # If we can find U, V, W of rank r such that the tensor product matches.
    
    # Use random search with structured patterns
    rng = np.random.RandomState(42)
    
    # Try candidate ranks
    for rank in range(15, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            U = rng.randn(n * m, rank) * 0.1
            V = rng.randn(m * p, rank) * 0.1
            W = rng.randn(p * n, rank) * 0.1
            
            # Iteratively refine: for each factor, solve a least squares problem
            # holding the other two fixed
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each (i,j), we need sum_c U[i*m+j, c] * (V[j*p+*, c] * W[*, i, c]) = 1
                # This is a linear system in U[i*m+j, :]
                # For fixed (i,j), define b_k = 1 for all k
                # A[c, c'] = sum_k V[j*p+k, c] * W[k*n+i, c']
                # Then U[i*m+j, :] should satisfy A * U[i*m+j, :] = b
                # But b is the vector of ones (length p)
                
                # Actually, let's reformulate. For each (i,j,k):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # 
                # For fixed (i,j), let u = U[i*m+j, :], and define for each k:
                # sum_c u_c * V[j*p+k, c] * W[k*n+i, c] = 1
                # This is a linear system: M_{ij} * u = 1 (vector of length p)
                # where M_{ij}[k, c] = V[j*p+k, c] * W[k*n+i, c]
                
                # Solve for U
                U_new = np.zeros((n * m, rank))
                for i in range(n):
                    for j in range(m):
                        # Build M: shape (p, rank)
                        M = np.zeros((p, rank))
                        for k in range(p):
                            for c in range(rank):
                                M[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        # Solve M * u = ones(p)
                        try:
                            u = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                            U_new[i * m + j] = u
                        except:
                            pass
                U = U_new
                
                # Update V similarly
                V_new = np.zeros((m * p, rank))
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, rank))
                        for i in range(n):
                            for c in range(rank):
                                M[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        try:
                            v = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                            V_new[j * p + k] = v
                        except:
                            pass
                V = V_new
                
                # Update W similarly
                W_new = np.zeros((p * n, rank))
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, rank))
                        for j in range(m):
                            for c in range(rank):
                                M[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        try:
                            w = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                            W_new[k * n + i] = w
                        except:
                            pass
                W = W_new
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            err += abs(s - 1.0)
                if err < 1e-6:
                    break
            
            # Convert to lists and validate
            U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n * m)]
            V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m * p)]
            W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p * n)]
            
            if _validate(U_list, V_list, W_list, n, m, p, tol=1e-6):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                break  # Found valid decomposition for this rank, move to next (lower) rank
    
    return best
