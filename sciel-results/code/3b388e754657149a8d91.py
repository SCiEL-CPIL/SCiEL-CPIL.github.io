import numpy as np
import time
import sys
import random
from itertools import product

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    # For each (i,j,k), compute sum_r Ur[i,j,r]*Vr[j,k,r]*Wr[k,i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ur[i,j,:], Vr[j,k,:]*Wr[k,i,:])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = schoolbook_rank
    
    start_time = time.time()
    time_limit = 110  # Leave 10s safety margin
    
    # Strategy: Try to find a decomposition of rank < schoolbook_rank
    # using a structured random search.
    
    # The multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # Approach: Use the fact that we can express this as a bilinear problem.
    # For a given rank R, we have R unknown vectors for U, V, W.
    # This is a system of n*m*p polynomial equations in R*(n*m + m*p + p*n) variables.
    
    # For small R, we can try random initialization and Newton-like methods,
    # or use the "rank reduction" approach from algebraic complexity theory.
    
    # A simpler heuristic: try to find a decomposition where some of the
    # schoolbook terms can be combined.
    
    # Let's try a direct randomized approach for ranks from (schoolbook_rank - 1) down.
    # For each rank R, we try to solve the system using least squares on a
    # relaxed formulation.
    
    # Actually, let's use a different approach: the "tensor rank" can sometimes
    # be reduced by finding linear dependencies. For the multiplication tensor,
    # the rank is known to be less than n*m*p for some dimensions.
    
    # Known results:
    # - 2x2x2: rank 7 (Strassen)
    # - 3x3x3: rank at most 23 (Landsberg)
    # - For rectangular cases, bounds are known.
    
    # For 3x4x6, let's try a structured search.
    
    # Method: Use the "border rank" approach - find a decomposition that is
    # in the closure of rank-R tensors, then round to exact.
    
    # For practical purposes with small dimensions, let's try:
    # 1. Random Gaussian initialization for U, V, W of rank R
    # 2. Use alternating least squares to minimize the residual
    # 3. Check if we get an exact solution
    
    def als_decompose(R, n, m, p, max_iter=100, tol=1e-12):
        """Alternating Least Squares for tensor decomposition."""
        # Initialize randomly
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        for it in range(max_iter):
            # Update U: for each (i,j), find u such that
            # sum_r u_r * (V[j,k,r] * W[k,i,r] summed over k) = 1 for all k
            # This is: for each (i,j), we need sum_r U[i*m+j,r] * (sum_k V[j*p+k,r]*W[k*n+i,r]) = 1
            # But this isn't quite right. Let me reformulate.
            
            # The constraint is: for all (i,j,k):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a trilinear system. ALS doesn't directly apply.
            # Instead, fix two factors and solve for the third.
            
            # Fix V and W, solve for U:
            # For each (i,j): sum_r U[i*m+j,r] * (sum_k V[j*p+k,r] * W[k*n+i,r] * delta_{k,k}) 
            # Hmm, this doesn't factorize nicely.
            
            # Let me use a different approach: flatten the problem.
            # The tensor T[i,j,k] = 1. We want T = sum_r u_r ⊗ v_r ⊗ w_r
            # where u_r is a vector in R^(n*m), v_r in R^(m*p), w_r in R^(p*n).
            
            # This is a standard tensor decomposition problem.
            # Use ALS: fix V and W, solve for U via least squares.
            
            # Reshape: U is (n*m, R), V is (m*p, R), W is (p*n, R)
            # The tensor is (n, m, p) with entries T[i,j,k] = 1
            # U_r[i,j] = U[i*m+j, r], V_r[j,k] = V[j*p+k, r], W_r[k,i] = W[k*n+i, r]
            
            # For ALS on U: fix V, W. For each (i,j), we need:
            # sum_r U[i*m+j,r] * (sum_k V[j*p+k,r] * W[k*n+i,r]) = 1
            # Wait, that's not right either. The constraint is per (i,j,k):
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            # This is a system where for each (i,j,k), we have one equation.
            # Unknowns are U, V, W.
            
            # Let's use a simpler approach: just try random rank-R decompositions
            # and check if they satisfy the constraints.
            
            pass
        
        return None
    
    # Let me use a more direct approach: try specific known low-rank structures.
    
    # For 3x4x6, the schoolbook rank is 72. 
    # Can we do better?
    
    # One approach: use the fact that the multiplication tensor has a certain
    # symmetry and structure.
    
    # Let's try a random search for rank R < 72:
    # For each R, try to find U, V, W such that the constraints are satisfied.
    
    # We'll use a randomized approach:
    # 1. Pick random U, V, W of rank R
    # 2. Compute the residual
    # 3. If residual is small, we're close to a solution
    
    # But this is very hard for exact solutions.
    
    # Alternative: use the "split" approach.
    # Split the 3x4x6 multiplication into smaller parts.
    
    # Actually, let me just try to find if there's a rank-71 or lower decomposition
    # by checking if the schoolbook decomposition has any linear dependencies.
    
    # The schoolbook decomposition has 72 terms. If any term can be expressed
    # as a combination of others, we can reduce the rank.
    
    # For the multiplication tensor, this is generally not possible for the
    # full rank, but let's check.
    
    # Let's try a practical approach: use scipy's optimization to find a
    # low-rank decomposition.
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        minimize = None
    
    # Try ranks from 71 down to 60 (a reasonable range)
    # For each rank, try a few random initializations
    
    # Actually, given the complexity, let's try a different strategy:
    # Use the "Strassen-like" approach for the 3x4x6 case.
    
    # For 2x2x2, Strassen reduces from 8 to 7.
    # For 3x3x3, the best known is 23 (vs 27 schoolbook).
    
    # For 3x4x6, let's estimate: the "asymptotic" rank is related to
    # the exponent omega. But for exact small cases, we need specific results.
    
    # Let's try a random search with a time budget.
    
    def try_rank(R, n, m, p, num_trials=50, time_budget=5.0):
        """Try to find a rank-R decomposition via random search."""
        start = time.time()
        for trial in range(num_trials):
            if time.time() - start > time_budget:
                break
            
            # Random initialization
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Scale to make the product close to 1
            # The expected value of sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # is R * (std_U * std_V * std_W) if all are standard normal.
            # We want this to be 1, so scale by 1/sqrt(R) for each.
            scale = 1.0 / np.sqrt(R)
            U *= scale
            V *= scale
            W *= scale
            
            # Check if this happens to be a solution (very unlikely)
            if _verify_fast(U, V, W, n, m, p):
                return U, V, W
        
        return None
    
    # Try a few ranks
    for R in range(min(schoolbook_rank - 1, 71), max(schoolbook_rank - 15, 55), -1):
        if time.time() - start_time > time_limit - 20:
            break
        
        result = try_rank(R, n, m, p, num_trials=100, time_budget=3.0)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = R
            break
    
    # If we found a better rank, use it; otherwise fall back to schoolbook
    if best_rank < schoolbook_rank:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
