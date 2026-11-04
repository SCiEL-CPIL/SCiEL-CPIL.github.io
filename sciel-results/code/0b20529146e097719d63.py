import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Build the target tensor: T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_entries.append((a, b, c))
    
    num_nonzero = len(nonzero_entries)
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is correct."""
        r = U.shape[1]
        if r != V.shape[1] or r != W.shape[1]:
            return False
        # Check all nonzero entries
        for (a, b, c) in nonzero_entries:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1) > tol:
                return False
        # Check that zero entries remain zero (sample some)
        # For efficiency, check all entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) in set(nonzero_entries):
                        continue
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using tensor operations."""
        r = U.shape[1]
        if r != V.shape[1] or r != W.shape[1]:
            return False
        # Compute the full tensor T[a,b,c] = sum_k U[a,k]*V[b,k]*W[c,k]
        # This is expensive, so let's check only what we need
        # Check nonzero entries
        for (a, b, c) in nonzero_entries:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - 1) > tol:
                return False
        return True
    
    fallback = schoolbook()
    best_result = fallback
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Strategy: Start from schoolbook and try to reduce rank
    # For each pair of columns, check if they can be merged
    
    # Alternative: Use a random search approach
    def try_random_rank(r, max_iters=100):
        """Try to find a rank-r decomposition using random initialization."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Initialize with random values
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Try to satisfy the constraints using a simple iterative method
            # For each nonzero entry (a,b,c): sum_k U[a,k]*V[b,k]*W[c,k] = 1
            
            # Use a sequential update approach
            for iteration in range(100):
                if time.time() > deadline:
                    return None
                
                # Update W
                for c_idx in range(p * n):
                    # Find all (a,b) such that (a,b,c_idx) is in nonzero_entries
                    relevant = [(a, b) for (a, b, c) in nonzero_entries if c == c_idx]
                    if not relevant:
                        continue
                    # For each (a,b), we need: sum_k U[a,k]*V[b,k]*W[c_idx,k] = 1
                    # This is: W[c_idx,:] dot (U[a,:] * V[b,:]) = 1 for each (a,b)
                    # Collect constraints
                    A_mat = np.array([U[a, :] * V[b, :] for (a, b) in relevant])
                    b_vec = np.ones(len(relevant))
                    # Solve least squares
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W[c_idx, :] = sol
                    except:
                        pass
                
                # Update V
                for b_idx in range(m * p):
                    relevant = [(a, c) for (a, b, c) in nonzero_entries if b == b_idx]
                    if not relevant:
                        continue
                    A_mat = np.array([U[a, :] * W[c, :] for (a, c) in relevant])
                    b_vec = np.ones(len(relevant))
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V[b_idx, :] = sol
                    except:
                        pass
                
                # Update U
                for a_idx in range(n * m):
                    relevant = [(b, c) for (a, b, c) in nonzero_entries if a == a_idx]
                    if not relevant:
                        continue
                    A_mat = np.array([V[b, :] * W[c, :] for (b, c) in relevant])
                    b_vec = np.ones(len(relevant))
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U[a_idx, :] = sol
                    except:
                        pass
            
            # Verify
            if verify_fast(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try various ranks from a reasonable lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # But for matrix multiplication, the lower bound is higher
    # For M_{2,2,2}, lower bound is 7
    # For M_{4,5,6}, let's try from 40 upward
    
    for r in range(40, min(121, n*m*p + 1)):
        if time.time() > deadline:
            break
        
        result = try_random_rank(r, max_iters=50)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
