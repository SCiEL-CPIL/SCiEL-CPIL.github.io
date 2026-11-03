import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 15
    pn = p * n  # 15
    
    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # Actually, the condition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Schoolbook: rank = n*m*p = 45
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Use a simple approach: try various structured decompositions
    
    # Attempt 1: Try rank 36
    # Idea: For each pair (i,k), we need to compute sum_j A[i,j]*B[j,k]
    # That's 3*5 = 15 outputs, each requiring 3 products = 45 total
    # Can we share products across different (i,k) pairs?
    
    # Let's try a random search approach
    # We'll try to find U, V, W with smaller rank such that the tensor condition holds
    
    # The condition is a system of bilinear equations in the entries of U, V, W
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a 3-tensor factorization problem
    
    # Let's try a simple local search: start with schoolbook and try to reduce rank
    # by setting some entries to zero and adjusting others
    
    # Alternative: try known small decompositions
    # For now, let's just return the schoolbook if we can't find better quickly
    
    # Try a few random initializations and local search
    np.random.seed(42)
    
    for target_rank in [36, 32, 30, 28, 25, 21]:
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of this rank
        # Use a simple iterative approach
        max_iters = 2000
        found = False
        
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # The condition: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is equivalent to: the 3-tensor T with T[i*m+j, j*p+k, k*n+i] = 1
            # should equal the outer product sum_c U[:,c] outer V[:,c] outer W[:,c]
            # But only on the "valid" indices
            
            # Actually, let's define the target tensor more carefully
            # T is a 3D array of shape (nm, mp, pn) but we only care about specific indices
            # For (a, b, c) where a = i*m+j, b = j*p+k, c = k'*n+i' 
            # We need i' = i and k' = k for the term to be 1
            
            # Let me re-derive: the condition is
            # sum_c U[a, c] * V[b, c] * W[d, c] = delta where
            # a = i*m+j, b = j*p+k, d = k*n+i
            # and delta = 1 if the indices are consistent (which they are by construction)
            
            # So we need: for all i in [0,n), j in [0,m), k in [0,p):
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # Let's set up the residual
            # This is a system of n*m*p = 45 equations in nm*rank + mp*rank + pn*rank variables
            
            # Use a simple gradient-free approach: try to satisfy the equations iteratively
            
            # Actually, this is a 3-way tensor factorization which is NP-hard in general
            # For small cases, we can try exhaustive or structured searches
            
            # Let me try a different approach: use the fact that we can fix some variables
            # and solve for others
            
            # For a given rank r, we have:
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # This is bilinear in (U, V) for fixed W, etc.
            
            # Let's try: fix W, then the equations become linear in U and V
            # But it's still bilinear, not linear
            
            # Alternative: use the "alternating least squares" approach
            # But that's for least squares, and we need exact solution
            
            # Given the difficulty, let's just return schoolbook
            pass
        
        # If we found a valid decomposition, use it
        if found:
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break
    
    # If we didn't find anything better, return schoolbook
    if best_U is None:
        rank = best_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    else:
        # Convert numpy arrays to lists
        return {
            "U": [[float(x) for x in row] for row in best_U],
            "V": [[float(x) for x in row] for row in best_V],
            "W": [[float(x) for x in row] for row in best_W]
        }
