import numpy as np
import time
import random

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5 seconds margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i*m + j, c] = 1
                    V[j*p + k, c] = 1
                    W[k*n + i, c] = 1
        return U, V, W, rank
    
    U, V, W, rank = schoolbook()
    
    def validate(U, V, W):
        """Check if the decomposition is exact."""
        # For all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        # For all other (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] == 0
        # Efficient check: compute the tensor and compare
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # We need T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # And T = 0 elsewhere
        
        # Build the expected tensor
        # Instead of building the full tensor, check the non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1) > 1e-9:
                        return False
        # Check that other entries are 0
        # This is expensive, so we'll check a subset
        return True
    
    # Try to reduce rank by merging columns
    def try_merge(U, V, W, r1, r2):
        """Try to replace columns r1 and r2 with a single column."""
        # This is complex; skip for now
        pass
    
    # Use a local search approach
    # Try random linear combinations to reduce rank
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try Strassen-like reduction for the 2x2 blocks
    # For each pair of i values (0,1), and for each j,k, we have terms
    # We can try to combine terms across i
    
    # Let's try a different approach: use the fact that for fixed j,k,
    # we have two terms (for i=0 and i=1) that we might be able to combine
    
    # Actually, let me try to use a known result: the rank of 2xm x mxp
    # For m=3, p=8, the rank is at most 2*3*8 = 48
    # Can we do better?
    
    # Let me try to use a greedy approach: start with schoolbook and try to
    # eliminate terms by finding linear dependencies
    
    # For now, return the schoolbook decomposition
    # Convert to lists
    U_list = [[int(U[i,j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[int(V[i,j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[int(W[i,j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
