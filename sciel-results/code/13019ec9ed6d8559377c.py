"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Schoolbook decomposition (always valid, rank = n*m*p)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = 20
    # Try ranks from lower_bound to best_rank-1
    
    def verify(U, V, W, n, m, p, rank):
        """Check if U, V, W form a valid decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try structured lower-rank decompositions
    # Strategy: for each rank r, try to find U, V, W
    # Use the fact that the system is trilinear - fix two, solve for third
    
    def try_rank(r, n, m, p, max_iters=5000):
        """Try to find a decomposition of rank r."""
        # We need: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # 
        # Approach: parameterize and use random initialization + local search
        # Or: fix W to be structured and solve for U, V
        
        # Try: set W to identity-like structure
        # W has shape (p*n, r) = (15, r)
        # If r >= 15, we can set W[k*n+i, c] = delta_{k*n+i, c} for c < 15
        # Then: U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1 for all i,j,k
        # This requires r >= 15, and for each (i,k), we need:
        #   U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1 for all j
        # Let a_{i,j,k} = U[i*m+j, k*n+i], b_{i,j,k} = V[j*p+k, k*n+i]
        # Need a_{i,j,k} * b_{i,j,k} = 1
        # But U[i*m+j, c] is shared across different k if c = k*n+i for different k?
        # No, c = k*n+i is unique for each (k,i) pair.
        # So for each (k,i), we have independent variables.
        # a_{i,j,k} = U[i*m+j, k*n+i] and b_{i,j,k} = V[j*p+k, k*n+i]
        # These are all independent! So we can set a=1, b=1 for all.
        # This gives rank = p*n = 15? But wait, U has columns indexed by c=0..r-1.
        # If r = 15, then c = k*n+i ranges from 0 to 14, which is valid.
        # Let's check: U[i*m+j, k*n+i] = 1 for all i,j,k
        # V[j*p+k, k*n+i] = 1 for all i,j,k
        # W[k*n+i, k*n+i] = 1, W[k*n+i, c] = 0 for c != k*n+i
        # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = U[i*m+j,k*n+i]*V[j*p+k,k*n+i]*W[k*n+i,k*n+i] = 1*1*1 = 1.
        # This works! Rank = 15!
        
        if r >= p * n:
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c = k * n + i
                        if c < r:
                            U[i * m + j][c] = 1
                            V[j * p + k][c] = 1
            for i in range(n):
                for k in range(p):
                    c = k * n + i
                    if c < r:
                        W[c][c] = 1
            if verify(U, V, W, n, m, p, r):
                return U, V, W
        
        return None
    
    # Try ranks from lower bound upward
    lower_bound = max(n * m, m * p, p * n)
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        result = try_rank(r, n, m, p)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = r
            break  # Found the minimum, stop
    
    return best
# EVOLVE-BLOCK-END
