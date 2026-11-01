"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Try to find low-rank exact decomposition
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    # W index: k*n + i
    
    # First, try small ranks with structured solutions
    # For all-ones tensor, try rank = p (one per k-column)
    # or rank = n*m (one per i,j row)
    
    # Try rank 6: one term per k
    # Term r for k=r: U[i,j,r]=1, V[j,r,r]=1, W[r,i,r]=1
    # Then sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = sum_r 1*delta_{j,r? no}...
    # This doesn't work directly.
    
    # Let's try a different approach: use the structure that T is all-ones
    # We need: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Try rank 4 with a structured solution:
    # Let's use the fact that we can split based on i and j
    
    # Approach: Try to find solutions by setting up the equations
    # For small rank, we can try integer/half-integer values
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank solutions
    # The tensor is all-ones of shape (2,2,6)
    # Key insight: T[i,j,k] = 1 can be written as a sum of rank-1 tensors
    
    # Try rank 6: one rank-1 tensor per k-value
    # For each k, we need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j
    # If we use rank 6 with term r corresponding to k=r:
    # U[i,j,r] = a_{i,j,r}, V[j,r,r] = b_{j,r}, W[r,i,r] = c_{i,r}
    # Then for fixed k=r: sum over r' != r gives 0 (if we set V[j,k,r']=0 for k!=r')
    # So we need: a_{i,j,r} * b_{j,r} * c_{i,r} = 1 for all i,j
    
    # This means a_{i,j,r} * b_{j,r} * c_{i,r} = 1
    # Let a_{i,j,r} = 1, b_{j,r} = 1, c_{i,r} = 1 -> works!
    # But wait, V[j,k,r] must be 0 when k != r for this to work
    # So V[j,k,r] = 1 if k==r else 0
    # Similarly W[k,i,r] = 1 if k==r else 0
    # And U[i,j,r] = 1 for all i,j
    
    # Let's verify: for (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # = sum_r 1 * (1 if k==r else 0) * (1 if k==r else 0)
    # = 1 if there exists r such that r==k, which is true for r=k
    # = 1 * 1 * 1 = 1. Yes!
    
    # So rank 6 works!
    rank_try = p  # 6
    U_try = [[0] * rank_try for _ in range(n * m)]
    V_try = [[0] * rank_try for _ in range(m * p)]
    W_try = [[0] * rank_try for _ in range(p * n)]
    
    for r in range(rank_try):
        for i in range(n):
            for j in range(m):
                U_try[i*m+j][r] = 1
        for j in range(m):
            V_try[j*p+r][r] = 1  # V[j*p+k, r] with k=r
        for i in range(n):
            W_try[r*n+i][r] = 1  # W[k*n+i, r] with k=r
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank_try):
                    s += U_try[i*m+j][r] * V_try[j*p+k][r] * W_try[k*n+i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        best_rank = rank_try
        best_result = {"U": U_try, "V": V_try, "W": W_try}
    
    # Try even lower: rank 4
    # Can we do rank 4? We need 4 rank-1 tensors summing to all-ones (2,2,6)
    # This is more complex. Let's try a structured approach.
    
    # Try rank 3: even more ambitious
    # For now, return the best found
    
    return best_result
# EVOLVE-BLOCK-END
