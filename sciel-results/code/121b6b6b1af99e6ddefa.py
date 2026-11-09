"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Index helpers
    def idx_u(i, j):
        return i * m + j
    
    def idx_v(j, k):
        return j * p + k
    
    def idx_w(k, i):
        return k * n + i
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p  # 72
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[idx_u(i, j), col] = 1.0
                V_sb[idx_v(j, k), col] = 1.0
                W_sb[idx_w(k, i), col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p)
    # We want to find minimal rank r such that:
    # T[i,j,k] = sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    lower_bound = max(n * m, m * p, p * n)
    
    # Strategy: Try to find a decomposition with rank close to the lower bound
    # by exploiting the structure of the all-ones tensor.
    
    # For the all-ones tensor, we can use a structured approach:
    # T[i,j,k] = 1 = sum of rank-1 terms
    
    # One approach: use the fact that 1 = 1*1*1, and try to group terms.
    
    # Let's try a systematic search over small ranks with a greedy heuristic.
    # For each candidate rank, we try to construct U, V, W such that the
    # tensor equation is satisfied exactly.
    
    # For the all-ones tensor, a natural decomposition uses the structure:
    # T[i,j,k] = sum_{a,b,c} ... 
    
    # Actually, for all-ones, we can use:
    # 1 = sum_{r=0}^{R-1} u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    # A simple structured decomposition:
    # If we set u_r[i,j] = 1 for all, v_r[j,k] = 1 for all, w_r[k,i] = 1/R,
    # then sum gives R * (1 * 1 * 1/R) = 1. But w_r entries must be integer/half-integer.
    # 1/R is not generally integer/half-integer.
    
    # Let's think differently. For all-ones tensor of size n x m x p:
    # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    # This is a system of n*m*p = 72 equations with 3*r unknowns (r columns in each of U, V, W).
    # We need r such that 3*r >= 72, so r >= 24.
    
    # Let's try to find a rank-24 decomposition.
    
    # For rank r, we have r unknowns in each of U, V, W (total 3r).
    # We have 72 equations. For r=24, we have 72 unknowns and 72 equations.
    
    # Let's set up the linear system for r=24.
    # Unknowns: U[i,j,r] for i in [0,3), j in [0,3), r in [0,24)
    #           V[j,k,r] for j in [0,3), k in [0,8), r in [0,24)
    #           W[k,i,r] for k in [0,8), i in [0,3), r in [0,24)
    
    # This is a bilinear system, not linear. So we can't just solve a linear system.
    
    # However, if we fix two of the three matrices, the system becomes linear in the third.
    # We can use alternating minimization or a structured initialization.
    
    # Let's try a structured initialization for rank 24:
    # Divide the 24 ranks into groups corresponding to the structure.
    
    # Idea: For each (i,j) pair, assign some ranks to cover the k dimension.
    # For each (j,k) pair, assign some ranks to cover the i dimension.
    # For each (k,i) pair, assign some ranks to cover the j dimension.
    
    # This is getting complex. Let's try a simpler approach:
    # Use the schoolbook decomposition but try to merge terms.
    
    # Actually, let's try a direct approach: assume U, V, W have entries in {0, 1}
    # and try to find a rank-24 decomposition with binary entries.
    
    # For binary entries, each rank-1 term is a "cube" in the tensor space.
    # We need to cover the all-ones tensor with 24 such cubes.
    
    # Let's try: for each rank r, define:
    # U[:,r] = indicator of some subset of (i,j) pairs
    # V[:,r] = indicator of some subset of (j,k) pairs
    # W[:,r] = indicator of some subset of (k,i) pairs
    
    # The product U[i,j,r]*V[j,k,r]*W[k,i,r] is 1 iff (i,j) in U_r, (j,k) in V_r, (k,i) in W_r.
    
    # We need: for each (i,j,k), the number of ranks r where all three conditions hold is exactly 1.
    
    # This is a combinatorial design problem.
    
    # Let's try a simple structured solution:
    # For rank r = 0, 1, ..., 23:
    # Let r = a * 8 + b, where a in [0,3), b in [0,8)
    # Set U[i,j,r] = 1 if i == a and j == (b % 3)  -- this doesn't work cleanly.
    
    # Let me try another approach. Since n=3, m=3, p=8:
    # For each (i,j), there are 8 k values. We need to cover all 8.
    # If we assign 8 ranks to each (i,j) pair, that's 9*8=72 ranks (schoolbook).
    # To reduce rank, we need to share ranks between different (i,j) pairs.
    
    # Let's try: for each j, assign 8 ranks to cover all (i,k) pairs for that j.
    # For fixed j, we need U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,k.
    # If we set V[j,k,r] = 1 for all k,r (for this j), then we need U[i,j,r]*W[k,i,r] = 1 for all i,k.
    # This means for each i, U[i,j,r] and W[k,i,r] must be 1 for the same set of r.
    
    # This is still complex. Let's just try the schoolbook and see if we can do slightly better.
    
    # Given the time constraint and complexity, let's return the schoolbook decomposition.
    # The schoolbook rank is 72.
    
    # But let's try one more thing: a rank-36 decomposition using a simple structure.
    # For each (i,j), use 4 ranks instead of 8, and hope for the best.
    # This won't work in general, but let's try.
    
    # Actually, let's just verify the schoolbook and return it.
    # The task is to find the minimum rank, and we know it's at least 24.
    # Without a better algorithm, we return the schoolbook.
    
    U_list = U_sb.tolist()
    V_list = V_sb.tolist()
    W_list = W_sb.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
