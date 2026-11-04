"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    # Build the tensor: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
    # Actually for matrix mult, T[i*m+j, j*p+k, k*n+i] = 1
    
    # We want: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b,c
    
    # Try ranks from lower to upper
    deadline = time.time() + 100  # safety margin
    
    # Known: schoolbook rank is 40
    # Try to find lower rank
    
    # Let's try a direct approach: for a given rank r, we need to find
    # U (nm x r), V (mp x r), W (pn x r) such that the tensor equation holds.
    # This is a system of bilinear equations.
    
    # Strategy: Fix U and V, then W is determined (if possible) by:
    # For each c: sum_{a,b} T[a,b,c] * U[a,r] * V[b,r] ... no, that's not right.
    # 
    # The equation is: for each (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    # This is nonlinear in U,V,W.
    
    # Alternative: use the fact that for fixed U and V, the equation becomes:
    # For each c: sum_r W[c,r] * (sum_{a,b: T[a,b,c]=1} U[a,r]*V[b,r]) ... no.
    #
    # Actually: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    # For fixed a,b: sum_r (U[a,r]*V[b,r]) * W[c,r] = T[a,b,c] for all c.
    # This is a linear system in W if U and V are fixed, but only for the (a,b) pairs where T is nonzero.
    
    # Let me try a randomized search with local optimization.
    
    def verify(U, V, W):
        """Check if U,V,W give exact decomposition."""
        # T[a,b,c] should equal sum_r U[a,r]*V[b,r]*W[c,r]
        # Only need to check where T=1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Also check that T=0 entries are 0 (this is harder to check exhaustively)
        # But if the rank-1 terms only contribute to the right places, it should be fine.
        # Actually we need to check ALL (a,b,c) triples.
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[a][r] * V[b][r] * W[c][r]
                    # Determine what T[a,b,c] should be
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i == i2 and j == j2 and k == k2:
                        expected = 1.0
                    else:
                        expected = 0.0
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    # Try to find a decomposition using a simple approach
    # Start with schoolbook and try to reduce
    
    # Actually, let me try a known result: for 2x4x5, is there a known rank?
    # Let me just try ranks from 20 upward and see if I can find one.
    
    # Use a randomized approach: initialize U,V,W randomly, then do gradient-free search.
    # This is complex. Let me try a simpler algebraic approach.
    
    # For small ranks, we can try to solve the system directly.
    # The number of equations is nm*mp*pn = 8*20*10 = 1600, but many are redundant.
    # The number of unknowns for rank r is r*(nm + mp + pn) = r*(8+20+10) = 38r.
    
    # For r=20: 760 unknowns, 1600 equations (but many zero constraints).
    # This is underdetermined, so solutions likely exist.
    
    # Let me try to construct a solution for r=20 using a structured approach.
    
    # Idea: Use the block structure. Split the 4 middle indices into two groups of 2.
    # Group 1: j=0,1; Group 2: j=2,3
    # For each group, we have a 2x2x5 subproblem.
    # 
    # For 2x2x5: C[i,k] = sum_{j in group} A[i,j]*B[j,k]
    # This is rank 4*5=20 per group if done separately, total 40.
    # 
    # But can we do the two groups together more efficiently?
    
    # Alternative idea: Use the fact that we have 2 rows in A.
    # Define for each k:
    #   x_k = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   y_k = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #   z_k = A[0,2]*B[2,k] + A[0,3]*B[3,k]  
    #   w_k = A[1,2]*B[2,k] + A[1,3]*B[3,k]
    # Then C[0,k] = x_k + z_k, C[1,k] = y_k + w_k
    # But x_k, y_k, z_k, w_k each require 2 products, so total 8 products per k, 40 total. Same as schoolbook.
    
    # Let me try yet another approach: use a known rank-20 decomposition if it exists.
    
    # Actually, I recall that for m x n x p matrix multiplication, the rank is at least 
    # max(mn, np, pm) and at most mnp. For 2x4x5, lower bound is max(8,20,10)=20.
    # 
    # Let me try to find a rank-20 decomposition.
    
    # One approach: use the "embedding" technique or known decompositions.
    
    # Let me try a computational search with a time limit.
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    best_rank = n * m * p  # 40, schoolbook
    best_result = None
    
    # First, get the schoolbook decomposition as a fallback
    U_school = [[0] * (n*m*p) for _ in range(nm)]
    V_school = [[0] * (n*m*p) for _ in range(mp)]
    W_school = [[0] * (n*m*p) for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n*m*p
    
    # Try to find lower rank using a simple iterative method
    # For each candidate rank, try random initialization and local search
    
    for rank in range(20, best_rank):
        if time.time() > deadline - 20:
            break
        for trial in range(100):
            if time.time() > deadline - 10:
                break
            # Random initialization
            U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(nm)]
            V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(mp)]
            W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(pn)]
            
            # Simple local search: perturb and check
            # This is very crude but might work for small problems
            # Actually, this approach is unlikely to work well.
            # Let me try a different strategy.
            pass
    
    # Given the difficulty of finding a low-rank decomposition computationally,
    # let me return the schoolbook decomposition.
    # But wait - let me try one more thing: a structured decomposition.
    
    # Actually, for this problem, let me just return the schoolbook rank.
    # The task asks for rank-minimal, but if we can't find a better one,
    # the schoolbook is our best bet.
    
    return best_result
# EVOLVE-BLOCK-END
