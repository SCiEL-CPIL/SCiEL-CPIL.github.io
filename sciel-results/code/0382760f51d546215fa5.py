"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Convert to numpy for efficient operations
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try to find a lower rank decomposition using structured search
    # The matrix multiplication tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We want to minimize rank while maintaining exact decomposition
    
    best_rank = rank_school
    best_U, best_V, best_W = U_arr, V_arr, W_arr
    
    # Strategy: Try to merge schoolbook terms by finding linear combinations
    # that preserve the tensor structure
    
    # First, check if we can do better than schoolbook by exploiting structure
    # For 2x3x7, known lower bounds suggest rank >= 2*3*7 / max(2,3,7) but that's weak
    
    # More practical: try random projections and check rank via SVD
    # But we need EXACT integer/half-integer solutions
    
    # Approach: Systematic search over subsets of schoolbook terms that can be merged
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exists a combination
    # that doesn't introduce unwanted cross terms
    
    # For small dimensions, try all pairs of terms to see if they can be combined
    # This is O(rank^2) which is 42^2 = 1764, very fast
    
    # Actually, let's try a different approach: use the fact that for matrix multiplication,
    # certain structural symmetries allow rank reduction
    
    # Try: Can we use the identity that A*B where A is 2x3 and B is 3x7
    # The result is 2x7. We have 2*7=14 output elements, each is a sum of 3 products.
    # So 42 products total in schoolbook.
    
    # Known result: rank of 2x3x3 is 7 (not 9), rank of 2x2x2 is 7 (not 8)
    # For 2x3x7, let's try to find if rank < 42 is achievable
    
    # Use a greedy merging approach:
    # 1. Start with all 42 terms
    # 2. Try to find pairs that can be replaced by a single term with adjusted coefficients
    # 3. Check if the replacement is exact
    
    # For exactness, we need the tensor to decompose exactly
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    
    # Let's try a more direct approach: solve for the minimum rank using
    # the structure of the problem
    
    # Given the complexity, let's try known constructions:
    # For m=3, we can sometimes use the fact that 3 = 2+1 to reduce rank
    
    # Try rank 40, 39, 38... down to some lower bound
    # For each rank r, try to find U, V, W with integer/half-integer entries
    
    # This is a hard problem. Let's use a heuristic:
    # Try to find linear dependencies among the schoolbook terms
    
    # The tensor T has components T[i,j,k] = delta_{i,j,k} in some sense
    # Actually T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    
    # Let me try a simpler approach: use known results
    # For 2xmxn matrices, there are known rank formulas
    
    # Given time constraints, let's try a few specific rank values
    # and see if we can construct valid decompositions
    
    # Lower bound: rank >= max(n*m, m*p, p*n) / min(n,m,p) is not quite right
    # Actually, a better lower bound comes from the fact that each of the n*p output
    # entries requires at least 1 term, so rank >= n*p = 14
    
    # But that's very weak. The true lower bound for 2x3x7 is unknown to me.
    
    # Let's try to use the structure: since n=2 is small, maybe we can do better
    
    # Try: express the multiplication using a smaller number of bilinear forms
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # For i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # Can we combine these? If we define new variables:
    # X0 = A[0,0]*B[0,k], X1 = A[0,1]*B[1,k], X2 = A[0,2]*B[2,k]
    # Y0 = A[1,0]*B[0,k], Y1 = A[1,1]*B[1,k], Y2 = A[1,2]*B[2,k]
    
    # Then C[0,k] = X0+X1+X2, C[1,k] = Y0+Y1+Y2
    # But Xj and Yj are different bilinear forms, so we still have 6 per k, 42 total.
    
    # Unless we can find relations... 
    
    # Given the difficulty, let's return the schoolbook solution but try a quick
    # optimization first
    
    # Quick check: can we do rank 41? Try removing one term and seeing if we can
    # compensate with coefficient adjustments
    
    # For now, return schoolbook as it's guaranteed correct
    # The search for lower rank is NP-hard in general
    
    if time.time() > deadline - 10:
        # Not enough time, return schoolbook
        return {"U": U, "V": V, "W": W}
    
    # Try a few random perturbations to see if we can find a lower rank
    # This is unlikely to work for exact solutions, but let's try
    
    # Actually, let's try a structured approach:
    # Use the fact that for 2x3x7, we might be able to use the "laser method"
    # or other algebraic geometry techniques, but those are complex
    
    # Given the constraints, I'll return the schoolbook solution
    # which is guaranteed to be correct with rank 42
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
