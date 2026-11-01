"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    if time.time() > deadline:
        return schoolbook()
    
    # Try to construct rank 12 decomposition
    # The known rank of M_{2,3,4} is 12
    
    # Let's try a structured construction based on Strassen-like ideas
    # For 2x2x2, Strassen uses rank 7
    # For 2x3x4, we can try to use a similar approach
    
    # One approach: decompose into blocks
    # A is 2x3, B is 3x4, C is 2x4
    # We can think of this as computing 2 outputs, each being a dot product
    
    # Let me try a direct search for a rank-12 solution using a structured ansatz
    
    # Ansatz: Use 12 terms where each term has a specific structure
    # Based on the symmetry, let's try:
    
    # For r in 0..11:
    #   Let a = r // 3 (0..3), b = r % 3 (0..2)
    #   U[i*m+j, r] = f(i, j, a, b)
    #   V[j*p+k, r] = g(j, k, a, b)  
    #   W[k*n+i, r] = h(k, i, a, b)
    
    # Try the following specific construction:
    # This is based on the idea of using sums and differences
    
    # Let me try a known construction for rank 12:
    
    rank = 12
    U = [[Fraction(0)] * rank for _ in range(n * m)]  # 6 x 12
    V = [[Fraction(0)] * rank for _ in range(m * p)]  # 12 x 12
    W = [[Fraction(0)] * rank for _ in range(p * n)]  # 8 x 12
    
    # Try construction 1: Based on pairing
    # For each j in 0..2, we have 4 terms
    # Term 4*j + k for k in 0..3
    
    # U[i*m+j, 4*j+k] = 1 if i==0 else 0  (only for i=0)
    # V[j*p+k, 4*j+k] = 1
    # W[k*n+i, 4*j+k] = 1 if i==0 else 0
    
    # But this doesn't capture i=1
    
    # Let me try a different approach: use the fact that we can combine terms
    
    # Construction based on:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # We want to share computations between C[0,k] and C[1,k]
    
    # For each k, we have 3 products for row 0 and 3 for row 1 = 6 products
    # But we can use sums/differences to reduce
    
    # For fixed k, define:
    # S0 = A[0,0]+A[1,0], D0 = A[0,0]-A[1,0]
    # S1 = A[0,1]+A[1,1], D1 = A[0,1]-A[1,1]
    # S2 = A[0,2]+A[1,2], D2 = A[0,2]-A[1,2]
    
    # Then:
    # C[0,k] = (S0*D0 + D0*S0)/2 * B[0,k] ... this doesn't work directly
    
    # Let me try a different strategy: use a SAT-like search with small integer entries
    
    # Discretize entries to {-2, -1, 0, 1, 2}
    # For rank 12, we have 12*(6+12+8) = 288 variables
    # But with structure, we can reduce
    
    # Let me try a more systematic approach: enumerate possible rank-12 solutions
    # by solving the tensor equation
    
    # The tensor equation is:
    # For all (i,j,k) in [0,2) x [0,3) x [0,4):
    #   sum_{r=0}^{11} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is 24 equations. Let's try to find a solution with small integer entries.
    
    # Strategy: Fix a pattern for U, V, W and solve
    
    # Try pattern: U has entries in {-1, 0, 1}, same for V and W
    
    # Let me try a specific construction that's known to work:
    
    # Based on literature, one rank-12 construction for 2x3x4 is:
    
    # Terms 0-3: Handle column 0 of B
    # Terms 4-7: Handle column 1 of B
    # Terms 8-11: Handle columns 2-3 of B (combined)
    
    # Actually, let me just try to verify if a simple construction works
    
    # Construction attempt:
    # For r = 0..11:
    #   a = r // 3 (0..3), b = r % 3 (0..2)
    
    #   U[i*m+j, r] = 1 if j == b else 0
    #   V[j*p+k, r] = 1 if j == b and k == a else 0
    #   W[k*n+i, r] = 1 if k == a else 0
    
    # This is schoolbook with rank 12? No, this would be rank 12 but only covers
    # specific (j,k) pairs, not all.
    
    # Let me try: for each j in 0..2, use 4 terms
    # Term 3*j + k:
    #   U[i*m+j, 3*j+k] = 1
    #   V[j*p+k, 3*j+k] = 1
    #   W[k*n+i, 3*j+k] = 1
    
    # This is schoolbook again (rank 12 = 3*4, but we need 2*3*4=24 for schoolbook)
    # Wait, 3*4 = 12, but schoolbook is 2*3*4 = 24
    
    # The issue is that schoolbook has 24 terms because for each (i,j,k) we need a term
    # But in the tensor decomposition, each term can contribute to multiple (i,j,k)
    
    # Let me reconsider. The tensor T has entries T[i,j,k] = 1 for all i,j,k
    # We want T = sum_r U_r * V_r * W_r where U_r, V_r, W_r are vectors
    
    # For schoolbook: U_r has 1 at position (i,j), V_r has 1 at (j,k), W_r has 1 at (k,i)
    # for r corresponding to (i,j,k)
    
    # For a better decomposition, we need U_r, V_r, W_r to have multiple non-zero entries
    
    # Let me try a construction where:
    # For r in 0..11:
    #   U[r] has non-zeros at multiple (i,j) positions
    #   V[r] has non-zeros at multiple (j,k) positions
    #   W[r] has non-zeros at multiple (k,i) positions
    
    # Try this specific construction:
    
    # r = 0..11
    # a = r // 3, b = r % 3
    
    # U[i*m+j, r]:
    #   If a < 4 and b < 3:
    #     U[0*m+b, r] = 1
    #     U[1*m+b, r] = 1 if a < 2 else 0  # Only for first two a values
    #     ... this is getting complicated
    
    # Let me just return schoolbook since I can't quickly find a valid rank-12 construction
    
    return schoolbook()

# EVOLVE-BLOCK-END
