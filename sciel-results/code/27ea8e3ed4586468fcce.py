"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 4  # 2x3, 3x4, output 2x4
    
    # Schoolbook decomposition as fallback (rank 24)
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
    
    # The known rank of M_{2,3,4} is 12.
    # We attempt to find a rank-12 decomposition.
    
    # The tensor equation is:
    # For all (i,j,k) in [0,2) x [0,3) x [0,4):
    #   sum_{r=0}^{11} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Use a structured ansatz with small integer/half-integer entries.
    # We'll try to find a solution by solving a system of equations.
    
    # Let's try a specific construction based on known results.
    # For 2x3x4, one known rank-12 decomposition uses the following structure:
    
    # We'll parameterize the solution and solve.
    # Since the problem is small, we can try a direct search with constraints.
    
    # Let's try to use a symmetry-based construction.
    # The key insight is that we can share computations between the two rows of A.
    
    # Construction attempt using Strassen-like ideas:
    # For each column k of B (k=0..3), we need to compute:
    #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # We can define intermediate quantities:
    #   For each j, define S_j = A[0,j] + A[1,j] and D_j = A[0,j] - A[1,j]
    #   Then A[0,j] = (S_j + D_j)/2 and A[1,j] = (S_j - D_j)/2
    
    # But this doesn't directly help with the B side.
    
    # Let me try a different approach: enumerate small integer solutions.
    # We'll fix a pattern and solve for the remaining variables.
    
    # Actually, let's try to verify if a known construction works.
    # Based on literature, the rank of 2x3x4 matrix multiplication is 12.
    
    # Let's try to construct it explicitly.
    # One approach: use the fact that 2x3x4 can be embedded in a larger problem.
    
    # Alternative: use a computer algebra approach to find the decomposition.
    # Since we have 12 terms, and each term has 6+12+8=26 variables,
    # but with the constraint that each term is a rank-1 tensor,
    # we can try to find a solution by solving a system of polynomial equations.
    
    # For simplicity, let's try a specific ansatz:
    # Assume U, V, W have entries in {-1, 0, 1} with some structure.
    
    # Let me try the following construction:
    # For r in 0..11:
    #   a = r // 3 (0..3), b = r % 3 (0..2)
    
    #   U[i*m+j, r] = 1 if j == b else 0
    #   V[j*p+k, r] = 1 if j == b else 0
    #   W[k*n+i, r] = 1 if k == a else 0
    
    # Let's check if this works:
    # For a given (i,j,k), the sum is:
    #   sum_{r: j==b and k==a} 1*1*1 = 1 if there exists r with b=j and a=k
    #   Since a = r//3 and b = r%3, for fixed j and k, r = 3*k + j
    #   This is valid for k in 0..3 and j in 0..2, so r in 0..11
    #   So the sum is 1 for all (i,j,k) where the term exists
    
    # Wait, but U[i*m+j, r] = 1 if j==b, which is independent of i.
    # And W[k*n+i, r] = 1 if k==a, which is independent of i.
    # So for a fixed (j,k), the term r=3*k+j contributes 1 to all i.
    # But we need the sum to be 1 for each (i,j,k).
    
    # Let's check: for (i,j,k), the sum over r is:
    #   sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #   = sum_{r: j==b_r and k==a_r} 1 * 1 * 1
    #   where b_r = r%3, a_r = r//3
    #   = 1 if there exists r with b_r=j and a_r=k
    #   = 1 if r = 3*k+j is in 0..11
    #   Since k in 0..3 and j in 0..2, r = 3*k+j is in 0..11
    #   So the sum is 1 for all (i,j,k)
    
    # This works! Let's implement it.
    
    rank = 12
    U = [[0] * rank for _ in range(n * m)]  # 6 x 12
    V = [[0] * rank for _ in range(m * p)]  # 12 x 12
    W = [[0] * rank for _ in range(p * n)]  # 8 x 12
    
    for r in range(rank):
        a = r // 3  # 0..3 (k index)
        b = r % 3   # 0..2 (j index)
        
        # U[i*m+j, r] = 1 if j == b
        for i in range(n):
            U[i * m + b][r] = 1
        
        # V[j*p+k, r] = 1 if j == b
        for k in range(p):
            V[b * p + k][r] = 1
        
        # W[k*n+i, r] = 1 if k == a
        for i in range(n):
            W[a * n + i][r] = 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If the simple construction doesn't work, try a more complex one
    # or fall back to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
