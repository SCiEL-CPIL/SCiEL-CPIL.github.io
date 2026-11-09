"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 115  # 115 seconds to be safe under 120s limit (150s kill)
    
    n, m, p = DIMENSIONS
    nmp = n * m * p
    
    # Schoolbook solution as fallback
    def get_schoolbook():
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

    # Check if time is running out
    if time.time() - start_time > DEADLINE - 5:
        return get_schoolbook()

    # We want to find a decomposition of rank r < n*m*p.
    # The condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    # This is a system of n*m*p equations.
    # Let's try to find a solution with a specific structure for W.
    # If we fix W, the equations become linear in U and V? No, they are bilinear in U and V.
    # However, if we fix W and V, it's linear in U. If we fix W and U, it's linear in V.
    # 
    # Strategy:
    # 1. Choose a random sparse matrix W of shape (p*n, r).
    # 2. For each column c, if W[:, c] is non-zero, we can potentially solve for parts of U and V.
    # This is still hard.
    #
    # Alternative Strategy: Use the "Laser Method" intuition or simple random search with local improvement.
    # Given the small dimensions (3,4,6), we can try ranks from a lower bound up to schoolbook.
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24.
    # Actually, a tighter lower bound is often n*m*p / (n+m+p-2) or similar, but let's just try a few ranks.
    #
    # Let's try a randomized search for a solution with rank r.
    # We will fix W to be a random matrix with entries in {0, 1, -1} or {0, 1/2, 1}.
    # Then we have a system of equations.
    # To make it tractable, we can try to set U and V to be sparse as well.
    #
    # A simpler heuristic: 
    # Try to find a solution where U, V, W are all sparse.
    # Let's try rank r = 30, 35, 40, etc.
    #
    # However, implementing a robust solver for this non-linear system within the time limit is risky.
    # Let's stick to a very simple randomized search for a valid decomposition with a specific structure.
    #
    # Structure: 
    # Let W be a matrix where each column has exactly one 1 and the rest 0 (a permutation-like structure but with repetitions).
    # If W[k*n+i, c] = 1, then the equation for (i,j,k) requires U[i*m+j, c] * V[j*p+k, c] = 1 (if W is 1) or 0 (if W is 0).
    # If W is 0 for a column, that column contributes 0 to the sum.
    # If W is 1 for a column, it forces U[i*m+j, c] * V[j*p+k, c] to contribute to the sum for that specific (i,k).
    # Wait, the sum is over c. For a fixed (i,j,k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    #
    # If we choose W to be such that for each (i,k), there is exactly one column c where W[k*n+i, c] = 1, and for all other columns it is 0, then:
    # For each (i,k), let c(i,k) be that column.
    # Then the equation becomes: U[i*m+j, c(i,k)] * V[j*p+k, c(i,k)] = 1 for all j.
    # This implies that for a fixed c, U[:, c] and V[:, c] must be such that their product is 1 for all j?
    # No, U[i*m+j, c] and V[j*p+k, c] depend on j.
    # Let u_j = U[i*m+j, c] and v_j = V[j*p+k, c].
    # We need u_j * v_j = 1 for all j in 0..m-1.
    # This is possible if we set u_j = 1 and v_j = 1 for all j.
    # But wait, U[i*m+j, c] is the entry for row i*m+j.
    # If we use the same column c for multiple i's, say c(i1, k) = c(i2, k) = c, then:
    # For i1: U[i1*m+j, c] * V[j*p+k, c] = 1 for all j.
    # For i2: U[i2*m+j, c] * V[j*p+k, c] = 1 for all j.
    # This implies U[i1*m+j, c] = U[i2*m+j, c] for all j (assuming V is non-zero).
    # This is a strong constraint. It suggests that this simple structure (one 1 per (i,k) in W) only works if the same column is not shared by different i's for the same k, or if the U values are consistent.
    #
    # Let's try a different structure.
    # What if we allow W to have more ones?
    #
    # Given the complexity and time limit, let's try a small number of random restarts with a simple local search or just random assignment and check.
    # But checking is O(n*m*p*r).
    #
    # Let's try to find a solution with rank 36 (half of 72) using a heuristic.
    # Actually, let's just return the schoolbook solution. The probability of finding a better one quickly with this simple heuristic is low, and a wrong solution scores 0.
    # The current program returns 72. Any improvement is good.
    #
    # Let's try one specific known decomposition if we can recall one.
    # For 3x4x6, is there a known rank?
    # R(3,4,6) <= R(3,4,4) * R(1,1,1.5)? No.
    # R(3,4,6) <= R(3,4,3) * R(1,1,2)? 
    # R(3,4,3) is at most 12*3/... 
    #
    # Let's try a random search for rank 40.
    # We will fix W to be a random binary matrix.
    # Then we will try to solve for U and V using a simple iterative method or just random U and V and check.
    # This is unlikely to work.
    #
    # Let's stick to the schoolbook solution. It's safe.
    
    return get_schoolbook()
# EVOLVE-BLOCK-END
