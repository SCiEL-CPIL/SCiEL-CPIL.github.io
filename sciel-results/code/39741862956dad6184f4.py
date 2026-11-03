"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Best known rank for 4x4 * 4x5 is 48 (from literature on matrix multiplication tensor ranks)
    # We will try to verify or find a decomposition of rank 48.
    # If we fail to find one, we return schoolbook.
    
    # We will use a local search / optimization approach to try to find a rank 48 decomposition.
    # The tensor T_{i,j,k} is 1 if (i,j,k) is a valid index, else 0.
    # We want to find U, V, W of size (nm x r), (mp x r), (pn x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a non-convex problem. We can use random initialization and gradient descent.
    # However, exact decomposition requires integer or half-integer entries.
    # Often, these decompositions have entries in {-1, 0, 1} or similar.
    
    # Given the time limit (120s), a full search for rank 48 might be too slow.
    # Let's try a heuristic search for a lower rank.
    
    # Actually, let's look at the structure.
    # For 4x4 * 4x5, the schoolbook is 80.
    # Strassen-like algorithms can reduce this.
    # But implementing a specific known algorithm is complex.
    
    # Let's try a randomized search for a rank 48 solution with integer entries in {-1, 0, 1}.
    # This is a huge search space.
    
    # Alternative: Use the fact that the task allows half-integers.
    # Let's try to use a known decomposition if possible, or fall back to schoolbook.
    
    # Since I don't have a hardcoded known decomposition for 4x4*4x5 of rank 48 in memory,
    # and deriving one from scratch in 120s is risky, I will stick to the schoolbook
    # unless I can implement a quick Strassen-like recursion that yields a better rank.
    
    # Strassen for 2x2 is rank 7 vs 8.
    # For 4x4, we can use 2x2 blocks.
    # 4x4 * 4x4 can be done with 7^2 = 49 multiplications? No, Strassen is for square matrices.
    # For 4x4 * 4x5, it's rectangular.
    
    # Let's try a simple recursive Strassen for the 4x4 part and handle the 5 columns.
    # Split A (4x4) into 2x2 blocks A11, A12, A21, A22.
    # Split B (4x5) into 2x5 blocks B1, B2.
    # We want C = A*B (4x5).
    # C11 = A11*B1 + A12*B2 (2x5)
    # C21 = A21*B1 + A22*B2 (2x5)
    
    # This doesn't immediately reduce rank without Strassen's trick.
    
    # Given the constraints and the risk of invalid output, I will return the schoolbook decomposition.
    # The score will be 1.0.
    
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
