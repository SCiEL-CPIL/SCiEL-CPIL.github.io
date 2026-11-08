"""
Matrix multiplication tensor decomposition for 3x4 * 4x7.
We seek a rank-minimal exact decomposition.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]
n, m, p = DIMENSIONS
N = n * m  # 12
M = m * p  # 28
P = p * n  # 21

# The tensor T[i, j, k] = 1 if (i,j,k) is a valid index, but the structure is:
# T[i, j, k] corresponds to the entry in the product matrix C[i, k] from A[i, j] * B[j, k].
# Specifically, the tensor T is of shape (n, m, p) such that T[i, j, k] = 1.
# The decomposition is:
# T[i, j, k] = sum_{r=0}^{R-1} U[i, r] * V[j, r] * W[k, r]
# where U is (n, R), V is (m, R), W is (p, R).
# Wait, the interface says:
# U shape (n*m, rank) -> indexed by i*m + j
# V shape (m*p, rank) -> indexed by j*p + k
# W shape (p*n, rank) -> indexed by k*n + i
# And the condition is:
# sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
# This is equivalent to the standard tensor decomposition of the matrix multiplication tensor.
# Let's re-verify the indices.
# The product C[i, k] = sum_j A[i, j] B[j, k].
# The tensor T has indices (i, j, k) and value 1.
# If we flatten U to (n*m, R), V to (m*p, R), W to (p*n, R),
# then the term for a specific (i, j, k) is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
# This matches the requirement that the sum over r equals 1 for all (i, j, k).

# Known lower bounds for rank of 3x4x4 matrix multiplication tensor:
# Strassen's algorithm gives rank 7 for 2x2x2.
# For 3x3x3, the rank is known to be 23 (Landsberg).
# For 3x4x7, we can try to construct a decomposition with rank less than 3*4*7 = 84.
# We can use a random search or a structured search.
# Since we need an *exact* decomposition, we can use a randomized algorithm that tries to find a rank-R decomposition by solving a system of linear equations or using numerical optimization and then rounding to integers/half-integers.

# A common strategy is to use the "random rank decomposition" approach.
# We can generate random vectors for U, V, W and then solve for the coefficients.
# However, the condition is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
# This is a multilinear system.

# Let's try a simpler approach: use the known decomposition for 2x2x2 (Strassen) and combine them.
# But 3x4x7 is not a power of 2.
# We can use a block decomposition.
# For example, decompose 3x4x7 into smaller blocks.
# 3 = 2 + 1, 4 = 2 + 2, 7 = 4 + 3.
# This might be complex.

# Let's try a random search with a deadline.
# We will try to find a decomposition of rank R for R = 30, 40, 50, ... up to 84.
# For each R, we will generate random matrices U, V, W of shape (N, R), (M, R), (P, R) with integer or half-integer entries.
# Then we will check if the decomposition is exact.
# Since the search space is huge, this is unlikely to succeed quickly.

# A better approach is to use the fact that the tensor is sparse and has a specific structure.
# We can use a greedy algorithm or a local search to minimize the rank.

# Given the time constraint (150s), we will implement a randomized local search.
# We start with the schoolbook decomposition (rank 84) and try to reduce the rank by merging or eliminating terms.

# However, a more robust method is to use the "randomized rounding" or "integer programming" approach, but that is complex.

# Let's stick to a simple random search with a deadline.
# We will try to find a decomposition of rank R for R = 30, 40, 50, 60, 70, 80, 84.
# For each R, we will generate random matrices U, V, W with entries in {0, 0.5, 1, 1.5, 2} and check if the decomposition is exact.
# If we find one, we return it.
# If not, we return the schoolbook decomposition.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1

    # Try to find a lower rank decomposition
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Random search for lower rank
    for R in range(30, rank_sb + 1, 10):
        if time.time() > deadline:
            break
        for attempt in range(1000):
            if time.time() > deadline:
                break
            # Generate random matrices with entries in {0, 0.5, 1, 1.5, 2}
            U = np.random.choice([0, 0.5, 1, 1.5, 2], size=(n * m, R))
            V = np.random.choice([0, 0.5, 1, 1.5, 2], size=(m * p, R))
            W = np.random.choice([0, 0.5, 1, 1.5, 2], size=(p * n, R))
            
            # Check if the decomposition is exact
            # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We need T[i, j, k] = 1 for all i, j, k.
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Convert to lists of integers or half-integers
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}

    return {"U": best_U, "V": best_V, "W": best_W}
