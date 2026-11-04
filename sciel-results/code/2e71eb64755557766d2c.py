"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

# Known rank-19 decomposition for 3x3x3 matrix multiplication (Landsberg 1998, simplified form)
# We need to provide the 19 rank-1 tensors for 3x3x3.
# Since I don't have the exact coefficients memorized and risk producing an invalid decomposition,
# I will use the schoolbook decomposition but attempt a simple improvement via block structure.

# Actually, let me think about whether we can use known lower-rank decompositions.
# M(3,3,3) <= 19 is known but I need the exact coefficients.
# Let me try a different approach: use the fact that we can split the 3x4x6 problem.

# For 3x4x6, the schoolbook rank is 3*4*6 = 72.
# Can we do better?
# One approach: split the 4 dimension. 3x4x6 = (3x2x6) + (3x2x6) with rank contributions.
# But that's still 3*2*6 + 3*2*6 = 36 + 36 = 72. No gain.

# Another: use the fact that M(2,2,2) = 7 (Strassen).
# 3x4x6: we can think of this as multiplying 3x4 by 4x6.
# Split the 4x4 middle dimension using Strassen's 7 multiplications for 2x2x2 blocks.
# But 4 is not a power of 2 in a way that directly helps for 3x4x6.

# Let me try: 3x4x6. We can group columns of A (3x4) into two 3x2 blocks.
# A = [A1, A2] where A1, A2 are 3x2.
# B (4x6) can be split into two 2x6 blocks: B = [B1; B2] where B1, B2 are 2x6.
# C = A1*B1 + A2*B2.
# Each A_i * B_i is 3x2 * 2x6 = 3x6, which has schoolbook rank 3*2*6 = 36.
# Total: 36 + 36 = 72. No improvement.

# What if we use Strassen on the 2x2 part?
# A1 is 3x2, B1 is 2x6. We can view this as: for each of the 3 rows of A1 and 6 columns of B1,
# we compute a 2x2 dot product... no, that's not right.

# Let me reconsider. The tensor for 3x4x6 has entries T[i,j,k] for i in [0,3), j in [0,4), k in [0,6).
# Schoolbook: rank 72.

# I'll stick with the schoolbook decomposition as it's guaranteed correct.
# The deadline guard is in place.

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() - start_time > deadline - 5.0:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
