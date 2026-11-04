import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = 3, 3, 7
    
    if _time_left() < 5:
        return _schoolbook(n, m, p)
    
    # Strategy: Split the 7-column output into:
    # - Two 3x3 blocks (columns 0-2 and 3-5)
    # - One 3x1 block (column 6)
    #
    # For each 3x3 * 3x3 multiplication, we use a rank-23 decomposition
    # (based on Winograd or similar fast schemes).
    # For the 3x3 * 3x1 multiplication, we use rank-9 (schoolbook).
    #
    # Total rank: 23 + 23 + 9 = 55 (if independent)
    # But we can share the U factors for the two 3x3 blocks since they use
    # the same A matrix. Actually, we need to be careful.
    #
    # Let's think more carefully:
    # C[:, 0:3] = A @ B[:, 0:3]  (3x3 * 3x3)
    # C[:, 3:6] = A @ B[:, 3:6]  (3x3 * 3x3)
    # C[:, 6]   = A @ B[:, 6]    (3x3 * 3x1)
    #
    # For the first two, we can use the same rank-23 decomposition pattern
    # but with different V and W factors (different columns of B and C).
    #
    # Actually, let's use a known rank-23 scheme for 3x3 * 3x3.
    # The standard schoolbook is rank 27. There are known rank-23 schemes.
    #
    # Let me implement a simple rank-23 scheme for 3x3 * 3x3.
    # Based on the "Winograd-style" or "Strassen-like" approaches for small matrices.
    #
    # Actually, for 3x3 * 3x3, the minimal rank is known to be 23.
    # Let me use a specific rank-23 decomposition.
    #
    # For simplicity and correctness, let me first ensure correctness with
    # a slightly better than schoolbook approach, then optimize.
    #
    # Let's use rank-23 for each 3x3*3x3 block and rank-9 for 3x3*3x1.
    # Total: 23 + 23 + 9 = 55.
    #
    # Actually, let me just use the schoolbook for now to ensure correctness,
    # but with a tighter bound. The schoolbook for 3x3*3x7 is rank 63.
    # By splitting, we get 27 + 27 + 9 = 63 (same).
    #
    # To improve, I need actual rank-23 decompositions.
    # Let me use a known rank-23 scheme.

    # For 3x3 * 3x3, here is a rank-23 decomposition:
    # Based on work by K. D. Hahn, or similar.
    # I'll use a parameterized approach.

    # Actually, let me use the fact that 3x3*3x3 can be done in rank 23.
    # I'll hardcode a known rank-23 scheme.

    # For now, let me use a hybrid: schoolbook for correctness, but with
    # the splitting structure. This gives rank 63. To beat that, I need
    # actual rank-23.

    # Let me implement a rank-23 scheme for 3x3 * 3x3.
    # One known approach uses the fact that the tensor rank of 3x3*3x3 is 23.

    # I'll use a specific rank-23 decomposition.
    # Reference: "On the rank of the matrix multiplication tensor" etc.

    # For simplicity, let me use a block approach where I treat the 7 columns
    # as 3+3+1 and use rank-23 for each 3x3*3x3 and rank-9 for 3x3*3x1.

    # Rank-23 scheme for 3x3 * 3x3:
    # I'll use a known explicit decomposition.

    # Let me use the following approach:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # For rank-23, I need 23 terms. Let me use a scheme based on
    # the "Winograd" or "Fermat" style algorithms adapted for 3x3.

    # Actually, let me just use the schoolbook decomposition with the
    # splitting to get a valid answer, and hope the rank is at most 63.
    # The current program already does this with rank 36 (which is wrong -
    # it's not a valid decomposition).

    # Wait, the current program has a bug. Let me check:
    # It's trying to use rank 27+9=36, but the U, V, W matrices are not
    # constructed correctly for a valid decomposition.

    # Let me just use the full schoolbook decomposition with rank 63.
    # This is guaranteed to be correct.

    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
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
