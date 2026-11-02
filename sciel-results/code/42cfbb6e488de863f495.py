"""Program to find rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication."""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a bilinear decomposition for 4x5 * 5x5 matrix multiplication.
    
    Strategy: Use schoolbook multiplication but try to find a better partition.
    
    A is 4x5, B is 5x5, C = A @ B is 4x5.
    
    Schoolbook: C[i,k] = sum_j A[i,j] * B[j,k]
    This requires 4*5*5 = 100 scalar multiplications.
    
    Let's try a block decomposition:
    Split A into A_left (4x4) and A_right (4x1)
    Split B into B_top (4x5) and B_bottom (1x5)
    
    C = A_left @ B_top + A_right @ B_bottom
    
    A_left @ B_top: 4x4 * 4x5 -> 4x4 result
    A_right @ B_bottom: 4x1 * 1x5 -> 4x5 result
    
    For A_left @ B_top (4x4 * 4x5):
    Split B_top into B_top_left (4x4) and B_top_right (4x1)
    A_left @ B_top = A_left @ B_top_left + A_left @ B_top_right
    
    So:
    - A_left @ B_top_left: 4x4 * 4x4 -> 16 mults (schoolbook)
    - A_left @ B_top_right: 4x4 * 4x1 -> 16 mults (schoolbook)
    - A_right @ B_bottom: 4x1 * 1x5 -> 20 mults (schoolbook)
    
    Total: 16 + 16 + 20 = 52
    
    Wait, that's already better than 100! Let me verify.
    
    Actually, for 4x4 * 4x4, schoolbook is 16 multiplications.
    For 4x4 * 4x1, each of the 4 rows of A_left multiplies with the 4x1 vector B_top_right,
    giving 4 results, each requiring 4 mults, so 16 total.
    For 4x1 * 1x5, each of the 4 elements of A_right multiplies with the 1x5 row B_bottom,
    giving 4*5 = 20 mults.
    
    Total = 16 + 16 + 20 = 52.
    
    But wait, can we do better? Let's think about using Strassen for the 4x4*4x4 part.
    
    Actually, let me just implement the 52-rank decomposition first and verify it works.
    
    The indices:
    A[i,j] where i in [0,4), j in [0,5)
    B[j,k] where j in [0,5), k in [0,5)
    C[i,k] where i in [0,4), k in [0,5)
    
    Part 1: A[0:4, 0:4] @ B[0:4, 0:4] -> contributes to C[0:4, 0:4]
    For each i in [0,4), k in [0,4):
      C[i,k] += sum_{j=0}^{3} A[i,j] * B[j,k]
      That's 4*4 = 16 terms, each with 4 products... no wait.
      
    Actually, schoolbook for 4x4*4x4:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    For each (i,k) pair, we have 4 products.
    Total products = 4*4*4 = 64? No, that's not right either.
    
    Let me be more careful. In the bilinear decomposition:
    C[i,k] = sum_{j} A[i,j] * B[j,k]
    
    Each term A[i,j] * B[j,k] is one bilinear term.
    The number of such terms is n*m*p = 4*5*5 = 100.
    
    So schoolbook has rank 100.
    
    Now, if I split:
    C = A_left @ B_top + A_right @ B_bottom
    
    A_left is 4x4 (indices i in [0,4), j in [0,4))
    B_top is 4x5 (indices j in [0,4), k in [0,5))
    A_right is 4x1 (indices i in [0,4), j=4)
    B_bottom is 1x5 (indices j=4, k in [0,5))
    
    C = A_left @ B_top + A_right @ B_bottom
    
    For A_left @ B_top:
    C1[i,k] = sum_{j=0}^{3} A_left[i,j] * B_top[j,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    This has 4*4*5 = 80 terms? No, 4 (i) * 4 (j) * 5 (k) = 80 terms.
    
    For A_right @ B_bottom:
    C2[i,k] = A_right[i,0] * B_bottom[0,k] = A[i,4] * B[4,k]
    This has 4*1*5 = 20 terms.
    
    Total = 80 + 20 = 100. Same as schoolbook!
    
    So the block decomposition doesn't help because we're just splitting the sum over j.
    
    To get a better rank, we need to use a non-trivial algorithm like Strassen.
    
    Let me try Strassen for the 4x4*4x4 part (A_left @ B_top_left where B_top_left is 4x4).
    
    Actually, let me think about this differently. The key insight is that for matrix multiplication,
    we can sometimes reduce the number of multiplications using algebraic identities.
    
    For 2x2*2x2, Strassen uses 7 multiplications instead of 8.
    For 4x4*4x4, we could use Strassen recursively, but the overhead might not be worth it for small sizes.
    
    Let me try a different approach: use the fact that 4x5*5x5 can be viewed as a subset of 5x5*5x5.
    
    Actually, let me just implement the schoolbook decomposition with rank 100 and make sure it's valid.
    The task says the score is the rank, so if I can't find a better decomposition, I should at least
    return a valid one.
    
    Wait, the current program already returns rank 100. Let me see if I can do better.
    
    Let me try using Strassen for a 2x2*2x2 sub-problem.
    
    Actually, I realize I should just implement a valid decomposition and hope for the best.
    Let me use the schoolbook approach but with a smarter indexing.
    
    Actually, let me try a different strategy: use the fact that the rank of matrix multiplication
    tensor is at most n*m*p, and try to find a decomposition with fewer terms.
    
    For now, let me just return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition: rank = n*m*p = 100
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
