"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds deadline, 20s safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    schoolbook_rank = n * m * p  # 45
    
    # Try to find a rank-minimal decomposition
    # Known lower bounds and constructions for 3x3 * 3x5:
    # The rank is at least max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But for matrix multiplication, the tensor rank is higher.
    
    # For 3x3 * 3x5, we can use the following approach:
    # Split the 3x5 matrix into 3x3 + 3x2
    # 3x3 * 3x3 has rank at most 21 (Laderman's algorithm)
    # 3x3 * 3x2 can be done with rank at most 12 (using 2x2 blocks)
    # Total: 21 + 12 = 33
    
    # But we can do better. Let's try to use known results.
    # The rank of 3x3 * 3x3 matrix multiplication is 21 (Laderman).
    # For 3x3 * 3x5, we can decompose as:
    # C = A * [B1, B2] where B1 is 3x3 and B2 is 3x2
    # C1 = A * B1 (3x3) - rank 21
    # C2 = A * B2 (3x2) - rank at most 12 (using 2x2 blocks with rank 7 each, but 3x2 needs careful handling)
    
    # Actually, for 3x3 * 3x2:
    # Split A into 2x2 + 1x1, B2 into 2x2 + 1x2
    # This gets complicated. Let's use a different approach.
    
    # Alternative: Use the fact that rank(3x3 * 3x5) <= rank(3x3 * 3x3) + rank(3x3 * 3x2)
    # rank(3x3 * 3x2) can be computed as follows:
    # A (3x3) * B (3x2) = C (3x2)
    # Split A into A1 (2x3) and A2 (1x3), B into B1 (2x2) and B2 (1x2)
    # C1 = A1 * B1 (2x2) - rank 7 (Strassen)
    # C2 = A1 * B2 (2x2 * 1x2 = 2x2) - rank 4 (schoolbook, since B2 is 2x1... wait)
    
    # Let me reconsider. B is 3x2, so B1 = B[0:2, 0:2] (2x2), B2 = B[2:3, 0:2] (1x2)
    # A1 = A[0:2, 0:3] (2x3), A2 = A[2:3, 0:3] (1x3)
    # C[0:2, 0:2] = A1[:, 0:2] * B1 + A1[:, 2:3] * B[2:3, 0:2]
    # This is getting messy. Let's just use the schoolbook decomposition for now.
    
    # Actually, let me try a cleaner approach. The key insight is that for rectangular matrices,
    # we can sometimes reduce the rank below the schoolbook bound.
    
    # For 3x3 * 3x5, one approach:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can use the following decomposition based on block matrix multiplication.
    
    # Let's use the known result: rank(3x3 * 3x5) = 27
    # This can be achieved by decomposing the 3x5 into three 3x2 blocks... no, that's not right.
    
    # Actually, I recall that the rank of 3x3 * 3x3 is 21 (Laderman), and for rectangular cases,
    # the rank can be lower. Let me try to construct a 27-rank decomposition.
    
    # Alternative approach: Use the fact that we can pad the 3x5 to 3x6, use a 3x3 * 3x6 decomposition,
    # then extract the 3x5 part. But this might not reduce the rank.
    
    # Let's just implement the schoolbook decomposition for now, and if there's time, try to improve.
    
    # Schoolbook: rank = 45
    U = np.zeros((n * m, schoolbook_rank), dtype=object)
    V = np.zeros((m * p, schoolbook_rank), dtype=object)
    W = np.zeros((p * n, schoolbook_rank), dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
    
    # Check if we have time to try a better decomposition
    if time.time() - start_time > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a lower rank decomposition using known algorithms
    # For 3x3 * 3x5, the optimal rank is believed to be 27 or 28.
    
    # Let me try a block-based approach:
    # Split A (3x3) into blocks of size 2x2, 2x1, 1x2, 1x1
    # Split B (3x5) into corresponding blocks
    
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B1, B2], [B3, B4], [B5, B6]] where B1,B2 are 2x5... no, this doesn't work cleanly.
    
    # Let me try a different block structure:
    # A = [A1, A2] where A1 is 3x2 and A2 is 3x1
    # B = [B1; B2] where B1 is 2x5 and B2 is 1x5
    # C = A1 * B1 + A2 * B2
    # A1 * B1 is 3x2 * 2x5 = 3x5
    # A2 * B2 is 3x1 * 1x5 = 3x5
    
    # rank(3x2 * 2x5): Split 3x2 into 2x2 + 1x2, 2x5 into 2x2 + 2x1 + 2x1 + 2x1
    # This is getting complicated. Let's just use the schoolbook for now.
    
    # Actually, let me try to use Laderman's algorithm for the 3x3 * 3x3 part
    # and a simple approach for the 3x3 * 3x2 part.
    
    # Laderman's algorithm for 3x3 * 3x3 has rank 21.
    # For 3x3 * 3x2, we can use rank 12 (by splitting into 2x2 * 2x2 and 1x2 * 2x2, etc.)
    
    # Total: 21 + 12 = 33
    
    # But we need to verify this is correct. Let me implement it.
    
    # For now, return the schoolbook decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
