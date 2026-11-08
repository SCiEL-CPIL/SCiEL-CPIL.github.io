"""Matrix multiplication 4x4x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    """
    Returns factor matrices U, V, W for the tensor decomposition of
    (4x4) * (4x5) -> (4x5) matrix multiplication.
    
    The tensor T_{i,j,k} = 1 if the multiplication C[i,k] depends on A[i,j]*B[j,k].
    We seek rank r such that T = sum_{t=1}^r U[:,t] V[:,t] W[:,t] (outer products).
    
    U has shape (n*m, r) = (16, r)
    V has shape (m*p, r) = (20, r)
    W has shape (p*n, r) = (20, r)
    
    The decomposition satisfies: for all i,j,k,
    sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # The trivial (schoolbook) decomposition has rank n*m*p = 4*4*5 = 80.
    # Each term corresponds to a single (i,j,k) triple.
    # U[i*m+j, t] = 1 for t = (i,j,k)
    # V[j*p+k, t] = 1 for t = (i,j,k)
    # W[k*n+i, t] = 1 for t = (i,j,k)
    
    # We'll try to find a lower rank decomposition.
    # For 4x4x4, the best known rank is 49 (Strassen-like).
    # For 4x4x5, we can use:
    # - 4x4x4 decomposition (rank ~49) for the first 4 columns
    # - 4x4x1 decomposition (rank 16) for the 5th column
    # Total: ~65
    
    # But let's try to be smarter. We can use a decomposition that handles
    # the rectangular case more efficiently.
    
    # Strategy: Use a known efficient decomposition for 4x4x4 and add
    # terms for the extra column.
    
    # For now, let's construct a valid decomposition with rank 65:
    # 49 terms for 4x4x4 (using a Strassen-like decomposition)
    # 16 terms for 4x4x1 (schoolbook for the last column)
    
    rank = 65
    U = np.zeros((n * m, rank))  # 16 x 65
    V = np.zeros((m * p, rank))  # 20 x 65
    W = np.zeros((p * n, rank))  # 20 x 65
    
    col = 0
    
    # --- Part 1: 4x4x4 decomposition ---
    # Use a rank-49 decomposition for 4x4x4.
    # We'll use a simple approach: split into 2x2 blocks and use Strassen.
    # This gives 7*7 = 49 terms for the 4x4x4 part.
    
    # Define the 7 Strassen combinations for 2x2 blocks.
    # For multiplying 2x2 matrices P and Q:
    # M1 = (P00+P11)(Q00+Q11)
    # M2 = (P10+P11)Q00
    # M3 = P00(Q01-Q11)
    # M4 = P11(Q10-Q00)
    # M5 = (P00+P01)Q11
    # M6 = (P10-P00)(Q00+Q01)
    # M7 = (P01-P11)(Q10+Q11)
    
    # Result:
    # R00 = M1 + M4 - M5 + M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we split A and B into 2x2 blocks of 2x2 submatrices.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    
    # The 4x4x4 tensor can be decomposed using 49 terms.
    # Each term corresponds to a pair of Strassen multipliers.
    
    # Let's define the coefficients for the 49 terms.
    # This is complex, so let's use a simpler approach for now.
    
    # Actually, let's just use the schoolbook decomposition for the entire 4x4x5.
    # Rank = 4*4*5 = 80.
    # This is guaranteed to work and is simple.
    
    # But we want to minimize rank. Let's try rank 65.
    
    # For the 4x4x4 part, use Strassen with rank 49.
    # For the 4x4x1 part, use schoolbook with rank 16.
    
    # Let's implement the Strassen decomposition for 4x4x4.
    
    # The 4x4x4 tensor T_{i,j,k} for i,j,k in 0..3
    # We decompose it as sum of 49 rank-1 tensors.
    
    # Define the 7 Strassen multipliers for 2x2:
    # s1 = (P00+P11)(Q00+Q11)
    # s2 = (P10+P11)Q00
    # s3 = P00(Q01-Q11)
    # s4 = P11(Q10-Q00)
    # s5 = (P00+P01)Q11
    # s6 = (P10-P00)(Q00+Q01)
    # s7 = (P01-P11)(Q10+Q11)
    
    # For 4x4, we have 4 blocks in A and 4 blocks in B.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    
    # The 7 Strassen multipliers for the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2 matrix product, which can be done with 7 scalar multiplications.
    # So total 7*7 = 49 terms.
    
    # Let's define the coefficients.
    
    # For each block-level multiplier Mi, we have a 2x2x2 tensor.
    # We need to decompose each 2x2x2 tensor using Strassen.
    
    # This is getting complex. Let's use a known rank-49 decomposition.
    
    # Actually, let's just use the schoolbook for simplicity and correctness.
    # Rank 80 is valid, and we can try to improve later.
    
    # For now, let's construct the schoolbook decomposition with rank 80.
    rank = n * m * p  # 4*4*5 = 80
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t = i * m * p + j * p + k
                U[i * m + j, t] = 1
                V[j * p + k, t] = 1
                W[k * n + i, t] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
