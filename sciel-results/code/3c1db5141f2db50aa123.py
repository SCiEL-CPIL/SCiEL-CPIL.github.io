import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 65
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen for 4x4x4
    # We will use 49 terms.
    # The Strassen algorithm for 2x2x2 has 7 terms.
    # For 4x4x4, we use 7*7 = 49 terms.
    # We can generate them by taking the Kronecker product of the 7 terms.
    
    # Let's define the 7 Strassen terms for 2x2x2.
    # Each term is a 2x2x2 tensor.
    # We can represent them as 4x4 matrices (flattened).
    
    # S1: (A11+A22)(B11+B22)
    # A11 is A[0,0], A22 is A[1,1]
    # B11 is B[0,0], B22 is B[1,1]
    # The term is (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # This contributes to C[0,0] and C[1,1].
    
    # This is getting complicated to code from scratch.
    # Let's use a known rank-49 decomposition.
    # I will use the fact that the rank is 49 and just use a placeholder for now.
    # Actually, I can use the schoolbook for the 4x4x4 part too, which is 64 terms.
    # Then the total rank is 64 + 16 = 80, which is the same as the original.
    # So I must use Strassen.
    
    # Let's try to find a rank-56 decomposition.
    # M_{4,4,5} can be embedded in M_{4,4,6}.
    # M_{4,4,6} has rank 4 * 4 * 6 = 96 (schoolbook).
    # Strassen for M_{4,4,6} might be 49 + 7 = 56?
    # If we use Strassen for the first 4 columns (49) and Strassen for the next 2 columns (7)?
    # The next 2 columns are 4x2.
    # M_{4,4,2} can be computed with rank 4 * 2 = 8 (schoolbook) or maybe less.
    # If we use Strassen for M_{2,2,2} (rank 7) and M_{2,2,1} (rank 4), the total is 11.
    # So 49 + 11 = 60.
    # This is better than 65.
    
    # Let's implement the rank-60 decomposition.
    # 49 for the first 4 columns.
    # 11 for the next 2 columns.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
