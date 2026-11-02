"""Matrix multiplication 5x5x5 tensor decomposition."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # We'll use a block decomposition approach.
    # Split 5 = 3 + 2.
    # The 5x5 matrix multiplication can be viewed as blocks:
    # A = [[A11(3x3), A12(3x2)], [A21(2x3), A22(2x2)]]
    # B = [[B11(3x3), B12(3x2)], [B21(2x3), B22(2x2)]]
    # C = [[C11, C12], [C21, C22]] where:
    # C11 = A11*B11 + A12*B21  (3x3)
    # C12 = A11*B12 + A12*B22  (3x2)
    # C21 = A21*B11 + A22*B21  (2x3)
    # C22 = A21*B12 + A22*B22  (2x2)
    
    # For each block multiplication, we can use known minimal ranks:
    # 3x3x3: rank 23 (Beynon)
    # 3x2x2: rank 12 (schoolbook, since 3*2*2=12, and no known better)
    # 2x3x2: rank 12
    # 2x2x2: rank 7 (Strassen)
    # 2x2x3: rank 12
    # 3x3x2: rank 18 (schoolbook)
    # etc.
    
    # Actually, let's think more carefully. The block structure gives us:
    # C11 = A11*B11 + A12*B21: this is a sum of two matrix products.
    # To compute this, we need to compute A11*B11 (3x3*3x3) and A12*B21 (3x2*2x3).
    # These are independent computations that we can do in parallel (tensor sum).
    # Rank for 3x3x3 is 23, for 3x2x3 (i.e., 3x2 matrix times 2x3 matrix = 3x3 matrix) is 9 (schoolbook).
    # Wait, A12 is 3x2 and B21 is 2x3, so A12*B21 is a 3x3 matrix.
    # The rank of 3x2x3 tensor: this is like multiplying a 3x2 matrix by a 2x3 matrix.
    # Schoolbook: 3*2*3 = 18. But actually, this is a bilinear map from (3x2) x (2x3) -> 3x3.
    # The rank of this tensor is at most 9? No. Let me think again.
    # Actually, the tensor for (n x m) * (m x p) -> (n x p) has rank at most n*m*p.
    # For 3x2*2x3 -> 3x3: n=3, m=2, p=3, so schoolbook rank is 18.
    # But wait, the output is 3x3, and we're summing two such products.
    
    # This is getting complicated. Let me use a different approach.
    
    # Alternative: Just use the schoolbook decomposition but try to find any structure.
    # Actually, for 5x5x5, the schoolbook rank is 125.
    # The best known upper bound for 5x5x5 is 132 (from Bini et al. or similar), 
    # but actually Strassen's algorithm gives rank 125 for 5x5x5 (since 5 is odd, we can't directly use Strassen).
    # 
    # Wait, actually for 5x5x5:
    # We can pad to 6x6x6 and use Strassen. Strassen for 6x6x6: 
    # 6 = 2+2+2, use 3x3 Strassen blocks? No.
    # Actually, the standard approach: use Strassen for 4x4x4 (rank 49? no, Strassen is for 2x2x2 rank 7).
    # 
    # Let me just return the schoolbook decomposition. It's correct and simple.
    
    rank = n * m * p  # 125
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
