import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if output[i,k] depends on A[i,j]*B[j,k]
    # T is indexed as [i*n+j, j*p+k, k*n+i] -> 1
    # Shape: (n*m, m*p, p*n)
    
    # Try to find a low-rank decomposition using integer/half-integer entries
    # Start with schoolbook rank = n*m*p = 144
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    # T[i*m+j, j*p+k, k*n+i] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_sb
    
    # Try Strassen-like approach for 4x4 x 4x9
    # Split into 2x2 blocks
    # A: 4x4 -> 2x2 blocks of 2x2
    # B: 4x9 -> 2x9 blocks (top 2 rows, bottom 2 rows)
    # C: 4x9 -> 2x9 blocks
    
    # Strassen for 2x2 * 2x9:
    # A = [[A11, A12], [A21, A22]] each 2x2
    # B = [[B1, B2], [B3, B4]] where B1,B3 are 2x9 (top/bottom rows of first 2 cols? No...)
    
    # Actually, B is 4x9, so split as:
    # B = [[B_top, B_bottom]] where B_top is 2x9, B_bottom is 2x9
    # But for Strassen, we need to split both dimensions.
    # Let me split B into 2x2 blocks of size 2x(9/2)... but 9 is odd.
    
    # Alternative: use 2x2 Strassen on the 2x2 dimension, treating the 9 as a parameter.
    # For each of the 9 columns, do 2x2 * 2x1 Strassen.
    # That's 7 multiplications per column * 9 columns = 63 rank-1 terms for the 2x2*2x9 part.
    # But we have 4 such 2x2*2x9 blocks (from the 4x4 * 4x9 structure).
    
    # Actually, let me think differently. The 4x4 * 4x9 multiplication can be seen as:
    # For each of the 4 output rows, compute a 1x4 * 4x9 = 1x9 product.
    # Each 1x4 * 4x9 is just a dot product times each column: rank 4.
    # Total: 4*4 = 16? No, that's not right either.
    
    # Let me try a direct algebraic approach with small rank.
    # The tensor T has shape (16, 36, 36) with T[i,j,k] = 1 iff i,j,k are compatible.
    
    # For a rank-r decomposition: T[i,j,k] = sum_{c=1}^r U[i,c]*V[j,c]*W[k,c]
    # We need U: 16xr, V: 36xr, W: 36xr
    
    # Let me try to find a decomposition with rank < 144 using a greedy or algebraic approach.
    
    # Try rank = 108 (3/4 of schoolbook)
    # Try rank = 96
    # Try rank = 72
    
    # For now, let's try a simple improvement: merge some rank-1 terms.
    # If two terms have the same U pattern, we can combine them.
    
    # Check if we can do better with a structured approach.
    # Use the fact that W[k*n+i] couples k and i.
    
    # Let me try a different block structure:
    # Split 4x4 into 2x2 blocks, and 4x9 into 2x9 blocks.
    # C[0:2, 0:9] = A[0:2,0:2]*B[0:2,0:9] + A[0:2,2:4]*B[2:4,0:9]
    # C[2:4, 0:9] = A[2:4,0:2]*B[0:2,0:9] + A[2:4,2:4]*B[2:4,0:9]
    
    # Each 2x2 * 2x9 multiplication:
    # Using Strassen: 7 multiplications of 2x2 * 2x9 "scalars" (which are 2x9 vectors)
    # But 2x2 * 2x9 is not standard Strassen. Let me use schoolbook for 2x2*2x9: rank = 2*2*9 = 36.
    # 4 such blocks: 4*36 = 144. Same as before.
    
    # Try: can we share computations between the 4 blocks?
    # The B blocks B[0:2,0:9] and B[2:4,0:9] are shared between top and bottom output blocks.
    # The A blocks A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4] are all different.
    
    # Let me try a different approach: use rank-1 terms where U, V, W have sparse patterns.
    
    # Time check
    if time.time() > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to find a lower-rank decomposition via random search with integer/half-integer coefficients
    # This is a heuristic approach
    
    # For now, return the schoolbook decomposition
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
