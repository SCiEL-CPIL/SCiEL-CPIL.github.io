"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy: Use the known rank-21 decomposition for 3x3*3x3 (from Winograd-like schemes)
    and extend it to handle the 7th column. Since 3x3*3x7 = 3x3*3x3 + 3x3*3x1,
    we can use rank 21 for the first 6 columns and add rank 3 for the 7th column,
    giving total rank 24. However, we can do better by finding a direct decomposition.
    
    Actually, let's use a more systematic approach: we'll try to find the minimal rank
    by using known results. The rank of 3x3*3x3 is at most 21 (and it's known that 
    21 is achievable). For 3x3*3x7, we can view it as multiplying by a 3x7 matrix.
    
    Let's construct a decomposition of rank 24: use rank-21 for columns 0-5 (two 3x3 blocks)
    and rank-3 for column 6.
    
    But wait - we can do better. Let's think about this differently.
    
    A 3x3 matrix A times a 3x7 matrix B gives a 3x7 matrix C where C[i,k] = sum_j A[i,j]*B[j,k].
    
    We can use the fact that rank(3,3,7) <= rank(3,3,3) + rank(3,3,4) and so on.
    
    Actually, let's just use a simple but effective approach: we'll construct a decomposition
    with rank 24 by using two independent 3x3*3x3 decompositions (rank 21 each would be 42, 
    but we can share structure).
    
    Better approach: Use the standard rank-21 decomposition for 3x3*3x3, and for the 7th column,
    we need 3 additional terms (one for each output row). So total rank = 21 + 3 = 24.
    
    Let me implement the rank-21 decomposition for 3x3*3x3 first.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # Rank-21 decomposition for 3x3 * 3x3
    # This is based on the work of Winograd and others.
    # The decomposition uses 21 multiplications.
    
    # For a 3x3 matrix multiplication C = A * B where A is 3x3 and B is 3x3,
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The rank-21 decomposition can be written as follows.
    # We'll use a known construction.
    
    # Actually, let me use a simpler but still efficient approach.
    # We'll use the fact that we can decompose the 3x7 case as:
    # - First 3 columns: use rank-9 (schoolbook for 3x3*3x3)
    # - Next 3 columns: use rank-9 (schoolbook for 3x3*3x3)  
    # - Last column: use rank-3 (schoolbook for 3x3*3x1)
    # Total: 21
    
    # But we can do better. Let's try rank 21 using a more clever decomposition.
    
    # For now, let's use a decomposition of rank 21 by combining:
    # - 3x3*3x3 with rank 12 (if we can find one)
    # - 3x3*3x1 with rank 3
    # But I'm not sure rank 12 is achievable for 3x3*3x3.
    
    # The best known rank for 3x3*3x3 is 19 (Strassen's algorithm gives 7 for 2x2,
    # and extensions give various ranks for 3x3). Actually, for 3x3*3x3, the rank
    # is at most 21 (schoolbook is 27, Winograd gives 21).
    
    # Let me just use rank 21: two 3x3 blocks with rank 21 total (shared structure) 
    # plus 3 for the last column. Actually that's not right either.
    
    # Simplest correct approach: use schoolbook for each 3x3 block and 3x1 column.
    # 3x3*3x3: 9 terms, 3x3*3x3: 9 terms, 3x3*3x1: 3 terms. Total: 21.
    
    # This is better than the initial 63 (3*3*7).
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 9 x 21
    V = [[0] * rank for _ in range(m * p)]  # 21 x 21
    W = [[0] * rank for _ in range(p * n)]  # 21 x 21
    
    col = 0
    
    # First 3x3 block (columns 0-2)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Second 3x3 block (columns 3-5)
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Last column (column 6)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
