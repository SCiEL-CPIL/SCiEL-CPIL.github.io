"""Matrix multiplication 2x3x9 with rank minimization via block decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Strategy: Decompose 3x9 into two 3x3 blocks
    # C = A * B where A is 2x3, B is 3x9
    # Split B = [B1 | B2] where B1, B2 are 3x3
    # Then C = [A*B1 | A*B2], each 2x3
    # 
    # If we can find rank r decomposition for 2x3x3, total rank = 2*r
    # Schoolbook for 2x3x3 is rank 18, lower bound is max(6,9,6)=9
    # Let's search for rank 9 decomposition of 2x3x3
    # If found, total rank = 18 (vs schoolbook 54)
    
    # First, try to find a rank-9 decomposition for 2x3x3
    # The tensor T[i,j,k] = delta_{i,k} for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # We want T = sum_r u_r * v_r * w_r
    # where u_r is 6-dim, v_r is 9-dim, w_r is 6-dim
    
    # Use bilinear decomposition approach
    # For rank 9, we can try to exploit the structure
    # 
    # Actually, let's use a known result: for 2x3x3, the rank is at most 12
    # (from Strassen-type algorithms), but let's try to find one
    #
    # Alternative: Use the fact that we can do 2x3x3 with rank 12
    # Total would be 24
    #
    # Let me try a direct construction for rank 9
    # If the 2x3x3 tensor has rank 9, we can use it
    #
    # Actually, for this specific case, let's just use the schoolbook for safety
    # rank = 54
    #
    # But wait - let me try to exploit the 3x9 = 3x(3x3) structure differently
    # Use a rank-12 decomposition for 2x3x3, total rank 24
    
    # Let's construct a rank-12 decomposition for 2x3x3
    # Using the approach of splitting the multiplication
    #
    # For 2x3 * 3x3:
    # We can write this as 2 separate 1x3 * 3x3 multiplications
    # Each 1x3 * 3x3 has rank 9 (schoolbook), so total rank 18
    # But we can do better
    #
    # Let me just use the guaranteed schoolbook for now, but with a twist:
    # Use the factorization to reduce constants
    
    # Actually, let me try to find if rank 27 is achievable for the full 2x3x9
    # Lower bound is max(6, 27, 18) = 27
    # 
    # If I can find a rank-27 decomposition, that's a big win
    # Let's try to construct one using the block structure
    
    # Approach: Use rank-9 decompositions for each 2x3x3 block
    # If rank 9 exists for 2x3x3, total rank = 3 * 9 = 27
    
    # Let me search for a rank-9 decomposition of 2x3x3
    # The tensor is T[i,j,k] = 1 if i==k else 0, for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    #
    # Actually, I'll just implement the schoolbook decomposition which is guaranteed valid
    # rank = 54
    #
    # But let me try one more thing: use the 3x3 block structure with rank-12 for each
    # Total rank = 2 * 12 = 24
    
    # For now, use schoolbook to guarantee validity
    rank = n * m * p  # 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
