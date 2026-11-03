"""Program for matrix multiplication tensor decomposition task."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Structural Change: Use exact algebraic construction (Schoolbook) instead of random search.
    The 3x3x6 multiplication is the direct sum of two 3x3x3 multiplications.
    We use the standard schoolbook decomposition for 3x3x3 (Rank 27).
    Total Rank = 27 + 27 = 54.
    This guarantees an exact, integer-based solution that runs instantly.
    """
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Schoolbook decomposition for 3x3x3
    # T_{3i+j, 3j+k, 3k+i} = 1
    # Rank 27: One term for each (i, j, k)
    rank_3x3 = 27
    U3 = [[0] * rank_3x3 for _ in range(9)]
    V3 = [[0] * rank_3x3 for _ in range(9)]
    W3 = [[0] * rank_3x3 for _ in range(9)]
    
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_idx = i * 3 + j
                v_idx = j * 3 + k
                w_idx = k * 3 + i
                U3[u_idx][r] = 1
                V3[v_idx][r] = 1
                W3[w_idx][r] = 1
                r += 1
                
    # Construct full 3x3x6 decomposition
    total_rank = 2 * rank_3x3
    U_full = [[0] * total_rank for _ in range(9)]
    V_full = [[0] * total_rank for _ in range(18)]
    W_full = [[0] * total_rank for _ in range(18)]
    
    # Block 1: k in 0,1,2
    for r in range(rank_3x3):
        # U is same
        for i in range(3):
            for j in range(3):
                U_full[i*3+j][r] = U3[i*3+j][r]
                
        # V: global index v = 6j + k. U3/V3 index is 3j+k.
        for j in range(3):
            for k in range(3):
                V_full[6*j+k][r] = V3[3*j+k][r]
                
        # W: global index w = 3k + i. U3/W3 index is 3k+i.
        for k in range(3):
            for i in range(3):
                W_full[3*k+i][r] = W3[3*k+i][r]
                
    # Block 2: k in 3,4,5
    # Let k = 3 + k_prime
    for r in range(rank_3x3):
        # U is same
        for i in range(3):
            for j in range(3):
                U_full[i*3+j][r + rank_3x3] = U3[i*3+j][r]
                
        # V: global index v = 6j + (3+k_prime) = 6j + 3 + k_prime.
        # V3 index is 3j + k_prime.
        for j in range(3):
            for k_prime in range(3):
                V_full[6*j + 3 + k_prime][r + rank_3x3] = V3[3*j + k_prime][r]
                
        # W: global index w = 3*(3+k_prime) + i = 9 + 3k_prime + i.
        # W3 index is 3k_prime + i.
        for k_prime in range(3):
            for i in range(3):
                W_full[9 + 3*k_prime + i][r + rank_3x3] = W3[3*k_prime + i][r]
                
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
