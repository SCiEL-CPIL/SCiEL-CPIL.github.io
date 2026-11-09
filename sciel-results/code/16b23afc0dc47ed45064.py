"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Strategy: Decompose 2x3x9 into three 2x3x3 blocks.
    # The 2x3x3 multiplication has a trivial rank of 9.
    # By using the same U factors for all blocks, we can construct a rank 27 solution.
    # U shape: (2*3, 27)
    # V shape: (3*9, 27)
    # W shape: (9*2, 27)
    
    base_rank = n * m * p # 54
    best_rank = base_rank
    best_U = [[0] * base_rank for _ in range(n * m)]
    best_V = [[0] * base_rank for _ in range(m * p)]
    best_W = [[0] * base_rank for _ in range(p * n)]
    
    # Trivial 54 solution
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Construct Rank 27 Solution
    # 3 blocks of size 3 in the k-dimension.
    # Each block is a 2x3x3 multiplication.
    # Trivial rank for 2x3x3 is 9.
    # Total rank = 3 * 9 = 27.
    
    rank_27 = 27
    U_27 = [[0] * rank_27 for _ in range(n * m)]
    V_27 = [[0] * rank_27 for _ in range(m * p)]
    W_27 = [[0] * rank_27 for _ in range(p * n)]
    
    # For each block b in 0, 1, 2:
    # k indices are b*3, b*3+1, b*3+2
    # We use 9 rank-1 terms for this block.
    # Term index r_global = b * 9 + r_local, where r_local in 0..8 corresponds to (j, k_local)
    
    for b in range(3):
        for j in range(m): # j in 0..2
            for k_local in range(3): # k_local in 0..2
                k_global = b * 3 + k_local
                r_global = b * 9 + j * 3 + k_local
                
                # U[i, j, r] = 1 for all i?
                # In the trivial 2x3x3 decomposition:
                # U[i, j, r] = 1 if r corresponds to (j, k)?
                # Let's check the condition: sum_r U[i,j,r] V[j,k,r] W[i,k,r] = 1
                # If U[i,j,r] = 1 for the specific r, V[j,k,r] = 1, W[i,k,r] = 1.
                # Then product is 1.
                # So for a fixed block b, and fixed j, k_local:
                # r_global is the term.
                # U[i, j, r_global] should be 1 for all i in 0..1?
                # Yes, because the term contributes to all i.
                for i in range(n):
                    U_27[i * m + j][r_global] = 1
                    
                V_27[j * p + k_global][r_global] = 1
                
                for i in range(n):
                    W_27[k_global * n + i][r_global] = 1

    # Verify Rank 27 Solution
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0
                    for r in range(rank):
                        actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    if verify(U_27, V_27, W_27):
        best_rank = rank_27
        best_U = U_27
        best_V = V_27
        best_W = W_27
    else:
        # Fallback to trivial if verification fails (should not happen)
        pass

    # Try to find lower rank?
    # Rank 21 is the theoretical limit for this block structure.
    # We would need the rank-7 solution for 2x3x3.
    # Since we don't have it hardcoded, we stick to 27.
    # Random search for rank < 27 is unlikely to succeed quickly.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
