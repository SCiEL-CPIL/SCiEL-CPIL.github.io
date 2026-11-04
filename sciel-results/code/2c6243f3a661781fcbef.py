"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Strategy:
    # 1. Attempt Rank 161: Strassen (2x2 blocks) + Laderman (3x3 blocks)
    #    6x6 is viewed as 2x2 blocks of 3x3 matrices.
    #    Block mult: Strassen (7 products).
    #    Inner mult: Laderman (23 products).
    #    Total rank: 7 * 23 = 161.
    # 2. Fallback Rank 189: Strassen (2x2 blocks) + Schoolbook (3x3 blocks)
    #    Total rank: 7 * 27 = 189.
    
    # Laderman's 23 terms for 3x3 matrix multiplication.
    # Each term is (U, V, W) where U, V, W are 3x3 matrices (flattened to 9 elements).
    # The decomposition satisfies: sum_t U_t[i,j] V_t[j,k] W_t[k,i] = 1 for all i,j,k in 0..2.
    # Note: The tensor for matrix mult M_{3,3,3} has T[i,j,k] = 1.
    # W index convention: W[k*n+i, t]. So W_t[k,i] is the coefficient for output (i,k) from product t.
    # Actually, for the 3x3 inner product, the output is a 3x3 matrix C.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The term contributes U[i,j] * V[j,k] * W[k,i] to the tensor T[i,j,k].
    # We need sum_t U_t[i,j] V_t[j,k] W_t[k,i] = 1.
    
    # I will use a known correct set of Laderman terms.
    # If I cannot guarantee correctness, I will use the fallback.
    # For this implementation, I will use the fallback (Rank 189) to ensure validity,
    # as hardcoding 23 complex terms is prone to error without a verified source in context.
    # However, to attempt improvement, I will structure the code to easily swap in Laderman.
    
    # Let's try to implement Rank 189 cleanly first, as it is the current best.
    # Then, if time permits, I might try to inject Laderman.
    # Given the constraint "Keep the program's entry-point function name(s)...", 
    # and the risk of invalid Laderman terms, I will stick to the verified Rank 189 solution 
    # but ensure it is as optimized as possible.
    
    block_size = 2
    blocks_per_dim = 3
    
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 matrices
    # Each pattern is a tuple: (u_coeffs, v_coeffs, w_coeffs)
    # coeffs are dicts mapping (row, col) to value
    strassen_patterns = [
        # M0 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M3 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M4 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # Update U
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col_idx] += val
                    
                    # Update V
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    
                    # Update W
                    for (pi, pj), val in w_coeffs.items():
                        # W index is k*n + i
                        # k is col of C (which is K block, pj local)
                        # i is row of C (which is I block, pi local)
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
