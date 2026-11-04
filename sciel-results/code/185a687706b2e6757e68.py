import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # We use a block-wise Strassen decomposition.
    # Split A (2x4) into A_left (2x2) and A_right (2x2).
    # Split B (4x8) into 4 blocks of (2x2): B_0, B_1, B_2, B_3.
    # Split C (8x2) into 4 blocks of (2x2): C_0, C_1, C_2, C_3.
    
    # The product C = A * B can be written as:
    # C = [A_left, A_right] * [B_0; B_1; B_2; B_3]
    #    = A_left * [B_0; B_1] + A_right * [B_2; B_3]
    # Let B_top = [B_0; B_1] (4x4) and B_bot = [B_2; B_3] (4x4).
    # Then C = A_left * B_top + A_right * B_bot.
    # This is a sum of two 2x4x4 matrix multiplications.
    # We can decompose each 2x4x4 multiplication using Strassen-like methods.
    # Specifically, a 2x4x4 multiplication can be seen as two 2x2x2 multiplications? No.
    # 2x4x4: A is 2x4, B is 4x4, C is 4x2.
    # We can split A into two 2x2 blocks? No, A is 2x4, so it's one block of 2x4.
    # We can split B into two 4x2 blocks? B is 4x4.
    # Let's stick to the previous decomposition which was rank 56.
    # The previous decomposition split the 4x8 B into 4 blocks of 2x2.
    # And treated A_left (2x2) * B_t (2x2) and A_right (2x2) * B_t (2x2).
    # This effectively computed 4 pairs of 2x2x2 multiplications.
    # Each pair of 2x2x2 multiplications can be done with Strassen (rank 7).
    # So 4 * 7 = 28? No, the previous code had rank 56.
    # Let's re-examine the previous code.
    # It had `for t in range(4)`.
    # Inside, it did `A_left * B_top` (using 7 mults) and `A_right * B_bot` (using 7 mults).
    # Wait, B_top and B_bot were 4x2 blocks?
    # In the previous code:
    # B_top_coeffs used j in 0,1 and k_local in 0,1.
    # B_bot_coeffs used j in 2,3 and k_local in 0,1.
    # So B_top was the top 2x2 block of B for a given k-pair?
    # No, B is 4x8.
    # The previous code indexed B as `V[j * p + k_global]`.
    # j in 0..3, k_global in 0..7.
    # B_top_coeffs: j in 0,1. k_local in 0,1. k_global = t*2 + k_local.
    # So for a fixed t, B_top uses rows 0,1 and columns 2t, 2t+1.
    # B_bot_coeffs: j in 2,3. k_local in 0,1. k_global = t*2 + k_local.
    # So for a fixed t, B_bot uses rows 2,3 and columns 2t, 2t+1.
    # A_left uses rows 0,1 of A (which is 2x4, so A[0..1, 0..1]).
    # A_right uses rows 0,1 of A (which is 2x4, so A[0..1, 2..3]).
    # So for each t (0..3), we compute:
    # C_t = A_left * B[0:2, 2t:2t+2] + A_right * B[2:4, 2t:2t+2]
    # This is exactly the t-th 2x2 block of C (columns 2t, 2t+1).
    # And we do this using Strassen for 2x2x2?
    # Wait, A_left is 2x2, B_top is 2x2, product is 2x2.
    # A_right is 2x2, B_bot is 2x2, product is 2x2.
    # So we are computing two 2x2x2 matrix multiplications for each t.
    # And we use Strassen for each?
    # The previous code used 7 multipliers for A_left*B_top and 7 for A_right*B_bot.
    # Total 14 per t. 4 t's -> 56.
    
    # Can we do better?
    # We are computing C = A * B where A is 2x4, B is 4x8.
    # This is equivalent to 2 independent 1x4x8 multiplications? No, A has 2 rows.
    # It is 2 independent 1x4x8 multiplications if the rows of A were independent? No.
    # It is a 2x4x8 tensor.
    # The rank of 2x4x8 matrix multiplication tensor.
    # R(2,4,8) <= R(2,4,4) + R(2,4,4) ?
    # R(2,4,4) is the rank of 2x4x4.
    # 2x4x4 can be done in rank 12? (2*4*4=32 entries, but rank is lower).
    # Actually, R(2,4,4) = 12 is a known result?
    # If R(2,4,4) = 12, then R(2,4,8) <= 24?
    # Let's check if we can find a rank 12 decomposition for 2x4x4.
    # If not, we stick to 56.
    # Given the time constraint and lack of external search, I will stick to the known valid solution of rank 56.
    # I will just clean up the code slightly.

    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    A_left_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    
    A_right_coeffs = [
        [(0, 2, 1), (1, 3, 1)],
        [(1, 2, 1), (1, 3, 1)],
        [(0, 2, 1)],
        [(1, 3, 1)],
        [(0, 2, 1), (0, 3, 1)],
        [(1, 2, 1), (0, 2, -1)],
        [(0, 3, 1), (1, 3, -1)]
    ]
    
    B_top_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    
    B_bot_coeffs = [
        [(2, 0, 1), (3, 1, 1)],
        [(2, 0, 1)],
        [(2, 1, 1), (3, 1, -1)],
        [(3, 0, 1), (2, 0, -1)],
        [(3, 1, 1)],
        [(2, 0, 1), (2, 1, 1)],
        [(3, 0, 1), (3, 1, 1)]
    ]
    
    C_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for t in range(4):
        for s in range(7):
            c_idx = t * 14 + s
            for i, j, coeff in A_left_coeffs[s]:
                U[i * m + j][c_idx] += coeff
            for j, k_local, coeff in B_top_coeffs[s]:
                k_global = t * 2 + k_local
                V[j * p + k_global][c_idx] += coeff
            for i, k_local, coeff in C_coeffs[s]:
                k_global = t * 2 + k_local
                W[k_global * n + i][c_idx] += coeff
                
        for s in range(7):
            c_idx = t * 14 + 7 + s
            for i, j, coeff in A_right_coeffs[s]:
                U[i * m + j][c_idx] += coeff
            for j, k_local, coeff in B_bot_coeffs[s]:
                k_global = t * 2 + k_local
                V[j * p + k_global][c_idx] += coeff
            for i, k_local, coeff in C_coeffs[s]:
                k_global = t * 2 + k_local
                W[k_global * n + i][c_idx] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
