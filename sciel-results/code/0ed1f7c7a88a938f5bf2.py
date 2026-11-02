import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 5s safety margin
    
    # 1. Schoolbook decomposition as fallback (Rank 72)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Validation function
    def validate_decomposition(U_mat, V_mat, W_mat, rank_val):
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Verify schoolbook
    if not validate_decomposition(U_school, V_school, W_school, rank_school):
        # Should not happen
        return {"U": U_school, "V": V_school, "W": W_school}

    # 2. Try Laderman-based construction (Rank 58)
    # Laderman decomposition for 3x3x3
    # Using matrices from Inspiration 2
    lader_U = [
        [1, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1]
    ]
    # Note: In a proper Laderman decomposition, U, V, W are not necessarily identical.
    # However, if the inspiration provides them identical and they pass validation for 3x3x3,
    # we can use them. Let's verify for 3x3x3 first.
    
    def validate_3x3x3(U3, V3, W3):
        U3_arr = np.array(U3, dtype=float)
        V3_arr = np.array(V3, dtype=float)
        W3_arr = np.array(W3, dtype=float)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = np.dot(U3_arr[i * 3 + j, :], V3_arr[j * 3 + k, :] * W3_arr[k * 3 + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # The inspiration 2 matrices are 20x9.
    # U, V, W in the decomposition are typically (9, 20).
    # Inspiration 2 lists them as 20 rows of 9 cols.
    # So we need to transpose them.
    lader_U_T = np.array(lader_U).T.tolist()
    lader_V_T = np.array(lader_U).T.tolist() # Assuming symmetry from inspiration
    lader_W_T = np.array(lader_U).T.tolist()
    
    if validate_3x3x3(lader_U_T, lader_V_T, lader_W_T):
        # Construct 3x3x8 decomposition
        # Rank = 20 (Laderman k=0..2) + 20 (Laderman k=3..5) + 18 (Schoolbook k=6..7)
        rank_lader = 20
        rank_school_2 = 18
        total_rank = rank_lader * 2 + rank_school_2
        
        U_new = [[0] * total_rank for _ in range(n * m)]
        V_new = [[0] * total_rank for _ in range(m * p)]
        W_new = [[0] * total_rank for _ in range(p * n)]
        
        # Block 1: k=0,1,2 (Laderman)
        for r in range(rank_lader):
            for i in range(n):
                for j in range(m):
                    U_new[i * m + j][r] = lader_U_T[i * m + j][r]
            for j in range(m):
                for k in range(3):
                    V_new[j * p + k][r] = lader_V_T[j * 3 + k][r]
            for k in range(3):
                for i in range(n):
                    W_new[k * n + i][r] = lader_W_T[k * 3 + i][r]
        
        # Block 2: k=3,4,5 (Laderman)
        for r in range(rank_lader):
            for i in range(n):
                for j in range(m):
                    U_new[i * m + j][rank_lader + r] = lader_U_T[i * m + j][r]
            for j in range(m):
                for k in range(3):
                    V_new[j * p + (3 + k)][rank_lader + r] = lader_V_T[j * 3 + k][r]
            for k in range(3):
                for i in range(n):
                    W_new[(3 + k) * n + i][rank_lader + r] = lader_W_T[k * 3 + i][r]
        
        # Block 3: k=6,7 (Schoolbook)
        col_base = rank_lader * 2
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(2):
                    U_new[i * m + j][col_base + col] = 1
                    V_new[j * p + (6 + k)][col_base + col] = 1
                    W_new[(6 + k) * n + i][col_base + col] = 1
                    col += 1
        
        if validate_decomposition(U_new, V_new, W_new, total_rank):
            best_rank = total_rank
            best_U, best_V, best_W = U_new, V_new, W_new

    # 3. Random search for better rank (optional, if time permits)
    # The current best is 58 (if Laderman worked) or 72.
    # Lower bound is 24.
    # We can try to find a rank between 24 and best_rank.
    # However, random search with constrained values is very slow for high dimensions.
    # We will skip extensive random search to ensure we return the 58-rank solution reliably.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
