import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 140.0  # 10s safety margin

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Naive rank-96 decomposition
    naive_rank = n * m * p  # 3*4*8 = 96
    U_naive = [[0] * naive_rank for _ in range(n * m)]
    V_naive = [[0] * naive_rank for _ in range(m * p)]
    W_naive = [[0] * naive_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j][col] = 1
                V_naive[j * p + k][col] = 1
                W_naive[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_naive, "V": V_naive, "W": W_naive}
    best_rank = naive_rank
    
    def verify(U, V, W, rank):
        """Check if the decomposition is valid"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Compute the tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U_arr[i*m+j, :rank], V_arr[j*p+k, :rank] * W_arr[k*n+i, :rank])
        
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True
    
    # Try to build a better decomposition using Strassen for 2x2x2 sub-blocks
    # Strategy:
    # Split n=3 into 2+1, m=4 into 2+2, p=8 into 4+4
    # For the 2x2x4 blocks (where n_block=2), split p=4 into 2+2 to get 2x2x2 sub-blocks
    # Each 2x2x2 can use Strassen rank 7 instead of naive rank 8
    # For the 1x2x4 blocks (where n_block=1), use naive rank 8
    
    # Let's try a different approach: use rank-7 Strassen for all 2x2x2 blocks
    # and see if we can reduce the total rank.
    
    # Block structure:
    # i_blocks: [0,1] (size 2), [2] (size 1)
    # j_blocks: [0,1] (size 2), [2,3] (size 2)
    # k_blocks: [0,1,2,3] (size 4), [4,5,6,7] (size 4)
    
    # For i_block=0 (size 2), j_block (size 2), k_block (size 4):
    #   Split k_block into two k_sub blocks of size 2
    #   Each gives a 2x2x2 block -> Strassen rank 7
    #   Total per (j_block, k_block): 2 * 7 = 14
    #   2 j_blocks * 2 k_blocks = 4 such groups -> 4 * 14 = 56
    
    # For i_block=1 (size 1), j_block (size 2), k_block (size 4):
    #   Naive rank = 1 * 2 * 4 = 8
    #   2 j_blocks * 2 k_blocks = 4 such groups -> 4 * 8 = 32
    
    # Total rank: 56 + 32 = 88
    
    def add_strassen_2x2x2(U_list, V_list, W_list, offset_i, offset_j, offset_k, col_start):
        """Add a 2x2x2 Strassen decomposition at the given offsets.
        Returns the new column index."""
        # Strassen 7 terms for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)  [not used in output]
        # M7 = (A01-A10)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        col = col_start
        
        # M1: (A00+A11)(B00+B11) -> C00:+1, C11:+1
        U_list[(offset_i + 0) * m + (offset_j + 0)][col] += 1
        U_list[(offset_i + 1) * m + (offset_j + 1)][col] += 1
        V_list[(offset_j + 0) * p + (offset_k + 0)][col] += 1
        V_list[(offset_j + 1) * p + (offset_k + 1)][col] += 1
        W_list[(offset_k + 0) * n + (offset_i + 0)][col] += 1
        W_list[(offset_k + 1) * n + (offset_i + 1)][col] += 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10:+1, C11:-1
        U_list[(offset_i + 1) * m + (offset_j + 0)][col] += 1
        U_list[(offset_i + 1) * m + (offset_j + 1)][col] += 1
        V_list[(offset_j + 0) * p + (offset_k + 0)][col] += 1
        W_list[(offset_k + 0) * n + (offset_i + 1)][col] += 1
        W_list[(offset_k + 1) * n + (offset_i + 1)][col] += -1
        col += 1
        
        # M3: A00(B10-B11) -> C01:+1, C11:+1
        U_list[(offset_i + 0) * m + (offset_j + 0)][col] += 1
        V_list[(offset_j + 1) * p + (offset_k + 0)][col] += 1
        V_list[(offset_j + 1) * p + (offset_k + 1)][col] += -1
        W_list[(offset_k + 0) * n + (offset_i + 0)][col] += 1
        W_list[(offset_k + 1) * n + (offset_i + 1)][col] += 1
        col += 1
        
        # M4: A11(B01-B00) -> C00:+1, C10:+1
        U_list[(offset_i + 1) * m + (offset_j + 1)][col] += 1
        V_list[(offset_j + 0) * p + (offset_k + 1)][col] += 1
        V_list[(offset_j + 0) * p + (offset_k + 0)][col] += -1
        W_list[(offset_k + 0) * n + (offset_i + 0)][col] += 1
        W_list[(offset_k + 1) * n + (offset_i + 1)][col] += 1
        col += 1
        
        # M5: (A00+A01)B11 -> C00:-1, C01:+1
        U_list[(offset_i + 0) * m + (offset_j + 0)][col] += 1
        U_list[(offset_i + 0) * m + (offset_j + 1)][col] += 1
        V_list[(offset_j + 1) * p + (offset_k + 1)][col] += 1
        W_list[(offset_k + 0) * n + (offset_i + 0)][col] += -1
        W_list[(offset_k + 1) * n + (offset_i + 1)][col] += 1
        col += 1
        
        # M7: (A01-A10)(B10+B11) -> C00:-1, C11:+1
        U_list[(offset_i + 0) * m + (offset_j + 1)][col] += 1
        U_list[(offset_i + 1) * m + (offset_j + 0)][col] += -1
        V_list[(offset_j + 1) * p + (offset_k + 0)][col] += 1
        V_list[(offset_j + 1) * p + (offset_k + 1)][col] += 1
        W_list[(offset_k + 0) * n + (offset_i + 0)][col] += -1
        W_list[(offset_k + 1) * n + (offset_i + 1)][col] += 1
        col += 1
        
        return col
    
    # Build the rank-88 decomposition
    target_rank = 88
    U_new = [[0] * target_rank for _ in range(n * m)]
    V_new = [[0] * target_rank for _ in range(m * p)]
    W_new = [[0] * target_rank for _ in range(p * n)]
    
    col = 0
    
    # Process 2x2x4 blocks for i_block=0 (rows 0-1)
    # Split into 2x2x2 sub-blocks
    for j_block in [0, 1]:  # j_blocks: [0,1] and [2,3]
        for k_block in [0, 1]:  # k_blocks: [0-3] and [4-7]
            for k_sub in [0, 1]:  # Split k_block into two 2-element sub-blocks
                offset_i = 0
                offset_j = j_block * 2
                offset_k = k_block * 4 + k_sub * 2
                col = add_strassen_2x2x2(U_new, V_new, W_new, offset_i, offset_j, offset_k, col)
    
    # Process 1x2x4 blocks for i_block=1 (row 2) - naive
    for j_block in [0, 1]:
        for k_block in [0, 1]:
            offset_i = 2
            offset_j = j_block * 2
            offset_k = k_block * 4
            for j in range(2):
                for k in range(4):
                    row_u = offset_i * m + (offset_j + j)
                    row_v = (offset_j + j) * p + (offset_k + k)
                    row_w = (offset_k + k) * n + offset_i
                    U_new[row_u][col] = 1
                    V_new[row_v][col] = 1
                    W_new[row_w][col] = 1
                    col += 1
    
    # Verify the new decomposition
    if col == target_rank and verify(U_new, V_new, W_new, target_rank):
        best_result = {"U": U_new, "V": V_new, "W": W_new}
        best_rank = target_rank
    
    if time.time() > DEADLINE:
        return best_result
    
    # Try to find further improvements by searching for linear dependencies
    # among the rank-88 terms
    
    # Convert to numpy for efficient computation
    U_arr = np.array(best_result["U"], dtype=float)
    V_arr = np.array(best_result["V"], dtype=float)
    W_arr = np.array(best_result["W"], dtype=float)
    rank = best_rank
    
    # For each pair of terms, check if one can be expressed as a linear combination
    # of the others. This is expensive, so we'll do a limited search.
    
    # Alternative: try to find a lower-rank decomposition by solving a system
    # of equations. For rank r, we need to find U, V, W of shape (nm, r), (mp, r), (pn, r)
    # such that for all i,j,k: sum_{r=1}^{R} U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # This is a non-convex problem. Let's try a greedy approach:
    # Start with the rank-88 solution and try to eliminate terms.
    
    # For now, return the best result found
    return best_result
