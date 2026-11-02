"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u in R^4, v in R^4, w in R^4 (indices for A,B,C entries).
    A is 2x2, B is 2x2, C = A*B is 2x2.
    Indices: A[i,j] at i*2+j, B[i,j] at i*2+j, C[i,j] at i*2+j.
    """
    # Strassen's 7 multiplications for C = A*B:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # We need: sum_c U[i,j,c] * V[k,l,c] = W[i,j,k,l,c]  (with appropriate indices)
    # For our global problem, we'll compose these.
    
    terms = []
    # Each term: (u_vec, v_vec, w_vec) of length 4 each
    # u corresponds to A coefficients, v to B, w to C
    
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
    u1 = [1, 0, 0, 1]  # A00 + A11
    v1 = [1, 0, 0, 1]  # B00 + B11
    w1 = [1, 0, 0, 1]  # C00 + C11
    
    # M2 = (A10+A11)*B00 -> contributes to C10, C11 (negative)
    u2 = [0, 0, 1, 1]  # A10 + A11
    v2 = [1, 0, 0, 0]  # B00
    w2 = [0, 0, 1, -1]  # C10 - C11
    
    # M3 = A00*(B01-B11) -> contributes to C01, C11
    u3 = [1, 0, 0, 0]  # A00
    v3 = [0, 1, 0, -1]  # B01 - B11
    w3 = [0, 1, 0, 1]  # C01 + C11
    
    # M4 = A11*(B10-B00) -> contributes to C00, C10
    u4 = [0, 0, 0, 1]  # A11
    v4 = [0, 0, 1, -1]  # B10 - B00
    w4 = [1, 0, 1, 0]  # C00 + C10
    
    # M5 = (A00+A01)*B11 -> contributes to C00 (negative), C01
    u5 = [1, 1, 0, 0]  # A00 + A01
    v5 = [0, 0, 0, 1]  # B11
    w5 = [-1, 1, 0, 0]  # -C00 + C01
    
    # M6 = (A10-A00)*(B00+B01) -> contributes to C11
    u6 = [-1, 0, 1, 0]  # -A00 + A10
    v6 = [1, 1, 0, 0]  # B00 + B01
    w6 = [0, 0, 0, 1]  # C11
    
    # M7 = (A01-A11)*(B10+B11) -> contributes to C00
    u7 = [0, 1, 0, -1]  # A01 - A11
    v7 = [0, 0, 1, 1]  # B10 + B11
    w7 = [1, 0, 0, 0]  # C00
    
    terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7),
    ]
    return terms

def construct():
    """
    Construct a rank-42 decomposition for 2x4 * 4x6 matrix multiplication.
    
    Strategy: Split A (2x4) into two 2x2 blocks A_left, A_right.
    Split B (4x6) into two 6x2 blocks... wait, B is 4x6.
    Let A = [A_L | A_R] where A_L, A_R are 2x2.
    Let B = [B_T; B_B] where B_T, B_B are 2x6.
    Then C = A_L * B_T + A_R * B_B.
    
    A_L * B_T is 2x2 * 2x6 -> can be done as three 2x2x2 multiplications (rank 7 each = 21)
    A_R * B_B is 2x2 * 2x6 -> can be done as three 2x2x2 multiplications (rank 7 each = 21)
    Total: 42.
    
    For 2x2 * 2x6: treat the 6 columns as three pairs of columns.
    Each 2x2x2 multiplication handles 2 columns of output.
    """
    n, m, p = 2, 4, 6
    # A is 2x4, B is 4x6, C is 2x6
    # A indices: i in 0..1, j in 0..3 -> index i*m+j
    # B indices: j in 0..3, k in 0..5 -> index j*p+k  
    # Wait, B is m x p = 4 x 6. B[j,k] where j=0..3, k=0..5. Index j*p+k.
    # C indices: i in 0..1, k in 0..5. Output factor W[k*n+i, r] per the trace convention.
    # So C[i,k] corresponds to W[k*n+i, r] = W[6i+k... wait no.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So for C[i,k], the row index in W is k*n+i. Since n=2, row = 2k+i.
    # C has 2*6=12 entries. W is (p*n, rank) = (12, rank).
    # W[2k+i, r] corresponds to C[i,k].
    
    # Split A = [A_L | A_R], A_L = A[:, 0:2], A_R = A[:, 2:4]
    # Split B = [B_T; B_B], B_T = B[0:2, :], B_B = B[2:4, :]
    # C = A_L * B_T + A_R * B_B
    # A_L is 2x2, B_T is 2x6. Product is 2x6.
    # A_R is 2x2, B_B is 2x6. Product is 2x6.
    
    # For A_L * B_T: 2x2 * 2x6
    # Split B_T into three 2x2 blocks: B_T[:, 0:2], B_T[:, 2:4], B_T[:, 4:6]
    # Each 2x2x2 multiplication gives 2x2 output for 2 columns of C.
    
    # Total rank: 3 * 7 + 3 * 7 = 42
    
    strassen_terms = _strassen_2x2x2()
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (12, 42)
    
    # Helper to get A index: A[i, j] -> i*m + j = i*4 + j
    # Helper to get B index: B[j, k] -> j*p + k = j*6 + k
    # Helper to get W index for C[i, k]: 2k + i (since n=2, W[k*n+i, r])
    
    col = 0
    # Process A_L * B_T: A_L = A[:, 0:2], B_T = B[0:2, :]
    # B_T has 6 columns, split into 3 blocks of 2 columns each.
    for block_col in range(3):
        # block_col handles columns 2*block_col and 2*block_col+1 of B_T (and thus of C)
        for (u_vec, v_vec, w_vec) in strassen_terms:
            # u_vec[i*2+j] corresponds to A_L[i, j] = A[i, j] (since A_L is A[:, 0:2])
            # v_vec[k*2+l] corresponds to B_T[k, l] where k is row of B_T (0,1), l is col within block
            # Actually v_vec is length 4: v_vec[0]=B_T[0,0], v_vec[1]=B_T[0,1], v_vec[2]=B_T[1,0], v_vec[3]=B_T[1,1]
            # But in the block, column offset is 2*block_col
            # So B_T[k, l] = B[k, 2*block_col + l] where k=0,1 (rows of B_T)
            # B index: k * 6 + (2*block_col + l)
            # w_vec[i*2+l] corresponds to C_L[i, l] = C[i, 2*block_col + l]
            # W index for C[i, k_col] is 2*k_col + i
            
            for idx in range(4):
                i_loc = idx // 2  # row of A_L (0 or 1)
                j_loc = idx % 2   # col of A_L (0 or 1)
                a_idx = i_loc * 4 + j_loc  # A[i_loc, j_loc]
                U[a_idx][col] += u_vec[idx]
                
                # v_vec: for 2x2 B_T block
                k_loc = idx // 2   # row of B_T (0 or 1)
                l_loc = idx % 2    # col within block (0 or 1)
                b_col_global = 2 * block_col + l_loc
                b_idx = k_loc * 6 + b_col_global  # B[k_loc, b_col_global]
                V[b_idx][col] += v_vec[idx]
                
                # w_vec: for 2x2 C block
                i_out = idx // 2   # row of C (0 or 1)
                k_out_loc = idx % 2  # col within block (0 or 1)
                k_out_global = 2 * block_col + k_out_loc
                w_idx = 2 * k_out_global + i_out  # W[k*n+i] = W[2*k_out_global + i_out]
                W[w_idx][col] += w_vec[idx]
            
            col += 1
    
    # Process A_R * B_B: A_R = A[:, 2:4], B_B = B[2:4, :]
    for block_col in range(3):
        for (u_vec, v_vec, w_vec) in strassen_terms:
            # u_vec[i*2+j] corresponds to A_R[i, j] = A[i, 2+j]
            # v_vec[k*2+l] corresponds to B_B[k, l] = B[2+k, 2*block_col + l]
            # w_vec[i*2+l] corresponds to C_R[i, l] = C[i, 2*block_col + l]
            
            for idx in range(4):
                i_loc = idx // 2
                j_loc = idx % 2
                a_idx = i_loc * 4 + (2 + j_loc)  # A[i_loc, 2+j_loc]
                U[a_idx][col] += u_vec[idx]
                
                k_loc = idx // 2
                l_loc = idx % 2
                b_col_global = 2 * block_col + l_loc
                b_idx = (2 + k_loc) * 6 + b_col_global  # B[2+k_loc, b_col_global]
                V[b_idx][col] += v_vec[idx]
                
                i_out = idx // 2
                k_out_loc = idx % 2
                k_out_global = 2 * block_col + k_out_loc
                w_idx = 2 * k_out_global + i_out
                W[w_idx][col] += w_vec[idx]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
