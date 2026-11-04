"""Matrix multiplication 2,5,6 - rank-minimal exact tensor decomposition."""
import time
import sys
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for 2x5 @ 5x6 multiplication.
    
    Strategy: Use a Kronecker-product / block-based approach combined with
    Strassen's algorithm where applicable, plus rank-1 terms for degenerate blocks.
    
    For 2x5 * 5x6:
    - Split the 5x5 middle into blocks: we can think of it as combining
      2x2, 2x3, 3x2, 3x3 type multiplications.
    - Actually, a cleaner approach: use the known rank bounds and construct
      via bilinear algorithms.
    
    Let's use a systematic approach:
    n=2, m=5, p=6.
    
    We can partition the indices:
    A is 2x5: rows i in {0,1}, cols j in {0..4}
    B is 5x6: rows j in {0..4}, cols k in {0..5}
    C is 2x6: rows i in {0,1}, cols k in {0..5}
    
    One effective approach is to use the "block Strassen" or "laser method" style,
    but for small dimensions, we can try to find an explicit low-rank decomposition.
    
    Known: rank(2x5x6) <= ?
    Schoolbook: 2*5*6 = 60
    
    We can try to get below 60 using:
    1. Strassen for 2x2x2 blocks (rank 7 instead of 8)
    2. Rank-1 for degenerate 2x1x2, 1x2x2, 2x2x1 blocks
    
    Let's partition the 5 middle dimension as 2+3 and the 6 output as 3+3 or 2+4.
    
    Actually, let's try a direct search for a low-rank decomposition using
    the structure of the multiplication tensor.
    
    For 2x5*5x6, we can view it as:
    C = A * B where A is 2x5, B is 5x6.
    
    Let's try to use the fact that we can combine some of the 5 intermediate
    products using linear combinations.
    
    Approach: Try to find a decomposition of rank r < 60 by exploiting
    the linearity in the middle dimension.
    
    For each pair (i, k) in {0,1} x {0..5}, C[i,k] = sum_j A[i,j] B[j,k].
    
    The tensor T has entries T[i,j,k] = 1 for all valid i,j,k (since the
    multiplication tensor for C[i,k] = sum_j A[i,j] B[j,k] has coefficient 1
    for each term A[i,j]*B[j,k] contributing to C[i,k]).
    
    Wait, the tensor is: sum_{i,j,k} A[i,j] B[j,k] C[i,k] (with C being the
    dual of the output). The coefficient of A[i,j] B[j,k] C[i,k] is 1.
    
    So we need: sum_c U[i*5+j, c] * V[j*6+k, c] * W[k*2+i, c] = 1 for all i,j,k.
    
    Let me try to construct a decomposition using the following idea:
    Group the 5 middle indices into two groups of sizes 2 and 3.
    For the 2-group, use Strassen (rank 7) for 2x2x2 sub-problems.
    For the 3-group, use schoolbook (rank 9) for 2x3x2 or similar.
    
    But we have mixed dimensions. Let me be more careful.
    
    Alternative: Use a general bilinear algorithm search.
    
    For 2x5*5x6, let's try rank values and see if we can find a valid decomposition.
    
    Actually, let's use a known result: the rank of 2xm*mp multiplication can be
    bounded. For 2x5*5x6, a good bound might be around 45-50.
    
    Let me try a constructive approach using block matrix multiplication.
    
    Write A as [A1 A2] where A1 is 2x2, A2 is 2x3.
    Write B as [B1; B2] where B1 is 2x6, B2 is 3x6.
    Then C = A1*B1 + A2*B2.
    
    A1*B1: 2x2 * 2x6. This is like multiplying a 2x2 by a 2x6.
    We can view this as 3 separate 2x2*2x2 multiplications (since 6 = 3*2).
    Each 2x2*2x2 can be done in rank 7 (Strassen), so total rank for A1*B1: 3*7 = 21.
    
    A2*B2: 2x3 * 3x6. This is 2x3 * 3x6.
    We can view B2 as [B2a B2b] where B2a is 3x3, B2b is 3x3.
    A2*B2a: 2x3 * 3x3. 
    A2*B2b: 2x3 * 3x3.
    
    For 2x3 * 3x3:
    C = A * B where A is 2x3, B is 3x3.
    C[i,k] = sum_j A[i,j] B[j,k].
    This is 2*3*3 = 18 terms in schoolbook.
    
    Can we do better? 2x3*3x3: rank is at most 18.
    Using Strassen-like: partition A as 2x3 = [A_left | A_right] where each is 2x1 or...
    This gets complicated.
    
    Let me just use schoolbook for the 3x3 parts: 18 each, so 36 for both.
    Total: 21 + 36 = 57. That's only slightly better than 60.
    
    Let me try a different partition: 5 = 1 + 4.
    A = [A1 | A2] where A1 is 2x1, A2 is 2x4.
    B = [B1; B2] where B1 is 1x6, B2 is 4x6.
    C = A1*B1 + A2*B2.
    
    A1*B1: 2x1 * 1x6. This is just outer product: rank = 2*6 = 12? No.
    C[i,k] = A[i,0] * B[0,k]. For each (i,k), this is a single product.
    So we need 2*6 = 12 rank-1 terms. Rank = 12.
    
    A2*B2: 2x4 * 4x6. 
    2x4 * 4x6: schoolbook is 2*4*6 = 48.
    Using Strassen for 2x2*2x2 blocks: 4 = 2+2, 6 = 3*2.
    So 2x4*4x6 = (2x2 * 2x6) * 2 blocks in the 4-dim.
    Each 2x2*2x6: partition 6 = 3*2, so 3 copies of 2x2*2x2.
    Each 2x2*2x2: rank 7 (Strassen).
    So 2x2*2x6: rank 3*7 = 21.
    2x4*4x6: two such blocks, rank 2*21 = 42.
    
    Total: 12 + 42 = 54.
    
    Better! Let's see if we can do even better.
    
    Try 5 = 2 + 3:
    A = [A1 | A2], A1: 2x2, A2: 2x3.
    B = [B1; B2], B1: 2x6, B2: 3x6.
    C = A1*B1 + A2*B2.
    
    A1*B1: 2x2 * 2x6.
    6 = 3*2, so 3 copies of 2x2*2x2.
    Each rank 7, total 21.
    
    A2*B2: 2x3 * 3x6.
    6 = 3*2, so 3 copies of 2x3*3x2.
    2x3*3x2: schoolbook is 2*3*2 = 12.
    Can we do better? 2x3*3x2.
    
    C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k].
    This is a 2x3 * 3x2 multiplication.
    
    Known: rank(2x3x2) = ?
    I believe rank(2x3x2) <= 11 or so. Let me think.
    
    Actually, 2x3*3x2: we can use the fact that this is equivalent to
    multiplying a 2x3 by a 3x2. The tensor rank is at most 2*3*2 = 12.
    
    One approach: use the identity matrix in the middle.
    C = A * I_3 * B, but that doesn't help directly.
    
    Let me just use schoolbook for 2x3*3x2: rank 12 each, 3 copies = 36.
    Total: 21 + 36 = 57. Worse than the 1+4 split.
    
    So 1+4 split gives 54. Let's try 5 = 1+1+3:
    A = [A1 | A2 | A3], A1: 2x1, A2: 2x1, A3: 2x3.
    B = [B1; B2; B3], B1: 1x6, B2: 1x6, B3: 3x6.
    C = A1*B1 + A2*B2 + A3*B3.
    
    A1*B1: rank 12.
    A2*B2: rank 12.
    A3*B3: 2x3*3x6 = 3 copies of 2x3*3x2, each rank 12, total 36.
    
    Total: 12 + 12 + 36 = 60. Same as schoolbook.
    
    So 1+4 seems best so far at 54.
    
    Let me try to improve 2x3*3x2.
    
    2x3*3x2: C[i,k] = sum_j A[i,j] B[j,k], i in {0,1}, j in {0,1,2}, k in {0,1}.
    
    There are 12 monomials: A[i,j]*B[j,k] for all (i,j,k).
    
    Can we express this with fewer than 12 rank-1 terms?
    
    The tensor is T[i,j,k] = 1 for all i,j,k.
    
    We need sum_c U[i,j,c] V[j,k,c] W[i,k,c] = 1 for all i,j,k.
    
    Note that the indices are i (2), j (3), k (2).
    
    This is the matrix multiplication tensor for 2x3 * 3x2.
    
    Known rank bounds: I believe rank(2x3x2) = 11. Let me try to construct it.
    
    Actually, let's just use a search or known decomposition.
    
    For now, let's implement the 1+4 split with rank 54 and see if we can
    refine the 2x3*3x2 part.
    
    Let me implement the 1+4 split:
    
    Total rank: 12 + 42 = 54.
    
    Let me code this up.
    """
    n, m, p = 2, 5, 6
    rank_total = 54
    
    U = [[0] * rank_total for _ in range(n * m)]  # 10 x 54
    V = [[0] * rank_total for _ in range(m * p)]  # 30 x 54
    W = [[0] * rank_total for _ in range(p * n)]  # 12 x 54
    
    col = 0
    
    # Part 1: A1 * B1 where A1 is 2x1 (j=0), B1 is 1x6 (j=0)
    # C[i,k] += A[i,0] * B[0,k]
    # This is 2x6 rank-1 terms.
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 12
    
    # Part 2: A2 * B2 where A2 is 2x4 (j=1..4), B2 is 4x6 (j=1..4)
    # Use Strassen for 2x2*2x2 blocks.
    # A2 has columns j=1,2,3,4. Split into two 2-column blocks: (j=1,2) and (j=3,4).
    # B2 has rows j=1,2,3,4. Split into two 2-row blocks: (j=1,2) and (j=3,4).
    # For each pair of blocks (A2_left * B2_top, A2_left * B2_bottom, etc.)
    # Actually, C += A2 * B2 = sum over block pairs.
    # A2 is 2x4, B2 is 4x6.
    # Split 4 into 2+2: A2 = [A2a | A2b] where A2a is 2x2, A2b is 2x2.
    # B2 = [B2a; B2b] where B2a is 2x6, B2b is 2x6.
    # C += A2a * B2a + A2b * B2b.
    
    # Each 2x2 * 2x6: split 6 into 3*2, so 3 copies of 2x2*2x2.
    # Each 2x2*2x2 uses Strassen rank 7.
    
    def strassen_2x2x2(U_mat, V_mat, W_mat, col, j_offset, k_offset):
        """
        Add Strassen decomposition for 2x2*2x2 multiplication.
        U_mat: U matrix (will modify in place)
        V_mat: V matrix
        W_mat: W matrix
        col: starting column
        j_offset: offset for j index in U and V (the middle dimension)
        k_offset: offset for k index in V and W (the output dimension)
        """
        # Strassen for 2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A10) * B11
        # M6 = (A11 - A00) * (B00 + B10)
        # M7 = (A01 - A11) * (B11 + B10)
        #
        # C00 = M1 - M4 + M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 + M6
        
        # Local indices: i_local in {0,1}, j_local in {0,1}, k_local in {0,1}
        # Global: i = i_local (since n=2, i is 0 or 1)
        # j = j_offset + j_local
        # k = k_offset + k_local
        
        # M1: (A00 + A11) * (B00 + B11)
        # U: A00 and A11 => j_local 0 and 1, i_local 0 and 1
        U_mat[0 * m + j_offset + 0][col] = 1
        U_mat[1 * m + j_offset + 1][col] = 1
        V_mat[(j_offset + 0) * p + k_offset + 0][col] = 1
        V_mat[(j_offset + 1) * p + k_offset + 1][col] = 1
        W_mat[(k_offset + 0) * n + 0][col] = 1
        W_mat[(k_offset + 1) * n + 1][col] = 1
        col += 1
        
        # M2: (A10 + A11) * B00
        U_mat[1 * m + j_offset + 0][col] = 1
        U_mat[1 * m + j_offset + 1][col] = 1
        V_mat[(j_offset + 0) * p + k_offset + 0][col] = 1
        W_mat[(k_offset + 0) * n + 1][col] = 1
        col += 1
        
        # M3: A00 * (B01 - B11)
        U_mat[0 * m + j_offset + 0][col] = 1
        V_mat[(j_offset + 0) * p + k_offset + 1][col] = 1
        V_mat[(j_offset + 1) * p + k_offset + 1][col] = -1
        W_mat[(k_offset + 1) * n + 0][col] = 1
        col += 1
        
        # M4: A11 * (B10 - B00)
        U_mat[1 * m + j_offset + 1][col] = 1
        V_mat[(j_offset + 1) * p + k_offset + 0][col] = 1
        V_mat[(j_offset + 0) * p + k_offset + 0][col] = -1
        W_mat[(k_offset + 0) * n + 1][col] = 1
        col += 1
        
        # M5: (A00 + A10) * B11
        U_mat[0 * m + j_offset + 0][col] = 1
        U_mat[1 * m + j_offset + 0][col] = 1
        V_mat[(j_offset + 1) * p + k_offset + 1][col] = 1
        W_mat[(k_offset + 1) * n + 1][col] = 1
        col += 1
        
        # M6: (A11 - A00) * (B00 + B10)
        U_mat[1 * m + j_offset + 1][col] = 1
        U_mat[0 * m + j_offset + 0][col] = -1
        V_mat[(j_offset + 0) * p + k_offset + 0][col] = 1
        V_mat[(j_offset + 1) * p + k_offset + 0][col] = 1
        W_mat[(k_offset + 0) * n + 1][col] = 1
        W_mat[(k_offset + 1) * n + 1][col] = 1
        col += 1
        
        # M7: (A01 - A11) * (B11 + B10)
        U_mat[0 * m + j_offset + 1][col] = 1
        U_mat[1 * m + j_offset + 1][col] = -1
        V_mat[(j_offset + 1) * p + k_offset + 1][col] = 1
        V_mat[(j_offset + 1) * p + k_offset + 0][col] = 1
        W_mat[(k_offset + 1) * n + 1][col] = 1
        col += 1
        
        return col
    
    # A2a * B2a: j_offset = 1, for k blocks
    for k_block in range(3):
        k_offset = k_block * 2
        col = strassen_2x2x2(U, V, W, col, j_offset=1, k_offset=k_offset)
    
    # A2b * B2b: j_offset = 3
    for k_block in range(3):
        k_offset = k_block * 2
        col = strassen_2x2x2(U, V, W, col, j_offset=3, k_offset=k_offset)
    
    # col should be 12 + 42 = 54
    assert col == rank_total, f"Expected {rank_total}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
