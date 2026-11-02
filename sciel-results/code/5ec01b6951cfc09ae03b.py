"""
Tensor decomposition for 3x4 x 4x8 matrix multiplication.
Uses Strassen's algorithm on the 4x4 x 4x8 core (after padding 3->4),
then prunes the unused row.
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2 * 2x2 -> 2x2 using Strassen's algorithm.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    
    Indexing:
    U[i*2+j, c] for input A[i,j]
    V[j*2+k, c] for input B[j,k]
    W[k*2+i, c] for output C[i,k]  (trace convention: W[k*n+i, r] with n=2)
    """
    n = 2
    rank = 7
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Strassen's 7 products for 2x2 blocks:
    # Let A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_strassen_decomposition(n, m, p):
    """
    Build a tensor decomposition for n x m * m x p using Strassen-style recursion.
    For 3x4 x 4x8:
    - Pad n=3 to 4, giving 4x4 x 4x8
    - Split into 2x2 blocks of 2x2 and 2x4 matrices
    - Use Strassen at the block level (7 products)
    - Each product is a 2x2 * 2x4 multiplication
    - Use schoolbook for 2x2 * 2x4 (rank 16 each)
    - Total rank: 7 * 16 = 112
    - Then prune the 4th row of A and corresponding output row
    
    Actually, let me think more carefully.
    
    4x4 x 4x8:
    A is 4x4, split into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    B is 4x8, split into 2x2 blocks: B11, B12, B21, B22 (each 2x4)
    C is 4x8, split into 2x2 blocks: C11, C12, C21, C22 (each 2x4)
    
    C11 = A11*B11 + A21*B21
    C12 = A11*B12 + A21*B22
    C21 = A12*B11 + A22*B21
    C22 = A12*B12 + A22*B22
    
    This is schoolbook at the block level. To use Strassen, I need to express
    these in terms of 7 bilinear forms.
    
    Let me define:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    Then:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Wait, this is for 2x2 * 2x2 where each "element" is a matrix. But here
    the blocks are 2x2 and 2x4, so the products are 2x2 * 2x4 -> 2x4.
    
    Each Mx is a product of a 2x2 matrix (sum/difference of A blocks) and a
    2x4 matrix (sum/difference of B blocks), giving a 2x4 result.
    
    The rank of a 2x2 * 2x4 multiplication is 2*2*4 = 16 (schoolbook).
    So total rank = 7 * 16 = 112.
    
    But we can do better! The 2x2 * 2x4 can be seen as 2 copies of 2x1 * 1x4,
    each of rank 8, so total 16. Or we can use Strassen on the 2x2 part if we
    view it as 2x2 * 2x2 tensor 1x1... no, that doesn't help.
    
    Actually, let me just implement the schoolbook for the 2x2 * 2x4 products
    and see what rank we get.
    
    For now, let me implement the block-level Strassen with schoolbook within blocks.
    """
    n, m, p = 4, 4, 8  # Padded dimensions
    
    # Get the Strassen 2x2x2 decomposition for the block level
    U_block, V_block, W_block = _strassen_2x2x2()
    
    # Now build the full decomposition.
    # A is 4x4: indices (i,j) with i in 0..3, j in 0..3
    # B is 4x8: indices (j,k) with j in 0..3, k in 0..7
    # C is 4x8: indices (i,k) with i in 0..3, k in 0..7
    
    # Block structure:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:4], B12 = B[0:2, 4:8], B21 = B[2:4, 0:4], B22 = B[2:4, 4:8]
    # C11 = C[0:2, 0:4], C12 = C[0:2, 4:8], C21 = C[2:4, 0:4], C22 = C[2:4, 4:8]
    
    # For each of the 7 Strassen products, we need to compute a 2x2 * 2x4 product.
    # The 2x2 matrix comes from a sum/difference of A blocks.
    # The 2x4 matrix comes from a sum/difference of B blocks.
    # The result is a 2x4 matrix.
    
    # Let me build the full U, V, W matrices.
    rank = 7 * 16  # 7 Strassen products, each of rank 16
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # For each Strassen product c (0..6), we have:
    # - A combination: sum of A blocks with coefficients from U_block
    # - B combination: sum of B blocks with coefficients from V_block
    # - Output combination: sum of C blocks with coefficients from W_block
    
    # The A combination is a 2x2 matrix. Let's call it A_comb.
    # A_comb[i,j] = sum over A block elements with coefficients.
    
    # Similarly for B_comb (2x4) and C_comb (2x4).
    
    # For the schoolbook decomposition of A_comb * B_comb -> C_comb:
    # For each (i,j) in A_comb and (k) in B_comb, we have a product A_comb[i,j] * B_comb[j,k]
    # that contributes to C_comb[i,k].
    
    # But A_comb[i,j] is a linear combination of A elements.
    # B_comb[j,k] is a linear combination of B elements.
    # C_comb[i,k] is a linear combination of C elements.
    
    # So for each (i,j,k), we have a term:
    # (sum_a U_A[a, i,j] * A[a]) * (sum_b V_B[b, j,k] * B[b]) -> sum_c W_C[c, i,k] * C[c]
    
    # This is a rank-1 term in the tensor decomposition.
    
    # Let me define the mappings:
    # A element index: a = i_A * 4 + j_A where i_A in 0..3, j_A in 0..3
    # B element index: b = j_B * 8 + k_B where j_B in 0..3, k_B in 0..7
    # C element index: c = k_C * 4 + i_C where i_C in 0..3, k_C in 0..7 (trace convention: W[k*n+i])
    
    # For block (bi, bj) of A (bi in 0..1, bj in 0..1), the elements are:
    # A[bi*2 + i, bj*2 + j] for i in 0..1, j in 0..1
    # Index: (bi*2 + i) * 4 + (bj*2 + j)
    
    # For block (bj, bk) of B (bj in 0..1, bk in 0..1), the elements are:
    # B[bj*2 + j, bk*4 + k] for j in 0..1, k in 0..3
    # Index: (bj*2 + j) * 8 + (bk*4 + k)
    
    # For block (bi, bk) of C (bi in 0..1, bk in 0..1), the elements are:
    # C[bi*2 + i, bk*4 + k] for i in 0..1, k in 0..3
    # Index: (bk*4 + k) * 4 + (bi*2 + i)  (trace convention)
    
    # Now, for each Strassen product c (0..6):
    # U_block[:, c] gives coefficients for A blocks.
    # V_block[:, c] gives coefficients for B blocks.
    # W_block[:, c] gives coefficients for C blocks.
    
    # U_block has shape (4, 7), indexed by [block_idx, c]
    # block_idx: 0=A11, 1=A12, 2=A21, 3=A22
    
    # V_block has shape (4, 7), indexed by [block_idx, c]
    # block_idx: 0=B11, 1=B12, 2=B21, 3=B22
    
    # W_block has shape (4, 7), indexed by [block_idx, c]
    # block_idx: 0=C11, 1=C12, 2=C21, 3=C22
    
    col = 0
    for c in range(7):
        # Get the block coefficients
        U_coeff = U_block[:, c]  # shape (4,)
        V_coeff = V_block[:, c]  # shape (4,)
        W_coeff = W_block[:, c]  # shape (4,)
        
        # For each (i, j) in the 2x2 A_comb and each k in the 2x4 B_comb:
        # A_comb[i, j] = sum_{bi, bj} U_coeff[bi*2+bj] * A[bi*2+i, bj*2+j]
        # B_comb[j, k] = sum_{bj, bk} V_coeff[bj*2+bk] * B[bj*2+j, bk*4+k]
        # C_comb[i, k] = sum_{bi, bk} W_coeff[bi*2+bk] * C[bi*2+i, bk*4+k]
        
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    # This is one schoolbook term in the c-th Strassen product.
                    
                    # A part: A_comb[i, j] = sum_{bi, bj} U_coeff[bi*2+bj] * A[bi*2+i, bj*2+j]
                    for bi in range(2):
                        for bj in range(2):
                            coeff_a = U_coeff[bi*2 + bj]
                            if coeff_a != 0:
                                a_idx = (bi*2 + i) * 4 + (bj*2 + j)
                                U[a_idx, col] += coeff_a
                    
                    # B part: B_comb[j, k] = sum_{bj, bk} V_coeff[bj*2+bk] * B[bj*2+j, bk*4+k]
                    for bj in range(2):
                        for bk in range(2):
                            coeff_b = V_coeff[bj*2 + bk]
                            if coeff_b != 0:
                                b_idx = (bj*2 + j) * 8 + (bk*4 + k)
                                V[b_idx, col] += coeff_b
                    
                    # C part: C_comb[i, k] = sum_{bi, bk} W_coeff[bi*2+bk] * C[bi*2+i, bk*4+k]
                    for bi in range(2):
                        for bk in range(2):
                            coeff_c = W_coeff[bi*2 + bk]
                            if coeff_c != 0:
                                c_idx = (bk*4 + k) * 4 + (bi*2 + i)
                                W[c_idx, col] += coeff_c
                    
                    col += 1
    
    # Now we have a decomposition for 4x4 x 4x8 with rank 112.
    # But we only need 3x4 x 4x8, so we can prune the 4th row of A.
    # The 4th row of A corresponds to i=3, which is in block bi=1, i=1.
    
    # Find all columns where U[3, :] is nonzero (i.e., involves A[3, j] for some j).
    # Actually, A[3, j] has index 3*4 + j = 12 + j for j in 0..3.
    # So we need to check U[12:16, :].
    
    # Let me find which columns are "active" (nonzero) for the first 12 rows of U.
    active_mask = np.any(U[:12, :] != 0, axis=0)
    
    # Keep only the active columns.
    U_red = U[:, active_mask]
    V_red = V[:, active_mask]
    W_red = W[:, active_mask]
    
    # Also, we need to remove the 4th row of C (i=3) from W.
    # C[3, k] has index k*4 + 3 for k in 0..7.
    # So rows 3, 7, 11, 15, 19, 23, 27, 31 of W.
    # Actually, W has shape (p*n, rank) = (32, rank).
    # Row index for C[i, k] is k*n + i = k*4 + i.
    # For i=3: rows 3, 7, 11, 15, 19, 23, 27, 31.
    
    # Keep only rows where i != 3, i.e., i in 0..2.
    # Row index k*4 + i: keep if i in 0..2, i.e., row % 4 != 3.
    keep_rows = [r for r in range(32) if r % 4 != 3]
    W_red = W_red[keep_rows, :]
    
    # Reshape W_red to have shape (p*n_actual, rank) where n_actual=3.
    # The kept rows are in order: for k=0..7, i=0,1,2.
    # So the new row index is k*3 + i.
    W_final = np.zeros((p*3, W_red.shape[1]))
    for new_idx, old_idx in enumerate(keep_rows):
        W_final[new_idx, :] = W_red[old_idx, :]
    
    return U_red, V_red, W_final

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_strassen_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
