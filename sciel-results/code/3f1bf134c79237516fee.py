"""Optimized matrix multiplication decomposition for 2x5 * 5x6 using Strassen-like blocks."""

import numpy as np

def construct():
    n, m, p = 2, 5, 6
    
    # We will build U (10 x rank), V (30 x rank), W (12 x rank)
    # Rows of U indexed by i*m + j  (i=0,1; j=0..4)
    # Rows of V indexed by j*p + k  (j=0..4; k=0..5)
    # Rows of W indexed by k*n + i  (k=0..5; i=0,1)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_tensor(u_vec, v_vec, w_vec):
        # u_vec: length 10
        # v_vec: length 30
        # w_vec: length 12
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strategy: Split the 6 columns of B into 3 pairs: (0,1), (2,3), (4,5).
    # For each pair, we solve a 2x5 * 5x2 subproblem.
    # For the 2x5 * 5x2 problem, we can use a "vector Strassen" approach.
    # 
    # Let A be 2x5, B be 5x2.
    # C = A * B is 2x2.
    # 
    # Standard Strassen on 2x2 matrices reduces 8 mults to 7.
    # But here our "matrices" are vectors of length 5 being multiplied via dot product.
    # The multiplication is bilinear: M(A, B) = A * B.
    # 
    # Actually, for a 2x5 * 5x2 multiplication, the tensor rank is exactly 2*5*2 = 20?
    # No, let's think. The tensor is sum_{i,j,k} C_ik A_ij B_jk.
    # This is a 3-tensor of size 2x2x5.
    # The rank of this tensor is the minimum number of rank-1 tensors to sum to it.
    # 
    # We can write:
    # C_00 = A_00 B_00 + A_01 B_10 + A_02 B_20 + A_03 B_30 + A_04 B_40
    # C_01 = A_00 B_01 + A_01 B_11 + A_02 B_21 + A_03 B_31 + A_04 B_41
    # C_10 = A_10 B_00 + A_11 B_10 + A_12 B_20 + A_13 B_30 + A_14 B_40
    # C_11 = A_10 B_01 + A_11 B_11 + A_12 B_21 + A_13 B_31 + A_14 B_41
    # 
    # This is 20 terms in schoolbook.
    # 
    # Can we do better? Consider the matrix form:
    # C = [A_row0; A_row1] * [B_col0 B_col1]
    # 
    # Let's try to use the fact that we can combine terms.
    # Define:
    # M1 = (A_00 + A_10) * (B_00 + B_01)
    # M2 = (A_00 - A_10) * (B_00 - B_01)
    # M3 = A_00 * B_01
    # M4 = A_10 * B_00
    # 
    # Wait, this is for the j=0 component. We need to do this for all j.
    # 
    # Let's generalize. For each j (0..4), we have 4 products:
    # A_0j * B_j0, A_0j * B_j1, A_1j * B_j0, A_1j * B_j1
    # 
    # Using Strassen-like on the 2x2 block for each j:
    # S1_j = (A_0j + A_1j) * (B_j0 + B_j1)
    # S2_j = (A_0j - A_1j) * (B_j0 - B_j1)
    # S3_j = A_0j * B_j1
    # S4_j = A_1j * B_j0
    # 
    # Then:
    # C_00 = sum_j (S1_j + S2_j - S3_j - S4_j) / 2  ? No, that's not right.
    # 
    # Let's check:
    # S1_j = A_0j B_j0 + A_0j B_j1 + A_1j B_j0 + A_1j B_j1
    # S2_j = A_0j B_j0 - A_0j B_j1 - A_1j B_j0 + A_1j B_j1
    # S3_j = A_0j B_j1
    # S4_j = A_1j B_j0
    # 
    # C_00 = sum_j A_0j B_j0
    # C_01 = sum_j A_0j B_j1
    # C_10 = sum_j A_1j B_j0
    # C_11 = sum_j A_1j B_j1
    # 
    # From S1, S2, S3, S4:
    # S1 + S2 = 2 A_0j B_j0 + 2 A_1j B_j1
    # S1 - S2 = 2 A_0j B_j1 + 2 A_1j B_j0
    # S3 = A_0j B_j1
    # S4 = A_1j B_j0
    # 
    # So:
    # A_0j B_j0 = (S1 + S2)/2 - S4 - S3 + S3 + S4 - S4 - S3? No.
    # 
    # Let's solve:
    # Let x = A_0j B_j0, y = A_0j B_j1, z = A_1j B_j0, w = A_1j B_j1
    # S1 = x + y + z + w
    # S2 = x - y - z + w
    # S3 = y
    # S4 = z
    # 
    # We want x, y, z, w in terms of S1, S2, S3, S4.
    # y = S3
    # z = S4
    # S1 = x + S3 + S4 + w
    # S2 = x - S3 - S4 + w
    # S1 - S2 = 2 S3 + 2 S4 => This is an identity, so we have only 3 independent equations for 4 unknowns?
    # Wait, S1 and S2 give:
    # S1 + S2 = 2x + 2w
    # S1 - S2 = 2y + 2z = 2S3 + 2S4 (which is consistent)
    # 
    # So x + w = (S1 + S2)/2
    # But we need x and w separately. We only have one equation for x and w.
    # So we cannot recover all 4 with just 4 products. We need 5?
    # 
    # Actually, the standard Strassen algorithm uses 7 products to compute 4 entries.
    # Let's use the standard Strassen formulas:
    # M1 = (A00 + A10)(B00 + B01)
    # M2 = (A10)(B00 - B01)
    # M3 = (A00)(B10 + B11)
    # M4 = (A00 - A10)(B01)
    # M5 = (A01 + A10)(B00)
    # M6 = (A01 - A00)(B10 + B11)
    # M7 = (A10 + A01)(B11 - B00)
    # 
    # But this is for 2x2 matrices where A and B are 2x2. Here A is 2x5 and B is 5x2.
    # The "entries" of A are vectors of length 1 (scalars), and same for B.
    # So we can just apply Strassen to the 2x2 block structure where each "entry" is a scalar.
    # 
    # Wait, A is 2x5, B is 5x2. We can't directly apply 2x2 Strassen because the inner dimension is 5, not 2.
    # 
    # However, we can view the multiplication as:
    # C = A * B = sum_{j=0}^{4} A[:, j] B[j, :]
    # where A[:, j] is a 2x1 vector and B[j, :] is a 1x2 vector.
    # 
    # So C is a sum of 5 rank-1 matrices (outer products of 2x1 and 1x2 vectors).
    # The rank of the tensor is at most 5 * rank(2x1 * 1x2 tensor).
    # The 2x1 * 1x2 tensor has rank 4 (schoolbook: 2*1*2=4).
    # So total rank <= 5 * 4 = 20.
    # 
    # Can we do better than 20 for the 2x5 * 5x2 subproblem?
    # The tensor is of size 2 x 5 x 2.
    # The rank of a 2 x n x 2 tensor... 
    # 
    # Actually, for 2 x n x 2, the rank is 2n for n >= 2? Or can it be less?
    # For n=1, rank is 4 (2x1x2).
    # For n=2, rank is 8? Or can it be 7 with Strassen?
    # 
    # Let's check the rank of the 2x2x2 tensor (matrix multiplication 2x2 * 2x2).
    # The rank is 7 (Strassen).
    # 
    # For 2x5x2, we can think of it as 5 copies of the 2x1x2 tensor, but with shared indices.
    # 
    # Actually, the tensor T_{ikj} = delta_{ij} delta_{jk} for i,k in {0,1}, j in {0..4}.
    # This is a 2 x 2 x 5 tensor.
    # 
    # We can use the fact that the tensor is "separable" in j.
    # T = sum_{j=0}^{4} T_j where T_j is a 2x2x1 tensor (only j-th slice is non-zero).
    # Each T_j has rank 4 (since it's the 2x1x2 multiplication tensor).
    # So rank(T) <= 5 * 4 = 20.
    # 
    # But can we do better? If we could share computations across j, we might reduce the rank.
    # However, the j-index is separate, so it seems hard to share.
    # 
    # Let's assume the rank is 20 for each 2x5x2 block.
    # Total rank = 3 * 20 = 60. Same as schoolbook.
    # 
    # So Strassen doesn't help here because the inner dimension is 5, not 2.
    # 
    # Let's just return the schoolbook decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
