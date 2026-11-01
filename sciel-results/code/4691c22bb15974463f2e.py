import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n=2, m=4, p=5):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The tensor T[i,j,k] = delta(i,i) for multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # In the decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, let me re-read the convention.
    # W[k*n+i, r] is the output factor.
    # The tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j] and B[j,k], else 0.
    # Actually the tensor is T[i, j, k] = delta_{i,i}... no.
    # The multiplication tensor has components T[i,j,k] where i is row of A/C, j is col of A / row of B, k is col of B/C.
    # T[i,j,k] = 1 for all i,j,k? No.
    # The structure tensor of matrix multiplication:
    # T = sum_{i,j,k} e_i^A * e_j^B * e_k^C * (delta_{i,i} delta_{j,j} delta_{k,k})... 
    # Actually, T[i,j,k] = 1 if the triple (i,j,k) corresponds to a valid multiplication term.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # Wait no. The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the coefficient is 1.
    # Let me verify: A ⊗ B ⊗ C where the product gives sum T[i,j,k] A[i,j] B[j,k] = C[i,k]
    # So sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k] for all i,k
    # This means T[i,j,k] = 1 for all i,j,k.
    
    # In the decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Note: W is indexed as W[k*n+i, r], so the output index is k*n+i.
    # This means the output C has index (i,k) but stored as k*n+i? Let me re-read.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W's first index is k*n+i, meaning row=k, col=i of the output? That seems unusual.
    # Actually, if C is n x p, then C[i,k] has i in [0,n), k in [0,p).
    # W[k*n+i] suggests k is the "outer" index and i is the "inner" index.
    # So the output is indexed by (k,i) where k is col of C, i is row of C? That's C^T?
    # Or maybe the convention is that the output matrix is stored column-major with (k,i) where
    # k is the row and i is the col? Let me just trust the convention.
    
    # Let me verify: for the schoolbook, U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # Then sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 if (i,j,k) is the matching triple, else 0.
    # This gives T[i,j,k] = 1 for each (i,j,k) pair, which is correct.
    
    # So verification: for all i in [n], j in [m], k in [p]:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    
    rank = U.shape[1]
    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_reconstructed[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    T_expected = np.ones((n, m, p))
    return np.allclose(T_reconstructed, T_expected, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Schoolbook decomposition of rank n*m*p = 40
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Convert to list of lists with integer entries
    U_sb_list = [[int(x) for x in row] for row in U_sb]
    V_sb_list = [[int(x) for x in row] for row in V_sb]
    W_sb_list = [[int(x) for x in row] for row in W_sb]
    
    best_rank = rank_schoolbook
    best_result = {"U": U_sb_list, "V": V_sb_list, "W": W_sb_list}
    
    # Try to find a lower rank decomposition using Strassen-style block decomposition
    # Strategy: split the 2x4x5 multiplication using block matrices
    
    # For 2x4 * 4x5:
    # A is 2x4, B is 4x5, C is 2x5
    # Split B into two 4x2.5 blocks? No, must be integer splits.
    # Split B into B1 (4x2) and B2 (4x3)
    # C = A * B1 + A * B2 (2x2) + (2x3)
    # Rank(A*B1) = 2*4*2 = 16, Rank(A*B2) = 2*4*3 = 24. Total = 40. No improvement.
    
    # Try Strassen on the 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Ai is 1x2 (since n=2, m=4, split m=2+2)
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x5 (since m=4, p=5)
    # Actually let me split differently.
    
    # Let's try: A is 2x4, split as A = [A1 | A2] where A1, A2 are 2x2
    # B is 4x5, split as B = [[B11], [B21]] where B11, B21 are 2x5
    # C = A1*B11 + A2*B21 (2x5)
    # Rank = 2*(2*2*5) = 20. That's the schoolbook for this split.
    # Can we do better? Use Strassen on the "multiplication" A1*B11 + A2*B21
    # This is like multiplying a 1x2 block matrix by a 2x1 block matrix... 
    # Actually it's sum of two 2x2 * 2x5 products.
    # Each 2x2*2x5 has rank 20. Sum has rank at most 40.
    # But we can potentially share. Let M1 = A1*B11, M2 = A2*B21. C = M1 + M2.
    # This doesn't help directly.
    
    # Alternative: Use the fact that n=2 is small.
    # C[i,k] = sum_j A[i,j] B[j,k] for i=0,1 and k=0..4
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    # 
    # Winograd approach: For the 4-dimension (j), use Winograd to reduce multiplications.
    # For n x m * m x p with fixed n and p, the Winograd technique can reduce rank.
    # 
    # Actually, let me try a direct approach: search for a rank-36 decomposition.
    # This is hard to do by search. Let me try known results.
    # 
    # R(2,4,5) - what's the known rank?
    # R(2,2,2) = 7 (Strassen)
    # R(2,4,4) = ? 
    # R(2,n,n) for various n...
    # 
    # Let me try a different approach: use the block Strassen on a 2x2 block structure.
    # Pad A to 2x4, B to 4x5. 
    # Consider splitting into 2x2 blocks in the m dimension:
    # A = [[A00, A01], [A10, A11]] where Aij are 1x2
    # B = [[B00, B01, B02], [B10, B11, B12]] where Bij are 2x1? No...
    
    # Let me think about this more carefully.
    # A is 2x4. Split columns: A = [A_L, A_R] where A_L, A_R are 2x2.
    # B is 4x5. Split rows: B = [B_T; B_B] where B_T, B_B are 2x5.
    # C = A_L * B_T + A_R * B_B (2x5 matrix)
    # 
    # Now, each product A_L * B_T is 2x2 * 2x5. 
    # Rank of 2x2*2x5 multiplication tensor = 2*2*5 = 20 (schoolbook).
    # But can we do better for 2x2 * 2x5?
    # R(2,2,5): We can use Strassen on the 2x2 part and multiply by the 5 columns.
    # Actually, 2x2*2x5 can be seen as 5 independent 2x2*2x1 multiplications? No.
    # C = A_L * B_T where A_L is 2x2, B_T is 2x5, C is 2x5.
    # C[i,k] = sum_j A_L[i,j] B_T[j,k] for i=0,1, k=0..4
    # This is like 5 separate 2x2*2x1 multiplications (each giving a 2x1 result).
    # Rank of 2x2*2x1 = R(2,2,1) = 4 (schoolbook, since it's just 4 multiplications).
    # Total rank if independent = 5*4 = 20.
    # But we can share: use Strassen on the 2x2 matrix multiply for each k.
    # R(2,2,1) = 4, so no Strassen benefit for 1-column.
    # 
    # Alternatively, think of B_T as a 2x5 matrix. The multiplication A_L * B_T can be
    # decomposed using the rank of the 2x2 matrix multiplication (Strassen rank 7)
    # applied to each "column block"? 
    # Actually, A_L * B_T = sum over the 2 columns of A_L times corresponding rows of B_T.
    # Let A_L = [a0, a1] where a0, a1 are column vectors (2x1).
    # B_T = [b0; b1] where b0, b1 are row vectors (1x5).
    # C = a0*b0 + a1*b1 (2x5 matrix, outer products summed)
    # Rank of each outer product a0*b0: this is a rank-1 2x5 matrix.
    # The tensor for a0*b0 has rank 2*1*5 = 10? No, the multiplication a0*b0 is:
    # C[i,k] = a0[i] * b0[k]
    # This is a bilinear map from R^2 x R^5 to R^10 (flattened).
    # The tensor has rank 2*5 = 10? No, it's rank 1 as a 3-tensor!
    # T[i,j,k] = delta(i,i)*delta(j,j)... no.
    # C[i,k] = a[i] * b[k]
    # T[i, j, k] where j indexes... hmm, this is a 2-input tensor.
    # Actually, the outer product a*b is a bilinear map R^2 x R^5 -> R^{2x5}.
    # The tensor T has T[i,k, (i',k')] = delta(i,i')*delta(k,k') for the output.
    # As a 3-tensor (input1, input2, output), T[a, b, c] = c^T (a b^T) = sum_ik c[i,k] a[i] b[k]
    # This is a rank-1 tensor: T = u ⊗ v ⊗ w where u is the identity on the output...
    # Actually T[i, k, (i', k')] = delta_{i,i'} delta_{k,k'}
    # This tensor has rank 10 (one for each (i,k) pair).
    # So a0*b0 has rank 10, a1*b1 has rank 10. Total 20. Same as schoolbook.
    
    # I'm stuck on finding a provably lower rank. Let me just try to verify that
    # a rank-36 decomposition exists by attempting to construct one.
    # 
    # Given the knowledge that rank 36 was achieved, let me try the following approach:
    # Use a 2x2 Strassen block decomposition with some padding.
    # 
    # Actually, let me try: R(2,4,5) <= R(2,4,4) + R(2,4,1) = ?
    # R(2,4,4): Split the 5 into 4+1. C = A*B4 + A*B1 where B4 is 4x4, B1 is 4x1.
    # R(2,4,1) = 8 (schoolbook: 2*4*1).
    # R(2,4,4): Can we do better than 32?
    #   R(2,4,4) <= 2*R(2,2,4) + ... 
    #   Or use Strassen: pad to 2x4 * 4x4. 
    #   Split A into [A1, A2] (2x2 each), B into [B1; B2] (2x4 each).
    #   C = A1*B1 + A2*B2 (2x4)
    #   R(2,2,4) = 16 (schoolbook). So R(2,4,4) <= 32.
    #   Can we do R(2,2,4) better? R(2,2,4) = 2*R(2,2,2) = 14 (using Strassen on 2x2*2x2 for each of 2 col blocks)?
    #   2x2*2x4: C = A*B where A is 2x2, B is 2x4.
    #   Split B into two 2x2 blocks: B = [B1, B2].
    #   C = A*B1 + A*B2 (2x2 + 2x2)
    #   Each A*B1 is 2x2*2x2, rank 7 (Strassen).
    #   So R(2,2,4) <= 7 + 7 = 14.
    #   Therefore R(2,4,4) <= 2*14 = 28.
    #   And R(2,4,5) <= R(2,4,4) + R(2,4,1) = 28 + 8 = 36.
    #
    # So the rank-36 decomposition is:
    # 1. Split B (4x5) into B4 (4x4) and B1 (4x1).
    # 2. For B4: split A (2x4) into A_L (2x2) and A_R (2x2), B4 (4x4) into B4_T (2x4) and B4_B (2x4).
    #    C4 = A_L * B4_T + A_R * B4_B (2x4)
    #    For A_L * B4_T: B4_T is 2x4. Split B4_T into [B4_T1, B4_T2] (2x2 each).
    #      A_L * B4_T1: 2x2*2x2, Strassen rank 7.
    #      A_L * B4_T2: 2x2*2x2, Strassen rank 7.
    #    Similarly for A_R * B4_B: 14 more.
    #    Total for B4: 28.
    # 3. For B1: A * B1 is 2x4*4x1, rank 8 (schoolbook).
    # Total: 28 + 8 = 36.
    
    # Let me implement this!
    
    def strassen_2x2_2x2(A, B, U, V, W, offset):
        """Decompose 2x2 * 2x2 multiplication using Strassen, rank 7.
        A is 2x2, B is 2x2, C = A*B is 2x2.
        Adds 7 columns to U, V, W starting at offset.
        Returns new offset.
        """
        # Strassen:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # In our tensor decomposition, we need to express:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # using 7 "multiplications" (rank-1 terms).
        #
        # Each Strassen multiplication M is a product of a linear combination of A entries
        # and a linear combination of B entries.
        # The output C entries are linear combinations of the M's.
        #
        # For the tensor decomposition, each rank-1 term corresponds to:
        # U entries: coefficients for A[i,j]
        # V entries: coefficients for B[j,k]  
        # W entries: coefficients for C[i,k]
        #
        # The Strassen decomposition gives:
        # M1 = (A00+A11) * (B00+B11)
        #   U: A00=1, A11=1, others 0
        #   V: B00=1, B11=1, others 0
        #   W: C00 += 1, C11 += 1 (from M1 contributing to C00 and C11)
        #   Wait, M1 contributes to C00 with +1 and C11 with +1.
        #   But in the tensor, W[k*n+i, r] is the coefficient for C[i,k] in the r-th term.
        #   So for M1: W[0*2+0] = W[0] += 1 (C00), W[1*2+1] = W[3] += 1 (C11)
        
        # Let me be more careful. The output C is 2x2, indexed C[i,k] where i=0,1 (row), k=0,1 (col).
        # W is indexed as W[k*n+i] = W[k*2+i].
        
        strassen_terms = [
            # (A_coeffs: {i,j: coeff}, B_coeffs: {j,k: coeff}, C_coeffs: {i,k: coeff})
            # M1 = (A00+A11)(B00+B11), contributes to C00 (+1), C11 (+1)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2 = (A10+A11)B00, contributes to C10 (+1), C11 (-1)
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # M3 = A00(B10-B11), contributes to C01 (+1), C11 (+1)
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # M4 = A11(B01-B00), contributes to C00 (+1), C10 (+1)
            ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5 = (A00+A01)B11, contributes to C00 (-1), C01 (+1)
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6 = (A10-A00)(B00+B01), contributes to C11 (+1)
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # M7 = (A01-A11)(B10+B11), contributes to C00 (+1)
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            # A is 2x2, indexed A[i,j] where i=0,1 (row), j=0,1 (col)
            # In our global indexing, A[i,j] maps to U[i*m_total + j_local]
            # But we need to map the local 2x2 indices to global indices.
            # This function is called with an offset into the U, V, W matrices.
            # The caller must provide the mapping from local (i,j) to global row index of U,
            # and local (j,k) to global row index of V, and local (i,k) to global col index of W.
            pass
        
        return offset
    
    # Actually, let me build the full decomposition directly.
    
    # The decomposition has rank 36:
    # - 28 terms for A * B4 (4x4 part)
    # - 8 terms for A * B1 (4x1 part)
    
    # Let me define the full set of rank-1 terms.
    # Each term is (U_row_indices_with_coeffs, V_row_indices_with_coeffs, W_row_indices_with_coeffs)
    
    terms = []
    
    # Part 1: A * B4 where B4 is the first 4 columns of B (4x4)
    # A is 2x4, split into A_L (cols 0-1) and A_R (cols 2-3), each 2x2.
    # B4 is 4x4, split into B4_T (rows 0-1) and B4_B (rows 2-3), each 2x4.
    # C4 = A_L * B4_T + A_R * B4_B (2x4)
    
    # For A_L * B4_T: A_L is 2x2 (A[i, 0:2]), B4_T is 2x4 (B[0:2, 0:4])
    # Split B4_T into two 2x2 blocks: B4_T1 (B[0:2, 0:2]) and B4_T2 (B[0:2, 2:4])
    # A_L * B4_T1: 2x2 * 2x2 -> contributes to C[0:2, 0:2]
    # A_L * B4_T2: 2x2 * 2x2 -> contributes to C[0:2, 2:4]
    
    # Strassen for A_L * B4_T1:
    # A_L[i,j] where i=0,1 (global row of A), j=0,1 (global col 0,1 of A)
    # B4_T1[j,k] where j=0,1 (global row 0,1 of B), k=0,1 (global col 0,1 of B)
    # C[i,k] where i=0,1 (global row of C), k=0,1 (global col 0,1 of C)
    
    strassen_222 = [
        # (A_coeffs, B_coeffs, C_coeffs) - local indices (i,j) for A, (j,k) for B, (i,k) for C
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),      # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),                 # M2
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),                 # M3
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),                 # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),                 # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),                 # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),                 # M7
    ]
    
    # A_L * B4_T1: A global (i, j) where i=0,1, j=0,1
    #               B global (j, k) where j=0,1, k=0,1
    #               C global (i, k) where i=0,1, k=0,1
    for a_c, b_c, c_c in strassen_222:
        U_idx = {i * m + j: coeff for (i,j), coeff in a_c.items()}
        V_idx = {j * p + k: coeff for (j,k), coeff in b_c.items()}
        W_idx = {k * n + i: coeff for (i,k), coeff in c_c.items()}
        terms.append((U_idx, V_idx, W_idx))
    
    # A_L * B4_T2: A global (i, j) where i=0,1, j=0,1 (same A_L)
    #               B global (j, k) where j=0,1, k=2,3 (B[0:2, 2:4])
    #               C global (i, k) where i=0,1, k=2,3
    for a_c, b_c, c_c in strassen_222:
        U_idx = {i * m + j: coeff for (i,j), coeff in a_c.items()}
        V_idx = {j * p + (k+2): coeff for (j,k), coeff in b_c.items()}
        W_idx = {(k+2) * n + i: coeff for (i,k), coeff in c_c.items()}
        terms.append((U_idx, V_idx, W_idx))
    
    # A_R * B4_B: A global (i, j) where i=0,1, j=2,3 (A[0:2, 2:4])
    #               B global (j, k) where j=2,3, k=0,1 (B[2:4, 0:2])
    #               C global (i, k) where i=0,1, k=0,1
    for a_c, b_c, c_c in strassen_222:
        U_idx = {i * m + (j+2): coeff for (i,j), coeff in a_c.items()}
        V_idx = {(j+2) * p + k: coeff for (j,k), coeff in b_c.items()}
        W_idx = {k * n + i: coeff for (i,k), coeff in c_c.items()}
        terms.append((U_idx, V_idx, W_idx))
    
    # A_R * B4_B2: A global (i, j) where i=0,1, j=2,3
    #               B global (j, k) where j=2,3, k=2,3 (B[2:4, 2:4])
    #               C global (i, k) where i=0,1, k=2,3
    for a_c, b_c, c_c in strassen_222:
        U_idx = {i * m + (j+2): coeff for (i,j), coeff in a_c.items()}
        V_idx = {(j+2) * p + (k+2): coeff for (j,k), coeff in b_c.items()}
        W_idx = {(k+2) * n + i: coeff for (i,k), coeff in c_c.items()}
        terms.append((U_idx, V_idx, W_idx))
    
    # Part 2: A * B1 where B1 is the last column of B (4x1)
    # A is 2x4, B1 is 4x1, C1 is 2x1
    # C1[i, 0] = sum_j A[i, j] B1[j, 0]
    # Schoolbook: 8 terms
    for i in range(n):
        for j in range(m):
            U_idx = {i * m + j: 1}
            V_idx = {j * p + 4: 1}  # B[j, 4]
            W_idx = {4 * n + i: 1}  # C[i, 4] -> W[4*2+i]
            terms.append((U_idx, V_idx, W_idx))
    
    # Total terms: 28 + 8 = 36
    assert len(terms) == 36, f"Expected 36 terms, got {len(terms)}"
    
    rank_36 = len(terms)
    U_36 = np.zeros((n * m, rank_36))
    V_36 = np.zeros((m * p, rank_36))
    W_36 = np.zeros((p * n, rank_36))
    
    for r, (U_idx, V_idx, W_idx) in enumerate(terms):
        for idx, coeff in U_idx.items():
            U_36[idx, r] = coeff
        for idx, coeff in V_idx.items():
            V_36[idx, r] = coeff
        for idx, coeff in W_idx.items():
            W_36[idx, r] = coeff
    
    # Verify
    if _verify(U_36, V_36, W_36, n, m, p):
        best_rank = rank_36
        U_36_list = [[int(x) for x in row] for row in U_36]
        V_36_list = [[int(x) for x in row] for row in V_36]
        W_36_list = [[int(x) for x in row] for row in W_36]
        best_result = {"U": U_36_list, "V": V_36_list, "W": W_36_list}
    else:
        # Fallback to schoolbook
        pass
    
    return best_result

# EVOLVE-BLOCK-END
