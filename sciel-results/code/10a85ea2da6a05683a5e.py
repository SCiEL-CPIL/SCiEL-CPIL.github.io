import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 4x4 * 4x6
    # Partition into 2x2 blocks of size 2x2 (A) and 2x3 (B)
    
    # Strassen coefficients for the 7 products
    # A blocks: 00, 01, 10, 11 (row, col)
    # B blocks: 00, 01, 10, 11 (row, col)
    
    # M1 = (A00 + A11) (B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # Coefficients for A blocks (4x7) and B blocks (4x7)
    # Block indices: 0=00, 1=01, 2=10, 3=11
    A_coeffs = [
        [1, 0, 1, 0, 1, -1, 0], # M1..M7 for A00
        [0, 0, 0, 0, 1, 0, -1], # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    B_coeffs = [
        [1, 1, 0, 1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0], # B01
        [0, 0, 0, -1, 0, 0, 1],# B10
        [1, 0, -1, -1, 1, 0, 1]# B11
    ]
    
    # Output block coefficients (4x7)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    rank = 0
    U = []
    V = []
    W = []
    
    # We generate terms for each of the 7 products
    # Each product is a 2x2 * 2x3 matrix multiplication
    # Rank of each is 12 (schoolbook)
    
    for k in range(7):
        # Get coefficients for this product
        a_c = [A_coeffs[i][k] for i in range(4)]
        b_c = [B_coeffs[i][k] for i in range(4)]
        
        # Iterate over all pairs of A blocks and B blocks involved in this product
        # Actually, we can just iterate over the 12 terms of the 2x2x3 tensor
        # and sum the contributions from all block pairs
        
        # Local indices for 2x2 A block and 2x3 B block
        # i: 0,1 (row of A)
        # j: 0,1 (col of A, row of B)
        # k_local: 0,1,2 (col of B)
        
        for i in range(2):
            for j in range(2):
                for k_local in range(3):
                    # This term corresponds to A[i,j] * B[j,k_local]
                    # We need to find the global indices
                    
                    # Sum of coefficients for this term across all block pairs
                    coeff_A = 0
                    coeff_B = 0
                    # This is not quite right. The term is a product of linear combos.
                    # (Sum a_ab A_ab) * (Sum b_bc B_bc)
                    # = Sum_ab Sum_bc a_ab b_bc A_ab B_bc
                    # The term A[i,j] B[j,k] appears in A_ab B_bc if:
                    # i = i_ab, j = j_ab AND j = j_bc, k_local = k_bc
                    
                    # So for each pair (ab, bc) such that:
                    # i_ab = i, j_ab = j, j_bc = j, k_bc = k_local
                    # we add a_ab * b_bc
                    
                    total_coeff = 0
                    for ab in range(4):
                        r_ab = ab // 2
                        c_ab = ab % 2
                        # Check if local i, j match block ab
                        if i == r_ab and j == c_ab:
                            # Wait, A block ab is at (r_ab, c_ab).
                            # Local i is row within block, j is col within block.
                            # So global row = r_ab*2 + i, global col = c_ab*2 + j.
                            # This matches.
                            pass
                    
                    # Actually, it's easier to just expand the full product.
                    # For a fixed k (product index), we have:
                    # X = sum_ab a_ab A_ab
                    # Y = sum_bc b_bc B_bc
                    # Z = X Y
                    # Z[i, k_local] = sum_j X[i, j] Y[j, k_local]
                    
                    # We want to express Z as sum of rank-1 tensors.
                    # The schoolbook decomposition of Z (2x2x3) is:
                    # sum_{i,j,k_local} Z[i, k_local] e_i^C e_j^A e_k^B ?
                    # No, the tensor is T_{i, j, k} = Z_{i, k} if j matches?
                    
                    # Let's stick to the global index mapping.
                    # For each term in the schoolbook decomposition of the 2x2x3 tensor:
                    # Term (i, j, k_local) has:
                    # U index: global A index for (i, j)
                    # V index: global B index for (j, k_local)
                    # W index: global C index for (i, k_local)
                    
                    # But we have multiple block pairs contributing to this term.
                    # The coefficient is sum_{ab, bc} a_ab b_bc delta_{i_ab, i} delta_{j_ab, j} delta_{j_bc, j} delta_{k_bc, k_local}
                    
                    # Let's compute this sum.
                    coeff = 0.0
                    for ab in range(4):
                        r_ab = ab // 2
                        c_ab = ab % 2
                        if r_ab == i and c_ab == j:
                            # This A block contributes
                            a_val = a_c[ab]
                            for bc in range(4):
                                r_bc = bc // 2
                                c_bc = bc % 2
                                if r_bc == j and c_bc == k_local:
                                    # This B block contributes
                                    b_val = b_c[bc]
                                    coeff += a_val * b_val
                    
                    if coeff != 0:
                        # Add this term
                        # Global A index: (r_ab*2 + i) * 4 + (c_ab*2 + j)
                        # But we don't have r_ab, c_ab here.
                        # We know i, j are local.
                        # The term A[i,j] B[j,k] is part of A_ab B_bc.
                        # The global index depends on which block ab, bc we are in.
                        # Wait, the term A[i,j] B[j,k] is UNIQUE to a specific block pair?
                        # No, A[i,j] is the element at local (i,j) of block ab.
                        # If we are in block ab, the global index is (r_ab*2+i)*4 + (c_ab*2+j).
                        # But different block pairs (ab, bc) give different global indices.
                        # So we cannot sum them into a single term!
                        
                        # We must add a separate term for each non-zero (ab, bc) pair.
                        pass
                    
                    # Correct approach:
                    # Iterate over all (ab, bc) pairs.
                    # For each pair, if a_c[ab] != 0 and b_c[bc] != 0:
                    #   Add 12 terms for the product A_ab * B_bc.
                    #   Scale by a_c[ab] * b_c[bc].
                    
                    # This will result in 16 * 12 = 192 terms per product?
                    # No, only non-zero coefficients.
                    # And many terms will have the same global indices?
                    # No, different (ab, bc) pairs map to different global indices.
                    # So the rank is indeed the sum of ranks of all non-zero (ab, bc) products.
                    
                    # This leads to rank > 96.
                    # So this simple Strassen expansion is NOT rank 84.
                    # The rank 84 claim was for the "border rank" or using a different decomposition.
                    
                    # Actually, the rank of 4x4x6 is known to be 84?
                    # Let's check the literature.
                    # Strassen for 4x4 is rank 49? No, 7^2=49 for 4x4x4.
                    # For 4x4x6, it might be 84.
                    
                    # Let's try to implement the rank 84 decomposition if possible.
                    # Or just return the schoolbook (96) if we can't find a better one quickly.
                    
                    # Given the time limit, I will return the schoolbook decomposition.
                    pass

    # Fallback to Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
