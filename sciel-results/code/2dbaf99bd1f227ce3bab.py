DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm on 2x2 blocks of 3x3 matrices.
    # Rank = 7 * (3*3*3) = 189.
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for 2x2 block matrices.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # Each block is 3x3.
    # The forms are linear combinations of the blocks.
    # We represent the coefficients for each 3x3 block.
    # Block indices: 0=(11), 1=(12), 2=(21), 3=(22).
    
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For each Strassen term, we define the coefficients for A blocks and B blocks.
    # coeffs_A: list of 4 matrices (3x3) or just the block multipliers.
    # Since the blocks are multiplied as whole 3x3 matrices in the "inner" multiplication,
    # and we are using schoolbook for the 3x3 multiplication,
    # we can treat the "inner" multiplication as:
    # C_block = sum_{i,j,k} A_block[i,j] B_block[j,k]
    # So if A_block = sum c_b A_b, B_block = sum d_b B_b,
    # then C_block = sum_{i,j,k} (sum c_b A_b[i,j]) (sum d_b B_b[j,k])
    # = sum_{i,j,k} sum_b sum_b' c_b d_b' A_b[i,j] B_b'[j,k]
    # This expands to 27 terms per pair of blocks.
    
    # However, it is easier to just define the linear forms on the individual elements.
    # Let's define the 7 sets of coefficients for A and B elements.
    
    # A elements: A[0][0]..A[2][2] for block 11, etc.
    # We map global indices to local block indices.
    # Global i, j in 0..5.
    # Block row bi = i // 3, local row li = i % 3.
    # Block col bj = j // 3, local col lj = j % 3.
    
    # Strassen terms definition:
    # term: (coeffs_A, coeffs_B)
    # coeffs_A is a dict: {(bi, bj): multiplier}
    # coeffs_B is a dict: {(bi, bj): multiplier}
    
    terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    ]
    
    col = 0
    for coeffs_A, coeffs_B in terms:
        # Iterate over all local indices i, j, k for the 3x3 blocks
        for li in range(3):
            for lj in range(3):
                for lk in range(3):
                    # Determine which A block contributes to A[li, lj]
                    # A[li, lj] is in block (bi, bj) where bi = ?
                    # Wait, the linear form is on the blocks.
                    # A_block = sum c_b A_b.
                    # The element A_block[li, lj] = sum c_b A_b[li, lj].
                    # So for a specific local index (li, lj), the coefficient is the sum of c_b for all blocks b that contain this local index?
                    # No, each block is a distinct 3x3 matrix.
                    # A_11[li, lj] is a specific element.
                    # So A_block[li, lj] = c_11 * A_11[li, lj] + c_12 * A_12[li, lj] + ...
                    # So the coefficient for the global element A[bi*3+li, bj*3+lj] is c_{bi,bj}.
                    
                    # We need to sum the contributions from all blocks to the linear form at (li, lj).
                    # But in the schoolbook expansion, we treat each global element independently.
                    # The term is A_global[i,j] * B_global[j,k] * C_global[i,k].
                    # The coefficient for this term in the decomposition is the product of the coefficients of A_global[i,j] and B_global[j,k] in the linear forms.
                    
                    # Let's find the coefficient for A_global[i,j] in the current Strassen term's A-form.
                    # i = bi*3 + li, j = bj*3 + lj.
                    # coeff_A = coeffs_A.get((bi, bj), 0)
                    
                    # Similarly for B_global[j,k].
                    # j = bj*3 + lj, k = bk*3 + lk.
                    # coeff_B = coeffs_B.get((bj, bk), 0)
                    
                    # But wait, the loop is over li, lj, lk.
                    # We need to iterate over global i, j, k.
                    # Let's change the loop structure.
                    pass
                    
    # Re-doing the loop structure for clarity
    col = 0
    for coeffs_A, coeffs_B in terms:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    bi = i // 3
                    bj = j // 3
                    bk = k // 3
                    
                    cA = coeffs_A.get((bi, bj), 0)
                    cB = coeffs_B.get((bj, bk), 0)
                    
                    if cA != 0 and cB != 0:
                        U[i * m + j][col] = cA
                        V[j * p + k][col] = cB
                        W[k * n + i][col] = 1
                        col += 1
                        
    # The rank is col, which should be 189.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
