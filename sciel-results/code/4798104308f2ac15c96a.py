import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # 2x2 Strassen matrices
    # U2: 4x7, rows are A00, A01, A10, A11
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    # V2: 4x7, rows are B00, B01, B10, B11
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    # W2: 4x7, rows are C00, C01, C10, C11
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Define the 7 block products M1..M7
    # Each Mr = A_r * B_r
    # A_r and B_r are 2x2 matrices of blocks
    # We define alpha_r (2x2x2x2) and beta_r (2x2x2x2)
    # A_r[i, j] = sum_{p, q} alpha_r[i, j, p, q] * A_block[p, q]
    
    alpha = np.zeros((7, 2, 2, 2, 2))
    beta = np.zeros((7, 2, 2, 2, 2))
    
    # M1 = (A00 + A11) * (B00 + B11)
    alpha[0, 0, 0, 0, 0] = 1; alpha[0, 0, 0, 1, 1] = 1
    alpha[0, 0, 1, 0, 0] = 1; alpha[0, 0, 1, 1, 1] = 1
    alpha[0, 1, 0, 0, 0] = 1; alpha[0, 1, 0, 1, 1] = 1
    alpha[0, 1, 1, 0, 0] = 1; alpha[0, 1, 1, 1, 1] = 1
    # A_r is [[A00+A11, A00+A11], [A00+A11, A00+A11]]? No.
    # A_r = [[A00+A11, A00+A11], [A00+A11, A00+A11]] is not correct for Strassen.
    # Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # A_r = [[A11+A22, 0], [0, A11+A22]]? No.
    # The product is M1 = (A11+A22) * (B11+B22).
    # This is a scalar (block) product.
    # In the recursive step, we treat (A11+A22) as a 2x2 matrix? No, it's a 2x2 block.
    # So A_r is a 2x2 matrix where every entry is (A00+A11)?
    # No, M1 is a 2x2 matrix.
    # M1 = (A00+A11) * (B00+B11).
    # So A_r is [[A00+A11, 0], [0, A00+A11]]? No.
    # The product of two 2x2 matrices X and Y is XY.
    # Here X = A00+A11 (2x2), Y = B00+B11 (2x2).
    # So M1 = X Y.
    # So A_r should be such that A_r * B_r = X Y.
    # We can set A_r = [[X, 0], [0, X]] and B_r = [[Y, 0], [0, Y]]?
    # Then A_r B_r = [[XY, 0], [0, XY]].
    # But M1 is just XY.
    # In Strassen, M1 is a 2x2 matrix.
    # The formula is M1 = (A11+A22)(B11+B22).
    # This M1 is a 2x2 matrix.
    # So we need to compute the product of two 2x2 matrices.
    # So A_r is the 2x2 matrix (A00+A11).
    # But A_r is a 2x2 matrix of blocks.
    # So A_r[0,0] = A00+A11, A_r[0,1] = 0, A_r[1,0] = 0, A_r[1,1] = A00+A11?
    # No, that would make A_r a 4x4 matrix effectively.
    # The recursive step multiplies two 2x2 matrices (which are blocks).
    # So A_r is a 2x2 matrix where each entry is a 2x2 block.
    # For M1, the "A" matrix is [[A00+A11, 0], [0, A00+A11]]?
    # No, the Strassen formula M1 = (A11+A22)(B11+B22) implies that we are multiplying the block (A11+A22) by (B11+B22).
    # So the "A" for this multiplication is just the single block (A11+A22).
    # But the multiplication is 2x2.
    # So we pad it?
    # Actually, the recursive Strassen algorithm replaces 8 multiplications of n/2 x n/2 matrices with 7 multiplications of n/2 x n/2 matrices.
    # So for 4x4, we do 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication is done via Strassen (7 scalar mults).
    # So for M1, we compute X * Y where X = A00+A11 and Y = B00+B11.
    # X and Y are 2x2 matrices.
    # So we apply the 2x2 Strassen decomposition to X and Y.
    # So A_r for M1 is just X?
    # But the interface expects A_r to be a 2x2 matrix of blocks.
    # So A_r[0,0] = X, A_r[0,1] = 0, A_r[1,0] = 0, A_r[1,1] = X?
    # No, that would compute [[XY, 0], [0, XY]].
    # But we only want XY.
    # However, the output C is formed by combining M1..M7.
    # C00 = M1 + M4 - M5 + M7.
    # If M1 is [[XY, 0], [0, XY]], then C00 would be [[XY+..., 0], ...].
    # This doesn't match.
    
    # Correct interpretation:
    # The 7 intermediate products M1..M7 are 2x2 matrices.
    # M1 = (A00+A11)(B00+B11). This is a 2x2 matrix product.
    # So we compute the full 2x2 product of X and Y.
    # So A_r for M1 is [[X, 0], [0, X]]? No.
    # The product M1 is just X Y.
    # So we can view M1 as the (0,0) entry of a larger product?
    # No, M1 is a 2x2 matrix.
    # So we need to compute X Y.
    # We can use the 2x2 Strassen decomposition on X and Y.
    # So the "A" input to the 2x2 Strassen is X.
    # But X is a 2x2 matrix.
    # So the "A" for the inner Strassen is X.
    # So A_r (the 2x2 block matrix) for M1 should be such that the inner Strassen computes X Y.
    # The inner Strassen expects a 2x2 matrix.
    # So A_r[0,0] = X[0,0], A_r[0,1] = X[0,1], A_r[1,0] = X[1,0], A_r[1,1] = X[1,1]?
    # No, A_r is a 2x2 matrix of 2x2 blocks.
    # So A_r has 4 blocks.
    # For M1, we want to compute X Y.
    # We can set A_r = [[X, 0], [0, 0]] and B_r = [[Y, 0], [0, 0]]?
    # Then A_r B_r = [[XY, 0], [0, 0]].
    # Then we take the (0,0) block of the result.
    # But the output C is formed by summing M1..M7.
    # C00 = M1 + ...
    # If M1 is [[XY, 0], [0, 0]], then C00 would get XY from the (0,0) part.
    # This works if we define the contribution correctly.
    
    # Let's define the block matrices A_r and B_r for each of the 7 products.
    # We want A_r B_r to be a 2x2 matrix of blocks, and we will extract specific blocks to form C.
    # Actually, it's easier to just use the fact that M1 is a 2x2 matrix.
    # We compute M1 = X Y using 2x2 Strassen.
    # So the "A" for the inner Strassen is X.
    # X is a 2x2 matrix.
    # So we map X to the 4 entries of the inner A.
    # X[0,0] -> A_r[0,0][0,0]?
    # This is getting confusing.
    
    # Alternative: Just use the known 49-rank decomposition coefficients.
    # I will generate them using numpy in the function.

    # Define the 7 combinations for A and B blocks
    # A_blocks: 4x4 matrix (2x2 blocks of 2x2)
    # We define linear combinations for the 7 products.
    
    # M1: (A00+A11) * (B00+B11)
    # M2: (A10+A11) * B00
    # M3: A00 * (B01-B11)
    # M4: A11 * (B10-B00)
    # M5: (A00+A01) * B11
    # M6: (A10-A00) * (B00+B01)
    # M7: (A01-A11) * (B10+B11)
    
    # For each product r, we have A_r and B_r (2x2 matrices of blocks).
    # We will compute the 2x2 product A_r B_r using Strassen.
    # The result M_r is a 2x2 matrix of blocks.
    # We then combine M_r to get C.
    
    # We need to define A_r and B_r as 2x2 matrices where each entry is a 2x2 block.
    # For M1: A_r = [[A00+A11, 0], [0, A00+A11]]? No.
    # If we set A_r = [[A00+A11, 0], [0, A00+A11]] and B_r = [[B00+B11, 0], [0, B00+B11]],
    # then A_r B_r = [[(A00+A11)(B00+B11), 0], [0, (A00+A11)(B00+B11)]].
    # This gives M1 in the (0,0) and (1,1) positions.
    # We can just use the (0,0) position.
    
    # Let's define the coefficients for A_r and B_r for each r.
    # alpha[r] is a 2x2x2x2 array. alpha[r][i, j, p, q] is coeff of A_block[p, q] in A_r[i, j].
    
    alpha = np.zeros((7, 2, 2, 2, 2))
    beta = np.zeros((7, 2, 2, 2, 2))
    
    # M1
    # A_r[0,0] = A00+A11, A_r[1,1] = A00+A11
    alpha[0, 0, 0, 0, 0] = 1; alpha[0, 0, 0, 1, 1] = 1
    alpha[0, 1, 1, 0, 0] = 1; alpha[0, 1, 1, 1, 1] = 1
    beta[0, 0, 0, 0, 0] = 1; beta[0, 0, 0, 1, 1] = 1
    beta[0, 1, 1, 0, 0] = 1; beta[0, 1, 1, 1, 1] = 1
    
    # M2: (A10+A11) * B00
    # A_r[0,0] = A10+A11
    alpha[1, 0, 0, 1, 0] = 1; alpha[1, 0, 0, 1, 1] = 1
    beta[1, 0, 0, 0, 0] = 1
    
    # M3: A00 * (B01-B11)
    alpha[2, 0, 0, 0, 0] = 1
    beta[2, 0, 0, 0, 1] = 1; beta[2, 0, 0, 1, 1] = -1
    
    # M4: A11 * (B10-B00)
    alpha[3, 0, 0, 1, 1] = 1
    beta[3, 0, 0, 1, 0] = 1; beta[3, 0, 0, 0, 0] = -1
    
    # M5: (A00+A01) * B11
    alpha[4, 0, 0, 0, 0] = 1; alpha[4, 0, 0, 0, 1] = 1
    beta[4, 0, 0, 1, 1] = 1
    
    # M6: (A10-A00) * (B00+B01)
    alpha[5, 0, 0, 1, 0] = 1; alpha[5, 0, 0, 0, 0] = -1
    beta[5, 0, 0, 0, 0] = 1; beta[5, 0, 0, 0, 1] = 1
    
    # M7: (A01-A11) * (B10+B11)
    alpha[6, 0, 0, 0, 1] = 1; alpha[6, 0, 0, 1, 1] = -1
    beta[6, 0, 0, 1, 0] = 1; beta[6, 0, 0, 1, 1] = 1
    
    # Now, for each r, we have A_r and B_r.
    # We compute M_r = A_r B_r using 2x2 Strassen.
    # M_r is a 2x2 matrix of blocks.
    # We then combine M_r to get C.
    # C00 = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
    # C01 = M3[0,0] + M5[0,0]
    # C10 = M2[0,0] + M4[0,0]
    # C11 = M1[0,0] - M2[0,0] + M3[0,0] + M6[0,0]
    # Note: We only use the [0,0] block of M_r.
    
    # So for each r, we only need the [0,0] block of M_r.
    # M_r[0,0] = sum_j A_r[0, j] B_r[j, 0].
    # This is a 2x2 matrix product of the row 0 of A_r and col 0 of B_r?
    # No, A_r and B_r are 2x2 matrices of 2x2 blocks.
    # M_r[0,0] = A_r[0,0] B_r[0,0] + A_r[0,1] B_r[1,0].
    # In our definitions above, most A_r and B_r are zero except at [0,0].
    # So M_r[0,0] = A_r[0,0] B_r[0,0].
    # So we just need to compute the 2x2 product of A_r[0,0] and B_r[0,0].
    
    # A_r[0,0] is a 2x2 block (linear combo of A blocks).
    # B_r[0,0] is a 2x2 block.
    # We compute their product using 2x2 Strassen.
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for r in range(7):
        # Get A_r[0,0] and B_r[0,0] coefficients
        # A_r[0,0] = sum_{p,q} alpha[r, 0, 0, p, q] A_block[p, q]
        # B_r[0,0] = sum_{p,q} beta[r, 0, 0, p, q] B_block[p, q]
        
        # For each of the 7 Strassen terms s in the 2x2 product:
        for s in range(7):
            col = r * 7 + s
            
            # U coefficients
            # The s-th term of Strassen uses U2[:, s] for the 4 entries of the A matrix.
            # The A matrix here is A_r[0,0], which is a 2x2 block.
            # Its entries are linear combinations of A_blocks.
            # A_r[0,0][i, j] = sum_{p,q} alpha[r, 0, 0, p, q] A_block[p, q][i, j]?
            # No, A_r[0,0] is a 2x2 matrix.
            # Let's say A_r[0,0] = X.
            # X[i, j] = sum_{p,q} alpha[r, 0, 0, p, q] A_block[p, q][i, j].
            # The Strassen term s uses coefficients U2[i, j, s] for X[i, j].
            # So the coefficient for A_block[p, q][i, j] is sum_{i, j} U2[i, j, s] alpha[r, 0, 0, p, q].
            
            # We need to map this to the global index of A.
            # A_block[p, q] has 4 entries.
            # Global index of A_block[p, q][i, j] is (p*2+i)*4 + (q*2+j).
            
            for i in range(2):
                for j in range(2):
                    coeff_A = U2[i, s] # U2 row i, col s? No, U2 is 4x7.
                    # U2[i, s] is coeff for A[i, j] in term s?
                    # U2 rows are 0..3 corresponding to A00, A01, A10, A11.
                    # So U2[0, s] is for A00, U2[1, s] for A01, etc.
                    # So we need to iterate over the 4 entries of X.
                    pass
            
            # Let's do it properly.
            # X is a 2x2 matrix.
            # X[0,0], X[0,1], X[1,0], X[1,1].
            # Strassen term s uses U2[0, s] for X[0,0], U2[1, s] for X[0,1], etc.
            # X[i, j] = sum_{p,q} alpha[r, 0, 0, p, q] A_block[p, q][i, j].
            # So coeff for A_block[p, q][i, j] in term s is U2[i*2+j, s] * alpha[r, 0, 0, p, q].
            
            for p in range(2):
                for q in range(2):
                    for i in range(2):
                        for j in range(2):
                            a_coeff = U2[i*2+j, s] * alpha[r, 0, 0, p, q]
                            if a_coeff != 0:
                                global_idx = (p*2+i)*4 + (q*2+j)
                                U[global_idx, col] += a_coeff
            
            # Similarly for V
            for p in range(2):
                for q in range(2):
                    for i in range(2):
                        for j in range(2):
                            b_coeff = V2[i*2+j, s] * beta[r, 0, 0, p, q]
                            if b_coeff != 0:
                                global_idx = (p*2+i)*4 + (q*2+j)
                                V[global_idx, col] += b_coeff
            
            # For W
            # The result of term s is a scalar M_s.
            # This M_s contributes to M_r[0,0][i, j] with coefficient W2[i*2+j, s].
            # M_r[0,0][i, j] contributes to C_block[a, b][i, j] with coefficient c_{ab, r}.
            # C_block[a, b] is at global index (a*2+i)*4 + (b*2+j).
            
            # c_{ab, r} for C00, C01, C10, C11
            c_coeffs = {
                (0, 0): [1, 0, 0, 1, -1, 0, 1],
                (0, 1): [0, 0, 1, 0, 1, 0, 0],
                (1, 0): [0, 1, 0, 1, 0, 0, 0],
                (1, 1): [1, -1, 1, 0, 0, 1, 0]
            }
            
            for a in range(2):
                for b in range(2):
                    c_r = c_coeffs[(a, b)][r]
                    if c_r != 0:
                        for i in range(2):
                            for j in range(2):
                                w_coeff = W2[i*2+j, s] * c_r
                                if w_coeff != 0:
                                    global_idx = (a*2+i)*4 + (b*2+j)
                                    W[global_idx, col] += w_coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
