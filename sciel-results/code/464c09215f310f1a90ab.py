import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation
    # 12 blocks of 2x2x2 (Strassen, rank 7)
    # 6 blocks of 2x1x2 (Schoolbook, rank 4)
    rank = 12 * 7 + 6 * 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen coefficients
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # We need to add contributions for each block.
    # Block (i_block, j_block) corresponds to A rows i_block*2..i_block*2+1, 
    # A cols j_block*2..j_block*2+1 (for 2x2) or j_block*2..j_block*2 (for 2x1)
    
    # A is 4x5. Rows 0-3. Cols 0-4.
    # B is 5x6. Rows 0-4. Cols 0-5.
    # C is 4x6. Rows 0-3. Cols 0-5.
    
    # A blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A13: rows 0-1, cols 4
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    # A23: rows 2-3, cols 4
    
    # B blocks:
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B13: rows 0-1, cols 4-5
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    # B23: rows 2-3, cols 4-5
    # B31: row 4, cols 0-1
    # B32: row 4, cols 2-3
    # B33: row 4, cols 4-5
    
    # C blocks:
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C13: rows 0-1, cols 4-5
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    # C23: rows 2-3, cols 4-5

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # a_r, a_c: top-left index of A block (2x2)
        # b_r, b_c: top-left index of B block (2x2)
        # c_r, c_c: top-left index of C block (2x2)
        
        # M1: (A11+A22)(B11+B22)
        # A indices: (a_r, a_c), (a_r, a_c+1), (a_r+1, a_c), (a_r+1, a_c+1)
        # B indices: (b_r, b_c), (b_r, b_c+1), (b_r+1, b_c), (b_r+1, b_c+1)
        
        # We need to define the 7 multiplications.
        # Each multiplication k involves a linear combination of A entries and B entries.
        # U[row, col] += coeff_A
        # V[row, col] += coeff_B
        # W[row, col] += coeff_C
        
        # M1
        # A: A11 + A22 -> (a_r, a_c) + (a_r+1, a_c+1)
        # B: B11 + B22 -> (b_r, b_c) + (b_r+1, b_c+1)
        # C: C11 += M1, C22 += M1
        U[a_r*m + a_c][col] += 1
        U[(a_r+1)*m + (a_c+1)][col] += 1
        V[b_r*p + b_c][col] += 1
        V[(b_r+1)*p + (b_c+1)][col] += 1
        W[b_c*n + a_r][col] += 1 # C11 top-left? No.
        # W index is k*n + i.
        # C block is at c_r, c_c.
        # C11 is C[c_r, c_c].
        # M1 contributes to C11 and C22.
        # C11 entry (0,0) of block -> global (c_r, c_c).
        # C22 entry (1,1) of block -> global (c_r+1, c_c+1).
        W[c_c*n + c_r][col] += 1
        W[(c_c+1)*n + (c_r+1)][col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        # A: (a_r+1, a_c) + (a_r+1, a_c+1)
        # B: (b_r, b_c)
        # C: C21 += M2, C22 -= M2
        U[(a_r+1)*m + a_c][col] += 1
        U[(a_r+1)*m + (a_c+1)][col] += 1
        V[b_r*p + b_c][col] += 1
        W[c_c*n + (c_r+1)][col] += 1 # C21 top-left
        W[(c_c+1)*n + (c_r+1)][col] -= 1 # C22 bottom-right? No.
        # C21 is block (1,0) of C. Top-left is (c_r+1, c_c).
        # C22 is block (1,1) of C. Top-left is (c_r+1, c_c+1).
        # M2 is 2x2.
        # C21 += M2. So all 4 entries of C21 get +M2?
        # Wait, Strassen formulas are for the whole 2x2 block.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # Each M is a 2x2 matrix.
        # So C11 (2x2) gets M1 (2x2) added.
        # This means each entry of C11 gets the corresponding entry of M1.
        # M1 is the product of two 2x2 matrices.
        # The decomposition of M1 into rank 1 tensors is handled by the 7 multiplications.
        # The 7 multiplications produce 7 scalar values.
        # The entries of C are linear combinations of these 7 scalars.
        # So for each of the 7 multiplications, we add 1 to U, V, and then add coefficients to W.
        # The W coefficients are 0 or +/- 1 depending on the formula.
        
        # Let's re-verify the W updates.
        # For M1:
        # C11 += M1. C11 is 2x2.
        # The 7 multiplications are scalar.
        # How do they map to the 4 entries of C11?
        # Actually, the tensor decomposition of 2x2x2 matrix multiplication is:
        # C = sum_{k=1}^7 (U_k A) (B V_k) W_k?
        # No, the standard form is:
        # C_{ij} = sum_k (sum_{a,b} U_{ab,k} A_{ab}) (sum_{c,d} V_{cd,k} B_{cd}) W_{ij,k}
        # Where U, V are 4x7, W is 4x7.
        # For Strassen:
        # M1: U1 = [[1,0],[0,1]] (A11+A22), V1 = [[1,0],[0,1]] (B11+B22).
        # W1 = [[1,0],[0,1]] (C11, C22).
        # So W1 has 1 at (0,0) and (1,1).
        # This matches my W update: W[c_c*n+c_r] += 1 and W[(c_c+1)*n+(c_r+1)] += 1.
        
        # M2: U2 = [[0,0],[1,1]] (A21+A22), V2 = [[1,0],[0,0]] (B11).
        # W2 = [[0,0],[1,-1]] (C21, C22).
        # C21 += M2. C22 -= M2.
        # W2 has 1 at (1,0) and -1 at (1,1).
        # Global indices:
        # C21 top-left: (c_r+1, c_c).
        # C22 top-left: (c_r+1, c_c+1).
        # So W[(c_c)*n + (c_r+1)] += 1.
        # W[(c_c+1)*n + (c_r+1)] -= 1.
        # My previous update:
        # W[c_c*n + (c_r+1)][col] += 1
        # W[(c_c+1)*n + (c_r+1)][col] -= 1
        # This looks correct.

        # I will implement the full Strassen block addition.
        pass

    def add_outer(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # A is 2x1 at (a_r, a_c)
        # B is 1x2 at (b_r, b_c)
        # C is 2x2 at (c_r, c_c)
        # Rank 4.
        # C11 = A1 B1
        # C12 = A1 B2
        # C21 = A2 B1
        # C22 = A2 B2
        # 4 multiplications.
        
        # 1: A1 B1 -> C11
        U[a_r*m + a_c][col] += 1
        V[b_r*p + b_c][col] += 1
        W[b_c*n + c_r][col] += 1
        col += 1
        
        # 2: A1 B2 -> C12
        U[a_r*m + a_c][col] += 1
        V[b_r*p + (b_c+1)][col] += 1
        W[(b_c+1)*n + c_r][col] += 1
        col += 1
        
        # 3: A2 B1 -> C21
        U[(a_r+1)*m + a_c][col] += 1
        V[b_r*p + b_c][col] += 1
        W[b_c*n + (c_r+1)][col] += 1
        col += 1
        
        # 4: A2 B2 -> C22
        U[(a_r+1)*m + a_c][col] += 1
        V[b_r*p + (b_c+1)][col] += 1
        W[(b_c+1)*n + (c_r+1)][col] += 1
        col += 1

    # Implement Strassen
    def add_strassen_impl(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # M1
        U[a_r*m + a_c][col] += 1
        U[(a_r+1)*m + (a_c+1)][col] += 1
        V[b_r*p + b_c][col] += 1
        V[(b_r+1)*p + (b_c+1)][col] += 1
        W[b_c*n + c_r][col] += 1
        W[(b_c+1)*n + (c_r+1)][col] += 1
        col += 1
        
        # M2
        U[(a_r+1)*m + a_c][col] += 1
        U[(a_r+1)*m + (a_c+1)][col] += 1
        V[b_r*p + b_c][col] += 1
        W[b_c*n + (c_r+1)][col] += 1
        W[(b_c+1)*n + (c_r+1)][col] -= 1
        col += 1
        
        # M3
        U[a_r*m + a_c][col] += 1
        V[b_r*p + (b_c+1)][col] += 1
        V[(b_r+1)*p + (b_c+1)][col] -= 1
        W[(b_c+1)*n + c_r][col] += 1
        W[(b_c+1)*n + (c_r+1)][col] += 1
        col += 1
        
        # M4
        U[(a_r+1)*m + (a_c+1)][col] += 1
        V[(b_r+1)*p + b_c][col] += 1
        V[b_r*p + b_c][col] -= 1
        W[b_c*n + (c_r+1)][col] += 1
        W[b_c*n + c_r][col] += 1
        col += 1
        
        # M5
        U[a_r*m + a_c][col] += 1
        U[a_r*m + (a_c+1)][col] += 1
        V[(b_r+1)*p + (b_c+1)][col] += 1
        W[b_c*n + c_r][col] -= 1
        W[(b_c+1)*n + c_r][col] += 1
        col += 1
        
        # M6
        U[(a_r+1)*m + a_c][col] += 1
        U[a_r*m + a_c][col] -= 1
        V[b_r*p + b_c][col] += 1
        V[b_r*p + (b_c+1)][col] += 1
        W[(b_c+1)*n + (c_r+1)][col] += 1
        col += 1
        
        # M7
        U[a_r*m + (a_c+1)][col] += 1
        U[(a_r+1)*m + (a_c+1)][col] -= 1
        V[(b_r+1)*p + b_c][col] += 1
        V[(b_r+1)*p + (b_c+1)][col] += 1
        W[b_c*n + c_r][col] += 1
        W[(b_c+1)*n + (c_r+1)][col] -= 1
        col += 1

    # Call blocks
    # A11 (0,0), B11 (0,0) -> C11 (0,0)
    add_strassen_impl(0, 0, 0, 0, 0, 0)
    # A11 (0,0), B12 (0,2) -> C12 (0,2)
    add_strassen_impl(0, 0, 0, 2, 0, 2)
    # A11 (0,0), B13 (0,4) -> C13 (0,4)
    add_strassen_impl(0, 0, 0, 4, 0, 4)
    
    # A12 (0,2), B21 (2,0) -> C21 (2,0)
    add_strassen_impl(0, 2, 2, 0, 2, 0)
    # A12 (0,2), B22 (2,2) -> C22 (2,2)
    add_strassen_impl(0, 2, 2, 2, 2, 2)
    # A12 (0,2), B23 (2,4) -> C23 (2,4)
    add_strassen_impl(0, 2, 2, 4, 2, 4)
    
    # A21 (2,0), B11 (0,0) -> C21 (2,0)
    add_strassen_impl(2, 0, 0, 0, 2, 0)
    # A21 (2,0), B12 (0,2) -> C22 (2,2)
    add_strassen_impl(2, 0, 0, 2, 2, 2)
    # A21 (2,0), B13 (0,4) -> C23 (2,4)
    add_strassen_impl(2, 0, 0, 4, 2, 4)
    
    # A22 (2,2), B21 (2,0) -> C21 (2,0)
    add_strassen_impl(2, 2, 2, 0, 2, 0)
    # A22 (2,2), B22 (2,2) -> C22 (2,2)
    add_strassen_impl(2, 2, 2, 2, 2, 2)
    # A22 (2,2), B23 (2,4) -> C23 (2,4)
    add_strassen_impl(2, 2, 2, 4, 2, 4)
    
    # Outer products
    # A13 (0,4), B31 (4,0) -> C13 (0,4)
    add_outer(0, 4, 4, 0, 0, 4)
    # A13 (0,4), B32 (4,2) -> C13 (0,4) ? No, B32 is cols 2-3. C13 is cols 4-5.
    # Wait.
    # C13 = A11 B13 + A12 B23 + A13 B33
    # B13 is cols 4-5. B23 is cols 4-5. B33 is cols 4-5.
    # So A13 multiplies B33.
    # A23 multiplies B33.
    
    # Let's re-check the C block formulas.
    # C11 = A11 B11 + A12 B21 + A13 B31
    # C12 = A11 B12 + A12 B22 + A13 B32
    # C13 = A11 B13 + A12 B23 + A13 B33
    # C21 = A21 B11 + A22 B21 + A23 B31
    # C22 = A21 B12 + A22 B22 + A23 B32
    # C23 = A21 B13 + A22 B23 + A23 B33
    
    # So outer products are:
    # A13 B31 -> C11? No, A13 is 2x1, B31 is 1x2.
    # A13 is cols 4. B31 is rows 4, cols 0-1.
    # Product is 2x2.
    # It contributes to C11?
    # C11 is rows 0-1, cols 0-1.
    # Yes.
    
    # A13 B32 -> C12.
    # A13 B33 -> C13.
    # A23 B31 -> C21.
    # A23 B32 -> C22.
    # A23 B33 -> C23.
    
    add_outer(0, 4, 4, 0, 0, 0) # A13 B31 -> C11
    add_outer(0, 4, 4, 2, 0, 2) # A13 B32 -> C12
    add_outer(0, 4, 4, 4, 0, 4) # A13 B33 -> C13
    add_outer(2, 4, 4, 0, 2, 0) # A23 B31 -> C21
    add_outer(2, 4, 4, 2, 2, 2) # A23 B32 -> C22
    add_outer(2, 4, 4, 4, 2, 4) # A23 B33 -> C23

    return {"U": U, "V": V, "W": W}
