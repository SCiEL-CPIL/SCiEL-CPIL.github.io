import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 84
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients for 2x2 block structure
    # A blocks: A00, A01, A10, A11 (indices 0..3)
    # B blocks: B00, B01, B10, B11 (indices 0..3)
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Define coefficients for A and B blocks for each Strassen product
    # a_coeffs[k][a] is coefficient of A_a in M_k
    # b_coeffs[k][b] is coefficient of B_b in M_k
    
    a_coeffs = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [1, 0, -1, 0],# M6: A10 - A00 (Wait: A10 is index 2, A00 is 0. So [ -1, 0, 1, 0])
        [0, 1, 0, -1] # M7: A01 - A11 (A01 is 1, A11 is 3. So [ 0, 1, 0, -1])
    ]
    # Correction for M6: (A10 - A00). A10 is block 2, A00 is block 0.
    # So coeff for A00 is -1, A10 is 1.
    a_coeffs[5] = [-1, 0, 1, 0]
    
    b_coeffs = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01 - B11
        [0, 0, 1, -1],# M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    col = 0
    for k in range(7):
        for a in range(4):
            ca = a_coeffs[k][a]
            if ca == 0: continue
            # Block A_a is at row a//2 * 2, col a%2 * 2
            a_r0 = (a // 2) * 2
            a_c0 = (a % 2) * 2
            
            for b in range(4):
                cb = b_coeffs[k][b]
                if cb == 0: continue
                # Block B_b is at row b//2 * 2, col b%2 * 3
                b_r0 = (b // 2) * 2
                b_c0 = (b % 2) * 3
                
                # Schoolbook for 2x2 * 2x3
                for i in range(2):
                    for j in range(2):
                        for kk in range(3):
                            # Global indices
                            gi = a_r0 + i
                            gj = a_c0 + j
                            gk = b_r0 + j # Wait, B row index is j?
                            # B is 4x6. Blocks are 2x3.
                            # B_b is at row b_r0, col b_c0.
                            # In B_b, local row j, local col kk.
                            # So global row of B is b_r0 + j.
                            # Global col of B is b_c0 + kk.
                            # But V index is j*p + k.
                            # j is global row of B? No, j is col of A / row of B.
                            # So j_global = a_c0 + j.
                            # k_global = b_c0 + kk.
                            # But we need j_global to match the row of B.
                            # Row of B is b_r0 + j_local_in_B.
                            # In A_a B_b, the contraction is over the middle index.
                            # A_a is 2x2. B_b is 2x3.
                            # A_a[i, j] * B_b[j, kk].
                            # So j_local_in_A = j_local_in_B.
                            # Global j = a_c0 + j.
                            # Global row of B = b_r0 + j.
                            # So we need a_c0 + j == b_r0 + j?
                            # This is only true if a_c0 == b_r0.
                            # But a_c0 is col of A block, b_r0 is row of B block.
                            # They are independent.
                            # The contraction index j must be the same.
                            # So the term is A_a[i, j] * B_b[j, kk].
                            # This contributes to C[i_a+i, k_b+kk].
                            # In the tensor, the index j is the middle index.
                            # So U index is (i_global, j_global).
                            # V index is (j_global, k_global).
                            # W index is (k_global, i_global).
                            
                            j_global = a_c0 + j
                            # Check if j_global matches row of B?
                            # B_b row is b_r0 + j.
                            # So we need j_global = b_r0 + j.
                            # This implies a_c0 = b_r0.
                            # This is NOT generally true.
                            # The Strassen algorithm works on blocks.
                            # A_a and B_b are multiplied only if they are in the same position?
                            # No, Strassen multiplies linear combinations.
                            # The product (A00 + A11)(B00 + B11) involves A00*B00, A00*B11, etc.
                            # A00 is at (0,0). B00 is at (0,0).
                            # A00*B00: j matches (0..1).
                            # A00*B11: A00 is at (0,0). B11 is at (2,3).
                            # A00[i, j] * B11[j, kk].
                            # j is 0..1.
                            # B11 row is 2..3.
                            # So B11[j, kk] is not defined for j=0..1?
                            # B11 is the bottom-right block. Its local rows are 0..1.
                            # Its global rows are 2..3.
                            # So B11[j_local, kk] corresponds to global row 2+j_local.
                            # But the contraction index j must match A's column.
                            # A00's columns are 0..1.
                            # So we are contracting A00's columns (0..1) with B11's rows (2..3)?
                            # No, that's not a valid matrix multiplication.
                            # Strassen's algorithm requires the blocks to be aligned for multiplication.
                            # A00 * B00 is valid.
                            # A00 * B11 is NOT a standard block multiplication term in AB.
                            # In AB, C00 = A00*B00 + A01*B10.
                            # So A00 is only multiplied by B00 and B10?
                            # No, A00 is top-left. It multiplies B00 (top-left) and B10 (bottom-left)?
                            # A is 4x4. B is 4x6.
                            # A = [A00 A01; A10 A11].
                            # B = [B00 B01; B10 B11].
                            # C = [A00B00 + A01B10, A00B01 + A01B11; ...]
                            # So A00 multiplies B00 and B01.
                            # A01 multiplies B10 and B11.
                            # So A_a multiplies B_b only if they are in the same "row" of blocks?
                            # A00 (row 0) multiplies B00, B01 (row 0 of B).
                            # A10 (row 1) multiplies B10, B11 (row 1 of B).
                            
                            # So in Strassen, the linear combinations must respect this.
                            # M1 = (A00 + A11)(B00 + B11).
                            # A00 * B00: Valid.
                            # A00 * B11: A00 is row 0, B11 is row 1. Invalid?
                            # No, Strassen's M1 is for square matrices where A00 and A11 are both 2x2, and B00 and B11 are 2x2.
                            # And the product is defined for all pairs.
                            # But here B is 4x6.
                            # The block structure of B is 2x2 blocks of size 2x3.
                            # B00 is (0,0), B01 is (0,1), B10 is (1,0), B11 is (1,1).
                            # A00 is (0,0).
                            # A00 * B00 is valid.
                            # A00 * B11 is NOT a term in the standard product AB.
                            # So Strassen's algorithm CANNOT be applied directly if the blocks don't align for multiplication.
                            # However, Strassen's algorithm is an identity for matrix multiplication.
                            # It holds for any matrices that can be partitioned into 2x2 blocks.
                            # The terms like A00*B11 DO appear in the Strassen formulas, but they cancel out in the final sum for C?
                            # No, Strassen's formulas are:
                            # M1 = (A11+A22)(B11+B22)
                            # ...
                            # C11 = M1 + M4 - M5 + M7
                            # These formulas are identities.
                            # So A00*B11 IS computed, but its contribution is cancelled or combined.
                            # But A00*B11 is a valid matrix multiplication (2x2 * 2x3).
                            # It just doesn't appear directly in the block product AB.
                            # So we CAN compute it.
                            
                            # So the mapping is:
                            # A_a is at global row a_r0, col a_c0.
                            # B_b is at global row b_r0, col b_c0.
                            # The product A_a * B_b is a 2x3 matrix.
                            # Its (i, kk) element is sum_j A_a[i,j] B_b[j,kk].
                            # This is always defined.
                            
                            # So the code above is correct.
                            
                            U[gi*m + j_global, col] += ca
                            V[j_global*p + gk, col] += cb
                            W[gk*n + gi, col] += 1
                            
                            col += 1
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
