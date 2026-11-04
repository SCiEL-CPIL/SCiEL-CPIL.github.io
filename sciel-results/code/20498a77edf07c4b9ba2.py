import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 4, 4, 5
    # We use a rank-56 Strassen-like decomposition for the 4x4x4 part
    # and rank-16 schoolbook for the 4x4x1 part.
    # Total rank = 72.
    rank = 72

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to get indices
    def Ai(i, j):
        return i * m + j

    def Bj(j, k):
        return j * p + k

    def Ci(k, i):
        return k * n + i

    col = 0

    # The 4x4x4 part can be decomposed into 8 independent 2x2x2 block products.
    # Each 2x2x2 product can be done in rank 7 (Strassen).
    # Total rank for 4x4x4 = 8 * 7 = 56.
    #
    # The 8 products correspond to the 8 block multiplications:
    # C_block(r, c) = sum_k A_block(r, k) * B_block(k, c)
    # But wait, that's the schoolbook approach which would be 8 * 4 = 32 rank-1 terms per block?
    # No.
    #
    # Let's clarify the tensor decomposition structure.
    # The tensor M_{4,4,4} can be viewed as a sum of 8 "block" tensors M_{r,k,c}
    # where each M_{r,k,c} is a 2x2x2 tensor (rank 8 in schoolbook, rank 7 in Strassen).
    # Actually, the standard way to apply Strassen to 4x4 is to view it as 8 independent 2x2 multiplications?
    # No, that's not right.
    #
    # Standard Strassen 4x4:
    # Split A, B, C into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A B = [[C00, C01], [C10, C11]]
    #
    # C00 = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B10-B11) + (A10-A00)(B00+B01) ? No.
    #
    # Let's use the 7-product Strassen for 4x4 directly.
    # The 7 products are 2x2 matrix multiplications.
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 - P4 + P5 + P6
    # C01 = P3 + P4
    # C10 = P2 + P3
    # C11 = P1 - P2 + P5 + P7
    #
    # Each P_i is a 2x2 matrix.
    # To compute a 2x2 matrix product X Y, we need rank 7 (Strassen).
    # So each P_i requires 7 rank-1 terms in the tensor decomposition.
    # Total rank = 7 * 7 = 49.
    #
    # Wait, the inspiration says rank 56 for 4x4x4. Why 56?
    # 56 = 8 * 7.
    # This implies the 4x4x4 problem is decomposed into 8 independent 2x2x2 problems.
    # This happens if we don't use the "cross-block" Strassen formulas, but instead
    # treat the 4x4 multiplication as 8 separate 2x2 block multiplications?
    # No, that would be schoolbook rank 32 for the blocks (8 blocks * 4 mults) or 8*8=64 if we use schoolbook for each 2x2.
    # If we use Strassen for each of the 8 block products, rank is 8 * 7 = 56.
    #
    # Let's verify if this is a valid decomposition.
    # The tensor M_{4,4,4} has entries M[i,j,k].
    # If we split into 2x2 blocks:
    # i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3}.
    # i = 2*i_b + i_l, j = 2*j_b + j_l, k = 2*k_b + k_l.
    # M[i,j,k] = 1.
    #
    # The block decomposition M_{4,4,4} = sum_{i_b, j_b, k_b} M_{i_b, j_b, k_b}
    # where M_{i_b, j_b, k_b} is a 2x2x2 tensor with entries
    # M_{i_b, j_b, k_b}[i_l, j_l, k_l] = 1 if (i_b, j_b, k_b) corresponds to a valid multiplication term?
    #
    # Actually, C[i,k] = sum_j A[i,j] B[j,k].
    # C[2*i_b+i_l, 2*k_b+k_l] = sum_{j_b, j_l} A[2*i_b+i_l, 2*j_b+j_l] B[2*j_b+j_l, 2*k_b+k_l].
    #
    # So the block C_{i_b, k_b} = sum_{j_b} A_{i_b, j_b} B_{j_b, k_b}.
    # This is a sum of 4 block products (j_b=0..3).
    # Each block product A_{i_b, j_b} B_{j_b, k_b} is a 2x2 * 2x2 = 2x2 matrix.
    # There are 4*4 = 16 such block products? No.
    # i_b in {0,1}, k_b in {0,1}. j_b in {0,1}.
    # So for each (i_b, k_b), there are 2 terms (j_b=0,1).
    # Total block products = 2 (i_b) * 2 (k_b) * 2 (j_b) = 8.
    #
    # So C_{i_b, k_b} = A_{i_b, 0} B_{0, k_b} + A_{i_b, 1} B_{1, k_b}.
    #
    # We can compute each term A_{i_b, j_b} B_{j_b, k_b} using Strassen (rank 7).
    # There are 8 such terms.
    # Total rank = 8 * 7 = 56.
    #
    # This is a valid decomposition! It's just the "block Strassen" where we don't exploit the linearity across j_b to reduce the number of 2x2 multiplications. It's essentially applying Strassen to each of the 8 block products independently.
    #
    # Let's implement this.
    #
    # The 8 terms are:
    # 1. A00 B00 -> contributes to C00
    # 2. A00 B01 -> contributes to C01
    # 3. A01 B10 -> contributes to C00
    # 4. A01 B11 -> contributes to C01
    # 5. A10 B00 -> contributes to C10
    # 6. A10 B01 -> contributes to C11
    # 7. A11 B10 -> contributes to C10
    # 8. A11 B11 -> contributes to C11
    #
    # Wait, C00 = A00 B00 + A01 B10.
    # C01 = A00 B01 + A01 B11.
    # C10 = A10 B00 + A11 B10.
    # C11 = A10 B01 + A11 B11.
    #
    # So the 8 terms are:
    # (A00, B00, C00)
    # (A01, B10, C00)
    # (A00, B01, C01)
    # (A01, B11, C01)
    # (A10, B00, C10)
    # (A11, B10, C10)
    # (A10, B01, C11)
    # (A11, B11, C11)
    #
    # Each term is a 2x2 * 2x2 = 2x2 product.
    # We use Strassen for each.
    
    def add_strassen_2x2(a_base, b_base, c_base, c_col):
        # a_base: (row, col) of A block (0 or 2, 0 or 2)
        # b_base: (row, col) of B block (0 or 2, 0 or 2)
        # c_base: (row, col) of C block (0 or 2, 0 or 2)
        
        # Local indices for 2x2 block
        # A entries: a_base[0]+r, a_base[1]+c
        # B entries: b_base[0]+r, b_base[1]+c
        # C entries: c_base[0]+r, c_base[1]+c
        
        def a_idx(r, c):
            return Ai(a_base[0]+r, a_base[1]+c)
        def b_idx(r, c):
            return Bj(b_base[0]+r, b_base[1]+c)
        def c_idx(r, c):
            return Ci(c_base[1]+c, c_base[0]+r) # W[k*n+i]
            
        # Strassen 2x2x2
        # P1 = (A00+A11)(B00+B11) -> C00, C11
        U[a_idx(0,0)][c_col] += 1
        U[a_idx(1,1)][c_col] += 1
        V[b_idx(0,0)][c_col] += 1
        V[b_idx(1,1)][c_col] += 1
        W[c_idx(0,0)][c_col] += 1
        W[c_idx(1,1)][c_col] += 1
        c_col += 1
        
        # P2 = (A10+A11)B00 -> C10, C11
        U[a_idx(1,0)][c_col] += 1
        U[a_idx(1,1)][c_col] += 1
        V[b_idx(0,0)][c_col] += 1
        W[c_idx(1,0)][c_col] += 1
        W[c_idx(1,1)][c_col] -= 1
        c_col += 1
        
        # P3 = A00(B10-B11) -> C01, C11
        U[a_idx(0,0)][c_col] += 1
        V[b_idx(1,0)][c_col] += 1
        V[b_idx(1,1)][c_col] -= 1
        W[c_idx(0,1)][c_col] += 1
        W[c_idx(1,1)][c_col] += 1
        c_col += 1
        
        # P4 = A11(B01-B00) -> C00, C01
        U[a_idx(1,1)][c_col] += 1
        V[b_idx(0,1)][c_col] += 1
        V[b_idx(0,0)][c_col] -= 1
        W[c_idx(0,0)][c_col] -= 1
        W[c_idx(0,1)][c_col] += 1
        c_col += 1
        
        # P5 = (A00+A01)B11 -> C00, C11
        U[a_idx(0,0)][c_col] += 1
        U[a_idx(0,1)][c_col] += 1
        V[b_idx(1,1)][c_col] += 1
        W[c_idx(0,0)][c_col] += 1
        W[c_idx(1,1)][c_col] += 1
        c_col += 1
        
        # P6 = (A10-A00)(B00+B01) -> C00, C10
        U[a_idx(1,0)][c_col] += 1
        U[a_idx(0,0)][c_col] -= 1
        V[b_idx(0,0)][c_col] += 1
        V[b_idx(0,1)][c_col] += 1
        W[c_idx(0,0)][c_col] += 1
        W[c_idx(1,0)][c_col] += 1
        c_col += 1
        
        # P7 = (A01-A11)(B10+B11) -> C11, C01
        U[a_idx(0,1)][c_col] += 1
        U[a_idx(1,1)][c_col] -= 1
        V[b_idx(1,0)][c_col] += 1
        V[b_idx(1,1)][c_col] += 1
        W[c_idx(1,1)][c_col] += 1
        W[c_idx(0,1)][c_col] -= 1
        c_col += 1
        
        return c_col

    # Define the 8 block products
    # (A_row_block, A_col_block, B_row_block, B_col_block, C_row_block, C_col_block)
    products = [
        (0, 0, 0, 0, 0, 0), # A00 B00 -> C00
        (0, 1, 1, 0, 0, 0), # A01 B10 -> C00
        (0, 0, 0, 1, 0, 1), # A00 B01 -> C01
        (0, 1, 1, 1, 0, 1), # A01 B11 -> C01
        (1, 0, 0, 0, 1, 0), # A10 B00 -> C10
        (1, 1, 1, 0, 1, 0), # A11 B10 -> C10
        (1, 0, 0, 1, 1, 1), # A10 B01 -> C11
        (1, 1, 1, 1, 1, 1), # A11 B11 -> C11
    ]
    
    for (ar, ac, br, bc, cr, cc) in products:
        a_base = (ar*2, ac*2)
        b_base = (br*2, bc*2)
        c_base = (cr*2, cc*2)
        col = add_strassen_2x2(a_base, b_base, c_base, col)
        
        if time.time() > deadline:
            break

    # Part 2: 4x4x1 (rank 16)
    # A (4x4) * B_col4 (4x1) -> C_col4 (4x1)
    # B_col4 is column 4 of B (indices 4, 9, 14, 19 in V)
    # C_col4 is column 4 of C (indices 16, 17, 18, 19 in W)
    
    for i in range(4):
        for j in range(4):
            if col >= rank:
                break
            U[i*4 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
