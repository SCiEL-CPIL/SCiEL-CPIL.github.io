"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Recursive Strassen algorithm for 4x4 matrix multiplication.
    Uses 7 multiplications at the top level (Strassen for 2x2 blocks),
    each of which is a 2x2 matrix multiplication using rank 7 (Strassen).
    Total rank: 7 * 7 = 49.
    
    Index conventions:
    - U has shape (n*m, rank) = (16, 49), indexed by [i*m+j, c]
    - V has shape (m*p, rank) = (16, 49), indexed by [j*p+k, c]  
    - W has shape (p*n, rank) = (16, 49), indexed by [k*n+i, c]
    
    The decomposition satisfies:
    C[i,k] = sum_j A[i,j] * B[j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    Wait, that's not quite right. Let me think about this more carefully.
    
    Actually, the tensor decomposition for matrix multiplication is:
    T[i,j,k] = delta[i,k] (where T represents the multiplication tensor)
    
    A rank-1 decomposition: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    
    But the problem states:
    - U: (n*m, rank) -> indexed by (i,j) pairs, so U[i*m+j, c]
    - V: (m*p, rank) -> indexed by (j,k) pairs, so V[j*p+k, c]
    - W: (p*n, rank) -> indexed by (k,i) pairs, so W[k*n+i, c]
    
    And the decomposition is:
    For each output element C[k*n+i] (which is C[i,k] in standard notation):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor form: The matrix multiplication tensor T has indices (i,j,k) where
    T[i,j,k] = 1 if this term contributes to C[i,k] from A[i,j]*B[j,k].
    
    Actually, the standard tensor for matrix multiplication of n×m and m×p matrices is:
    T[i,j,k] = 1 (for all valid i,j,k), and the multiplication is:
    C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k]
    
    A rank-1 decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    (using the index conventions from the problem)
    
    So we need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    Wait, that's not right either. Let me re-read the problem.
    
    "output factor follows the official trace convention W[k*n+i,r]"
    
    I think the decomposition is:
    The matrix product C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]
    
    This can be written as a tensor network:
    C[i,k] = sum_{j,c} A[i,j] * U[i,j,c] * V[j,k,c] * W[k,i,c] * B[j,k]
    
    No wait, that doesn't make sense. Let me think differently.
    
    The tensor T for matrix multiplication has T[i,j,k] = 1 for all i,j,k in the valid ranges.
    A rank-r decomposition is: T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    So we need: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=4
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Implement recursive Strassen for 4x4 matrix multiplication
    # Top level: 7 Strassen products, each a 2x2 matrix multiplication (rank 7)
    # Total rank: 49
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Strassen's 7 products for multiplying two 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For 4x4: Split into 2x2 blocks of 2x2 submatrices
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = [[C00, C01], [C10, C11]] where each Ci is 2x2
    #
    # Each Mi is a 2x2 matrix product, which itself uses Strassen (rank 7)
    # So for each of the 7 top-level products, we have 7 rank-1 terms
    # Total: 7 * 7 = 49
    
    # For each top-level product s (0-6), and each Strassen sub-product t (0-6),
    # we have one rank-1 term.
    
    # Define the Strassen coefficients for a 2x2 multiplication
    # For product M_s = (A_linear_comb) * (B_linear_comb)
    # A coefficients: a_coeffs[s][block] where block is (00,01,10,11)
    # B coefficients: b_coeffs[s][block]
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # A block indices: 00=(0,0), 01=(0,1), 10=(1,0), 11=(1,1)
    # For 2x2 blocks within 4x4:
    # A00 covers rows 0-1, cols 0-1
    # A01 covers rows 0-1, cols 2-3
    # A10 covers rows 2-3, cols 0-1
    # A11 covers rows 2-3, cols 2-3
    
    # Strassen sub-products for a 2x2 matrix multiplication (rank 7)
    # For a 2x2 matrix P * Q where P = [[p00,p01],[p10,p11]], Q = [[q00,q01],[q10,q11]]
    # N1 = (p00+p11)*(q00+q11)
    # N2 = (p10+p11)*q00
    # N3 = p00*(q01-q11)
    # N4 = p11*(q10-q00)
    # N5 = (p00+p01)*q11
    # N6 = (p10-p00)*(q00+q01)
    # N7 = (p01-p11)*(q10+q11)
    #
    # R00 = N1 + N4 - N5 + N7
    # R01 = N3 + N5
    # R10 = N2 + N4
    # R11 = N1 - N2 + N3 + N6
    
    # For each top-level Strassen product s, the "A" and "B" are linear combinations
    # of 2x2 blocks. Each such linear combination is itself a 2x2 matrix.
    # Then we multiply these two 2x2 matrices using Strassen (7 rank-1 terms).
    
    # Let's define:
    # For top-level product s, A_s = sum_{block} a_coeff[s][block] * A_block
    # For top-level product s, B_s = sum_{block} b_coeff[s][block] * B_block
    #
    # Then M_s = A_s * B_s, and we decompose this 2x2 multiplication into 7 rank-1 terms.
    
    # A coefficients for each top-level product (indexed by block: 00, 01, 10, 11)
    a_coeffs = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [1, 0, -1, 0], # M6: A10 - A00
        [0, 1, 0, -1], # M7: A01 - A11
    ]
    
    # B coefficients for each top-level product (indexed by block: 00, 01, 10, 11)
    b_coeffs = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1],  # M7: B10 + B11
    ]
    
    # Strassen sub-product coefficients for 2x2 multiplication
    # For N_t = (P_linear_comb) * (Q_linear_comb)
    # P coefficients: p_coeffs[t][elem] where elem is (00,01,10,11)
    # Q coefficients: q_coeffs[t][elem]
    
    p_coeffs = [
        [1, 0, 0, 1],  # N1: p00 + p11
        [0, 0, 1, 1],  # N2: p10 + p11
        [1, 0, 0, 0],  # N3: p00
        [0, 0, 0, 1],  # N4: p11
        [1, 1, 0, 0],  # N5: p00 + p01
        [1, 0, -1, 0], # N6: p10 - p00
        [0, 1, 0, -1], # N7: p01 - p11
    ]
    
    q_coeffs = [
        [1, 0, 0, 1],  # N1: q00 + q11
        [1, 0, 0, 0],  # N2: q00
        [0, 1, 0, -1], # N3: q01 - q11
        [0, 0, 1, -1], # N4: q10 - q00
        [0, 0, 0, 1],  # N5: q11
        [1, 1, 0, 0],  # N6: q00 + q01
        [0, 0, 1, 1],  # N7: q10 + q11
    ]
    
    # C output coefficients: C_block = sum_t c_coeffs[t][block] * N_t
    c_coeffs = [
        [1, 0, 0, 1],  # N1 contributes to C00, C11
        [0, 0, 1, 0],  # N2 contributes to C10
        [0, 1, 0, 0],  # N3 contributes to C01
        [1, 0, 0, 0],  # N4 contributes to C00, C10
        [1, 0, 0, 0],  # N5 contributes to C00 (with -1)
        [0, 0, 0, 1],  # N6 contributes to C11
        [1, 0, 0, 1],  # N7 contributes to C00, C11
    ]
    
    # Wait, I need to be more careful. Let me define the C coefficients properly.
    # C00 = N1 + N4 - N5 + N7
    # C01 = N3 + N5
    # C10 = N2 + N4
    # C11 = N1 - N2 + N3 + N6
    
    # So for each output block (00, 01, 10, 11), the coefficient of N_t:
    # c_out[output_block][t]
    c_out = [
        [1, 0, 0, 1, -1, 0, 1],  # C00: N1 + N4 - N5 + N7
        [0, 0, 1, 0, 1, 0, 0],   # C01: N3 + N5
        [0, 1, 0, 1, 0, 0, 0],   # C10: N2 + N4
        [1, -1, 1, 0, 0, 1, 0],  # C11: N1 - N2 + N3 + N6
    ]
    
    # Now, for each top-level product s and sub-product t:
    # The rank-1 term has:
    # U part: for each element (i,j) in the 4x4 A matrix, the coefficient
    #   comes from a_coeffs[s][block_of(i,j)] * p_coeffs[t][elem_of(i,j)]
    # V part: for each element (j,k) in the 4x4 B matrix, the coefficient
    #   comes from b_coeffs[s][block_of(j,k)] * q_coeffs[t][elem_of(j,k)]
    # W part: for each element (k,i) in the 4x4 C matrix, the coefficient
    #   comes from c_out[output_block_of(k,i)][t] * (top-level c coefficient)
    
    # Wait, I also need the top-level C coefficients. The 7 top-level products
    # combine to form the 4 output blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each output block, the coefficient of M_s:
    c_top = [
        [1, 0, 0, 1, -1, 0, 1],  # C00: M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],   # C01: M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C10: M2 + M4
        [1, -1, 1, 0, 0, 1, 0],  # C11: M1 - M2 + M3 + M6
    ]
    
    # Block mapping for 4x4 matrix:
    # Block 00: rows 0-1, cols 0-1 -> elements (0,0),(0,1),(1,0),(1,1)
    # Block 01: rows 0-1, cols 2-3 -> elements (0,2),(0,3),(1,2),(1,3)
    # Block 10: rows 2-3, cols 0-1 -> elements (2,0),(2,1),(3,0),(3,1)
    # Block 11: rows 2-3, cols 2-3 -> elements (2,2),(2,3),(3,2),(3,3)
    
    def get_block(i, j):
        """Get block index (0-3) for element (i,j) in 4x4 matrix."""
        if i < 2 and j < 2:
            return 0  # 00
        elif i < 2 and j >= 2:
            return 1  # 01
        elif i >= 2 and j < 2:
            return 2  # 10
        else:
            return 3  # 11
    
    def get_elem_in_block(i, j):
        """Get element index within block (0-3) for element (i,j) in 4x4 matrix.
        Within a 2x2 block: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3"""
        local_i = i % 2
        local_j = j % 2
        return local_i * 2 + local_j
    
    def get_output_block(k, i):
        """Get output block index for C[k,i] (which is C[i,k] in standard notation).
        Wait, the problem says W[k*n+i, r], so the output is indexed by (k,i).
        In standard matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k].
        So the output element at position (k,i) in W corresponds to C[i,k].
        The block structure for C[i,k]:
        Block 00: i in 0-1, k in 0-1
        Block 01: i in 0-1, k in 2-3
        Block 10: i in 2-3, k in 0-1
        Block 11: i in 2-3, k in 2-3
        """
        if i < 2 and k < 2:
            return 0  # 00
        elif i < 2 and k >= 2:
            return 1  # 01
        elif i >= 2 and k < 2:
            return 2  # 10
        else:
            return 3  # 11
    
    def get_elem_in_output_block(k, i):
        """Get element index within output block for C[i,k].
        Within a 2x2 block: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        where the block is indexed by (i_local, k_local)"""
        local_i = i % 2
        local_k = k % 2
        return local_i * 2 + local_k
    
    # Now build the decomposition
    for s in range(7):  # Top-level Strassen product
        for t in range(7):  # Sub-level Strassen product
            c_idx = s * 7 + t  # Column index in U, V, W
            
            # U part: for each (i,j) in A
            for i in range(4):
                for j in range(4):
                    block_a = get_block(i, j)
                    elem_a = get_elem_in_block(i, j)
                    coeff = a_coeffs[s][block_a] * p_coeffs[t][elem_a]
                    if coeff != 0:
                        U[i * m + j][c_idx] = coeff
            
            # V part: for each (j,k) in B
            for j in range(4):
                for k in range(4):
                    block_b = get_block(j, k)
                    elem_b = get_elem_in_block(j, k)
                    coeff = b_coeffs[s][block_b] * q_coeffs[t][elem_b]
                    if coeff != 0:
                        V[j * p + k][c_idx] = coeff
            
            # W part: for each (k,i) in C (output indexed as W[k*n+i])
            for k in range(4):
                for i in range(4):
                    out_block = get_output_block(k, i)
                    out_elem = get_elem_in_output_block(k, i)
                    # The coefficient is c_top[out_block][s] * c_out[out_elem][t]
                    # Wait, c_out is indexed by [output_block][t] for the sub-product
                    # But I defined c_out as [output_block][t] where t is the sub-product index
                    # Actually, I think I need to reconsider.
                    
                    # The top-level product M_s contributes to output block out_block
                    # with coefficient c_top[out_block][s]
                    # The sub-product N_t within M_s contributes to the element out_elem
                    # of that block with coefficient... 
                    
                    # Actually, the sub-product N_t is a scalar (rank-1 term) that
                    # contributes to the entire 2x2 block. The way Strassen works:
                    # N_t is a scalar product, and it contributes to multiple output
                    # elements within the block.
                    
                    # I think the correct interpretation is:
                    # M_s = A_s * B_s is a 2x2 matrix
                    # M_s[out_elem] = sum_t c_out[out_block_of_M_s][out_elem][t] * N_t
                    # But wait, c_out should be indexed by [output_position_in_block][t]
                    
                    # Let me redefine: For a 2x2 matrix product, the output element
                    # at position (r,c) in the 2x2 result is:
                    # R00 = N1 + N4 - N5 + N7
                    # R01 = N3 + N5
                    # R10 = N2 + N4
                    # R11 = N1 - N2 + N3 + N6
                    #
                    # So for output element at local position (r,c) in the 2x2 block,
                    # the coefficient of N_t is:
                    r = out_elem // 2  # local row in 2x2 block
                    c = out_elem % 2   # local col in 2x2 block
                    
                    if r == 0 and c == 0:
                        sub_coeff = [1, 0, 0, 1, -1, 0, 1][t]  # N1+N4-N5+N7
                    elif r == 0 and c == 1:
                        sub_coeff = [0, 0, 1, 0, 1, 0, 0][t]   # N3+N5
                    elif r == 1 and c == 0:
                        sub_coeff = [0, 1, 0, 1, 0, 0, 0][t]   # N2+N4
                    else:  # r == 1 and c == 1
                        sub_coeff = [1, -1, 1, 0, 0, 1, 0][t]  # N1-N2+N3+N6
                    
                    top_coeff = c_top[out_block][s]
                    coeff = top_coeff * sub_coeff
                    if coeff != 0:
                        W[k * n + i][c_idx] = coeff
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, rank):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
                    # Wait, no. The decomposition should satisfy:
                    # For the matrix multiplication tensor, T[i,j,k] = 1
                    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False, (i, j, k, total)
        return True, None
    
    is_valid, error_info = verify(U, V, W, n, m, p, rank)
    
    if not is_valid:
        # Fallback to schoolbook if Strassen implementation is wrong
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
