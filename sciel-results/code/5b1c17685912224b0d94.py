"""Program for matrix multiplication tensor decomposition 2x5x6."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # We'll use a block decomposition approach.
    # Split m=5 into blocks: m1=2, m2=3
    # Split p=6 into blocks: p1=2, p2=2, p3=2
    # 
    # For each (m_block, p_block) pair, we compute A[m_block, n] * B[m_block, p_block]
    # where A is n x m_block and B is m_block x p_block.
    #
    # Block (2,2): rank 7 (Strassen)
    # Block (3,2): We can split 3x2 into (1,2) + (2,2) -> rank 2 + 7 = 9
    #              Or use a direct 3x2 decomposition. Let's think...
    #              Actually for 3x2: the schoolbook is 6. Can we do better?
    #              3x2 tensor: C[i,k] = sum_j A[i,j]*B[j,k], i in 0..1, j in 0..2, k in 0..1
    #              This is equivalent to multiplying a 2x3 matrix by a 3x2 matrix.
    #              Rank of 2x3x2 tensor... schoolbook is 6.
    #              We can split the 3 into 1+2: rank(1,3,2) + rank(2,3,2)
    #              rank(1,3,2) = 1*3*2 = 6 (schoolbook, but actually it's just 2 since it's a row vector times matrix... no)
    #              Wait, for 1x3 * 3x2: C[0,k] = sum_j A[0,j]*B[j,k]. This is 2 outputs, each is a dot product of length 3.
    #              We can do: M1 = A[0,0]*B[0,0], M2 = A[0,1]*B[1,0], M3 = A[0,2]*B[2,0], 
    #                         M4 = A[0,0]*B[0,1], M5 = A[0,1]*B[1,1], M6 = A[0,2]*B[2,1]
    #              That's 6. Can we do better? 
    #              C[0,0] = a0*b0 + a1*b1 + a2*b2
    #              C[0,1] = a0*c0 + a1*c1 + a2*c2
    #              This is like computing two dot products. The rank is at most 6, and I believe it's exactly 6
    #              unless there's some clever sharing.
    #              
    #              Actually, let me reconsider. For a 1 x m x p tensor, the rank is m*p (schoolbook) 
    #              because each output entry is an independent dot product and there's no way to share
    #              across different output entries since the A vector is the same but B is different.
    #              Wait, actually we CAN share: if we compute partial products, but in tensor decomposition
    #              each "multiplication" is a single scalar product of a linear form in A and a linear form in B.
    #              
    #              For 1x3x2: 
    #              C00 = a0*b0 + a1*b1 + a2*b2
    #              C01 = a0*c0 + a1*c1 + a2*c2
    #              Can we express this with fewer than 6 multiplications?
    #              Let's try 5:
    #              M1 = (a0+a1)(b0+c0) = a0*b0 + a0*c0 + a1*b0 + a1*c0
    #              M2 = (a1+a2)(b1+c1) = a1*b1 + a1*c1 + a2*b1 + a2*c1
    #              M3 = (a2+a0)(b2+c2) = a2*b2 + a2*c2 + a0*b2 + a0*c2
    #              M4 = a0*(b0-c0) ... this is getting complicated.
    #              
    #              Actually, the rank of the 1 x m x p tensor is m*p for m,p >= 1, because the tensor
    #              is "generic" enough. Let me just use schoolbook for the small blocks.
    #
    # So my plan:
    # Split m=5 into [2, 3]
    # Split p=6 into [2, 2, 2]
    #
    # Blocks:
    # (m=2, p=2): Strassen, rank 7
    # (m=3, p=2): schoolbook, rank 6
    #
    # Total: 3 * (7 + 6) = 3 * 13 = 39
    #
    # Wait, that's much better than 60! Let me verify this is correct.
    #
    # Actually, I need to be more careful. The matrix A is 2x5, B is 5x6.
    # A has columns a0..a4 (each is a 2-vector)
    # B has rows b0..b4 (each is a 6-vector)
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #
    # If I split the j-index into [0,1] and [2,3,4]:
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{4} A[i,j]*B[j,k]
    #
    # The first part is a 2x2 * 2x6 multiplication.
    # The second part is a 2x3 * 3x6 multiplication.
    #
    # For 2x2 * 2x6: I can split the 6 into three 2s, giving 3 * rank(2x2x2) = 3*7 = 21
    # For 2x3 * 3x6: I can split the 6 into three 2s, giving 3 * rank(2x3x2)
    #
    # What's rank(2x3x2)? 
    # C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k], i in {0,1}, k in {0,1}
    # Schoolbook: 6
    # Can we do better? Let's think...
    # C00 = a00*b00 + a01*b10 + a02*b20
    # C01 = a00*b01 + a01*b11 + a02*b21
    # C10 = a10*b00 + a11*b10 + a12*b20
    # C11 = a10*b01 + a11*b11 + a12*b21
    #
    # This is a 4-output tensor. Each output is a bilinear form.
    # I believe the rank is 6 for generic 2x3x2, but let me try to find a decomposition with rank < 6.
    #
    # Actually, I recall that rank(2,3,2) = 6. Let me just use schoolbook.
    #
    # So total rank = 3 * (7 + 6) = 39
    #
    # But wait, can I do even better by splitting differently?
    # What if I split m=5 into [1,1,1,1,1] and p=6 into [2,2,2]?
    # Then each block is 2x1 * 1x2, which has rank 2 (since C[i,k] = A[i,0]*B[0,k], just 2 multiplications... 
    # actually 1 multiplication per output, so 2 total for 2 outputs? No, C has 2x2=4 entries, each is A[i,j]*B[j,k], 
    # so 4 schoolbook multiplications, but we can share: M1 = A[0,0]*B[0,0], M2 = A[0,0]*B[0,1], 
    # M3 = A[1,0]*B[0,0], M4 = A[1,0]*B[0,1]. That's 4. Or we can do M1 = (A[0,0]+A[1,0])*(B[0,0]+B[0,1]), etc.
    # For 2x1x2: rank is 4? Or can we do 3?
    # C00 = a*b, C01 = a*c, C10 = d*b, C11 = d*c
    # M1 = (a+d)(b+c) = ab+ac+db+dc
    # M2 = (a+d)(b-c) = ab-ac+db-dc
    # M3 = a*(b+c) = ab+ac
    # M4 = d*(b+c) = db+dc
    # C00 = (M1+M2)/2, C01 = (M1-M2)/2, C10 = (M3+M4)/2 - (M1+M2)/2 + ... this is getting messy.
    # I think rank(2,1,2) = 4.
    #
    # So 5 blocks of (2,1,2) * 3 p-blocks of size 2: 5*3*4 = 60. Same as schoolbook.
    #
    # The [2,3] split with [2,2,2] p-split giving 39 seems good. Let me implement that.
    #
    # Actually, let me reconsider. Can I split p=6 into [3,3] instead of [2,2,2]?
    # Then:
    # (m=2, p=3): rank(2,2,3) * 2 blocks
    # (m=3, p=3): rank(2,3,3) * 2 blocks
    #
    # rank(2,2,3): schoolbook is 12. Can we do better?
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k], i in {0,1}, k in {0,1,2}
    # This is 2x2 * 2x3. We can think of it as 3 separate 2x2x1 multiplications, each rank 4, total 12.
    # Or we can use Strassen-like: split the 3 into 1+2.
    # rank(2,2,1) + rank(2,2,2) = 4 + 7 = 11.
    # So rank(2,2,3) <= 11.
    #
    # rank(2,3,3): schoolbook is 18. Split 3 into 1+2: rank(2,3,1) + rank(2,3,2) = 6 + 6 = 12.
    # Or split the m=3 into 1+2: rank(2,1,3) + rank(2,2,3) = 6 + 11 = 17. Worse.
    # So rank(2,3,3) <= 12.
    #
    # Total: 2*(11 + 12) = 2*23 = 46. Worse than 39.
    #
    # What about splitting m=5 into [2,2,1] and p=6 into [2,2,2]?
    # (m=2,p=2): 7, (m=2,p=2): 7, (m=1,p=2): rank(2,1,2)=4
    # Total per p-block: 7+7+4 = 18
    # Total: 3*18 = 54. Worse than 39.
    #
    # So [2,3] x [2,2,2] giving 39 seems best so far.
    #
    # Let me also consider: can I do rank(2,3,2) better than 6?
    # I'll try to find a rank-5 decomposition for 2x3x2.
    # C00 = a00*b00 + a01*b10 + a02*b20
    # C01 = a00*b01 + a01*b11 + a02*b21
    # C10 = a10*b00 + a11*b10 + a12*b20
    # C11 = a10*b01 + a11*b11 + a12*b21
    #
    # Let me try:
    # M1 = (a00+a01)(b00+b10) = a00*b00 + a00*b10 + a01*b00 + a01*b10
    # M2 = (a01+a02)(b10+b20) = a01*b10 + a01*b20 + a02*b10 + a02*b20
    # M3 = (a00+a02)(b01+b21) = a00*b01 + a00*b21 + a02*b01 + a02*b21
    # M4 = (a10+a11)(b00+b10) = a10*b00 + a10*b10 + a11*b00 + a11*b10
    # M5 = (a11+a12)(b10+b20) = a11*b10 + a11*b20 + a12*b10 + a12*b20
    # M6 = (a10+a12)(b01+b21) = a10*b01 + a10*b21 + a12*b01 + a12*b21
    #
    # This is 6 terms. To get C00, I need a00*b00 + a01*b10 + a02*b20.
    # From M1: a00*b00 is there. From M2: a01*b10 and a02*b20 are there.
    # But M1 also has a00*b10, a01*b00. M2 also has a01*b20, a02*b10.
    # So I need to cancel these. This suggests I need more terms or a different approach.
    #
    # I'll stick with rank 6 for 2x3x2.
    #
    # Final plan: rank 39
    # Split m=5 into [2,3], p=6 into [2,2,2]
    # For each p-block of size 2:
    #   Compute 2x2 * 2x2 using Strassen (rank 7)
    #   Compute 2x3 * 3x2 using schoolbook (rank 6)
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, coeff in u_coeffs:
            if coeff != 0:
                U[idx][col] += coeff
        for idx, coeff in v_coeffs:
            if coeff != 0:
                V[idx][col] += coeff
        for idx, coeff in w_coeffs:
            if coeff != 0:
                W[idx][col] += coeff
        col += 1
    
    # Strassen 2x2x2: A is 2x2 (rows i=0,1; cols j=0,1), B is 2x2 (rows j=0,1; cols k=0,1)
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # U indices: i*m + j (but within the block, we use local indices)
    # V indices: j*p + k (within block)
    # W indices: k*n + i (within block)
    
    def strassen_2x2x2(i_off, j_off, k_off):
        # i_off: offset for i in U and W
        # j_off: offset for j in U and V
        # k_off: offset for k in V and W
        
        # M1 = (A00 + A11)(B00 + B11)
        add_term(
            [(i_off*2 + 0)*2 + j_off + 0, 1],  # A[0,0]
            [(i_off*2 + 1)*2 + j_off + 1, 1],  # A[1,1]
            [(j_off + 0)*2 + k_off + 0, 1],    # B[0,0]
            [(j_off + 1)*2 + k_off + 1, 1],    # B[1,1]
            [(k_off + 0)*2 + i_off*2 + 0, 1],  # C[0,0]
            [(k_off + 1)*2 + i_off*2 + 1, 1],  # C[1,1]
        )
        
        # M2 = (A10 + A11)B00
        add_term(
            [(i_off*2 + 1)*2 + j_off + 0, 1],  # A[1,0]
            [(i_off*2 + 1)*2 + j_off + 1, 1],  # A[1,1]
            [(j_off + 0)*2 + k_off + 0, 1],    # B[0,0]
            [(k_off + 0)*2 + i_off*2 + 1, 1],  # C[1,0]
            [(k_off + 1)*2 + i_off*2 + 1, -1], # C[1,1]
        )
        
        # M3 = A00(B10 - B11)
        add_term(
            [(i_off*2 + 0)*2 + j_off + 0, 1],  # A[0,0]
            [(j_off + 1)*2 + k_off + 0, 1],    # B[1,0]
            [(j_off + 1)*2 + k_off + 1, -1],   # B[1,1]
            [(k_off + 0)*2 + i_off*2 + 0, 1],  # C[0,0]
            [(k_off + 1)*2 + i_off*2 + 1, 1],  # C[1,1]
        )
        
        # M4 = A11(B01 - B00)
        add_term(
            [(i_off*2 + 1)*2 + j_off + 1, 1],  # A[1,1]
            [(j_off + 0)*2 + k_off + 1, 1],    # B[0,1]
            [(j_off + 0)*2 + k_off + 0, -1],   # B[0,0]
            [(k_off + 0)*2 + i_off*2 + 0, 1],  # C[0,0]
            [(k_off + 0)*2 + i_off*2 + 1, 1],  # C[1,0]
        )
        
        # M5 = (A00 + A01)B11
        add_term(
            [(i_off*2 + 0)*2 + j_off + 0, 1],  # A[0,0]
            [(i_off*2 + 0)*2 + j_off + 1, 1],  # A[0,1]
            [(j_off + 1)*2 + k_off + 1, 1],    # B[1,1]
            [(k_off + 0)*2 + i_off*2 + 0, -1], # C[0,0]
            [(k_off + 1)*2 + i_off*2 + 0, 1],  # C[0,1]
        )
        
        # M6 = (A10 - A00)(B00 + B01)
        add_term(
            [(i_off*2 + 1)*2 + j_off + 0, 1],  # A[1,0]
            [(i_off*2 + 0)*2 + j_off + 0, -1], # A[0,0]
            [(j_off + 0)*2 + k_off + 0, 1],    # B[0,0]
            [(j_off + 0)*2 + k_off + 1, 1],    # B[0,1]
            [(k_off + 1)*2 + i_off*2 + 1, 1],  # C[1,1]
        )
        
        # M7 = (A01 - A11)(B10 + B11)
        add_term(
            [(i_off*2 + 0)*2 + j_off + 1, 1],  # A[0,1]
            [(i_off*2 + 1)*2 + j_off + 1, -1], # A[1,1]
            [(j_off + 1)*2 + k_off + 0, 1],    # B[1,0]
            [(j_off + 1)*2 + k_off + 1, 1],    # B[1,1]
            [(k_off + 0)*2 + i_off*2 + 0, 1],  # C[0,0]
        )
    
    # Schoolbook 2x3x2: A is 2x3, B is 3x2
    def schoolbook_2x3x2(i_off, j_off, k_off):
        for j in range(3):
            for k in range(2):
                add_term(
                    [(i_off*2 + 0)*3 + j_off + j, 1],  # A[0,j]
                    [(i_off*2 + 1)*3 + j_off + j, 1],  # A[1,j]
                    [(j_off + j)*2 + k_off + k, 1],    # B[j,k]
                    [(k_off + k)*2 + i_off*2 + 0, 1],  # C[0,k]
                    [(k_off + k)*2 + i_off*2 + 1, 1],  # C[1,k]
                )
    
    # Now iterate over p-blocks
    # p=6 split into [2,2,2]
    p_blocks = [(0, 2), (2, 4), (4, 6)]
    
    # m=5 split into [2,3]
    # m_block 0: j=0,1 (size 2)
    # m_block 1: j=2,3,4 (size 3)
    
    for k_start, k_end in p_blocks:
        # 2x2 * 2x2 block (m=2, p=2)
        strassen_2x2x2(i_off=0, j_off=0, k_off=k_start)
        # 2x3 * 3x2 block (m=3, p=2)
        schoolbook_2x3x2(i_off=0, j_off=2, k_off=k_start)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
