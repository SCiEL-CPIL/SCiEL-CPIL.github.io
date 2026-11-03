"""Winograd Fourier-based matrix multiplication for 2x4 * 4x5."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5

    # We'll build the decomposition using Winograd-style transforms.
    # For 2x2*2x2, Winograd gives rank 4 (vs Strassen's 7, schoolbook's 4).
    # Actually Winograd F(2) for 2x2*2x2 has rank 4.
    # For 2x2*2x1, rank is 2 (Winograd) vs 4 (schoolbook).
    # For 2x2*2x3, we can use Winograd F(2) on the 2x2 part and handle the 3rd column.
    
    # Strategy: Split the 2x4 * 4x5 into blocks.
    # A is 2x4, B is 4x5.
    # Split A into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3).
    # Split B into B_top (4x? no, B is 4x5).
    # B rows 0-1 go with A_left, B rows 2-3 go with A_right.
    # So C = A_left * B_top + A_right * B_bottom
    # where B_top is B[0:2, :] (2x5) and B_bottom is B[2:4, :] (2x5).
    
    # For each 2x2 * 2x5 product, we further split the 2x5 into 2x2 + 2x2 + 2x1
    # or 2x3 + 2x2, etc.
    
    # Let's use Winograd F(2) for 2x2*2x2 (rank 4) and handle 2x2*2x1 (rank 2).
    # Total: 2 blocks * (2 * rank(2x2*2x2) + 1 * rank(2x2*2x1))
    # = 2 * (2*4 + 1*2) = 2 * 10 = 20.
    
    # But wait, can we do better with 2x2*2x3?
    # Winograd for 2x2*2x3: treat as 2x2*2x2 + 2x2*2x1, rank 4+2=6.
    # Or a direct Winograd F(2) for 2x2*2x3? That would be rank 4 (same as 2x2*2x2) if we can interpolate 3 points from 2x2... no.
    # Actually for 2x2*2xN, the Winograd rank is 4 for any N>=2 (using the same 2x2 transform on B columns).
    # Hmm, no. The Winograd algorithm for MxN * NxP uses transforms on A (size MxN) and B (NxP).
    # For 2x2 * 2x5: A is 2x2, B is 2x5.
    # Winograd: transform A to size 2x2 (identity for 2x2 with step 2), transform each column of B.
    # Actually the standard Winograd F(2) for 2x2*2x2 has rank 4.
    # For 2x2*2x5, we can apply the same 4 multiplications to each pair of B columns? No.
    
    # Let me think more carefully about Winograd for 2x2 * 2x5.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, b02, b03, b04], [b10, b11, b12, b13, b14]]
    # C = A*B, 2x5
    # C[i][k] = sum_j A[i][j] * B[j][k]
    
    # Winograd F(2) on the j-dimension:
    # Transform A: A' = [[a00+a10, a01+a11], [a00-a10, a01-a11]]? No.
    # Actually Winograd for 2x2*2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00-b10)  -- wait, let me use the standard form.
    
    # Standard Winograd 2x2*2x2 (rank 4):
    # Let A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # m1 = (a0+a3)(b0+b3)
    # m2 = (a2+a3)(b0-b2)
    # m3 = a0(b1-b3)
    # m4 = (a0+a1)(b1-b3)  -- hmm, I need to be more careful.
    
    # Actually the minimal rank for 2x2*2x2 is 7 (Strassen). Winograd F(2) gives rank 4 for 1D convolutions but for 2x2*2x2 matrix mult, the rank is 7.
    # Wait, no. Winograd's algorithm for 2x2*2x2 matrix multiplication has rank 4? Let me reconsider.
    
    # Actually, the tensor rank of 2x2*2x2 matrix multiplication is 7 (Strassen's bound). Winograd's algorithm is for convolutions, not matrix multiplication. For matrix multiplication, the minimum known rank for 2x2*2x2 is 7.
    
    # So I was wrong. Let me reconsider the approach.
    
    # For 2x4 * 4x5, the naive approach has rank 2*4*5 = 40.
    # The current program achieves rank 36 using Strassen (rank 7) for 2x2*2x2 and schoolbook (rank 4) for 2x2*2x1.
    
    # Can we do better? The theoretical lower bound is ceil(2*4*5 / something) ... not helpful.
    # Known results: R(2,4,5) <= 36 (from the current approach). Can we get 35 or lower?
    
    # Let me try a different block decomposition.
    # What if we use Strassen for 2x2*2x2 (rank 7) but find a better way for the 2x2*2x1 parts?
    # 2x2*2x1 has rank 4 (it's just 4 scalar multiplications, and it's known to be rank 4).
    
    # Alternative: Can we combine the 2x2*2x1 with adjacent 2x2*2x2 to save rank?
    
    # Let me try: 2x4*4x5. Split A into two 2x2 blocks, B into two 2x5 blocks.
    # Each 2x2*2x5: split B into 2x2+2x2+2x1.
    # Rank per 2x2*2x5 = 7+7+4 = 18.
    # Total = 2*18 = 36.
    
    # What if we split B into 2x3+2x2?
    # 2x2*2x3: what's the rank? It's a 2x2*2x3 tensor. 
    # C[i][k] = A[i][0]*B[0][k] + A[i][1]*B[1][k] for k=0,1,2.
    # This has 2*2*3 = 12 entries, rank at most 12.
    # Using Strassen-like: treat as 2x2*2x2 (rank 7) + 2x2*2x1 (rank 4) = 11.
    # Or directly: 2x2*2x3 can be done with rank 8? Let me check.
    # Actually, I think 2x2*2x3 has rank 8.
    
    # If 2x2*2x3 has rank 8, then 2x2*2x5 = 2x2*2x3 + 2x2*2x2 = 8+7 = 15.
    # Total = 2*15 = 30. That would be great!
    
    # Let me verify: can 2x2*2x3 be done with rank 8?
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1, b2], [b3, b4, b5]]
    # C[0][0] = a0*b0 + a1*b3
    # C[0][1] = a0*b1 + a1*b4
    # C[0][2] = a0*b2 + a1*b5
    # C[1][0] = a2*b0 + a3*b3
    # C[1][1] = a2*b1 + a3*b4
    # C[1][2] = a2*b2 + a3*b5
    
    # This is like two independent 2x1*2x3 problems... no, it's interleaved.
    # Actually it's a 2x2 matrix times a 2x3 matrix.
    # Rank of this tensor: I believe it's 8.
    
    # Let me construct a rank-8 decomposition for 2x2*2x3.
    # Using the fact that it's equivalent to 2 separate 2x1*2x3 (each rank 4? no, 2x1*2x3 has rank 4).
    # Wait, 2x1*2x3: C[0][k] = a0*B[0][k] + a1*B[1][k]. This has rank 2 (for each k independently) but as a tensor, it's rank 2? No.
    # 2x1*2x3: the tensor has entries T[i][j][k] = A[i] * B[j][k] where i in {0}, j in {0,1}, k in {0,1,2}.
    # This is just A[i] * B[j][k], which is rank 1 as a 3-tensor? No, A has 1 element, B has 6 elements.
    # The tensor is sum over j of A[i]*B[j][k], which is rank... it's a 1x6 tensor, rank 1.
    # Hmm, I'm confusing myself.
    
    # Let me just go with the known rank 36 approach but try to optimize.
    # Actually, let me try rank 34 by using a slightly different block structure.
    
    # New idea: Use the fact that 2x4*4x5 can be decomposed as:
    # C = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :]
    # Each term is 2x2 * 2x5.
    # For 2x2 * 2x5, use Strassen for the first 4 columns (2x2*2x4 = two 2x2*2x2) and schoolbook for the last column.
    # 2x2*2x4 = 2 * 7 = 14. Plus 2x2*2x1 = 4. Total 18 per 2x2*2x5.
    # Grand total: 2 * 18 = 36.
    
    # I'll stick with rank 36 but implement it more cleanly with a Winograd-inspired structure.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1
    
    def add_strassen_2x2(a_i0, a_j0, b_i0, b_j0, c_i0, c_j0):
        """Add Strassen decomposition for 2x2*2x2 block.
        A block at rows a_i0..a_i0+1, cols a_j0..a_j0+1
        B block at rows b_i0..b_i0+1, cols b_j0..b_j0+1
        C block at rows c_i0..c_i0+1, cols c_j0..c_j0+1
        """
        def ma(loc):  # A index: local 0-3 -> global
            r, c = loc // 2, loc % 2
            return (a_i0 + r) * m + (a_j0 + c)
        def mb(loc):  # B index
            r, c = loc // 2, loc % 2
            return (b_i0 + r) * p + (b_j0 + c)
        def mc(loc):  # C index (W convention: k*n+i)
            r, c = loc // 2, loc % 2
            return (c_j0 + c) * n + (c_i0 + r)
        
        # Strassen 7 multiplications
        # M1 = (A00+A11)(B00+B11)
        add_rank1_term({ma(0): 1, ma(3): 1}, {mb(0): 1, mb(3): 1}, {mc(0): 1, mc(3): 1})
        # M2 = (A10+A11)B00
        add_rank1_term({ma(2): 1, ma(3): 1}, {mb(0): 1}, {mc(2): 1, mc(3): -1})
        # M3 = A00(B01-B11)
        add_rank1_term({ma(0): 1}, {mb(1): 1, mb(3): -1}, {mc(1): 1, mc(3): 1})
        # M4 = A11(B10-B00)
        add_rank1_term({ma(3): 1}, {mb(2): 1, mb(0): -1}, {mc(0): 1, mc(2): 1})
        # M5 = (A00+A01)B11
        add_rank1_term({ma(0): 1, ma(1): 1}, {mb(3): 1}, {mc(0): -1, mc(1): 1})
        # M6 = (A10-A00)(B00+B01)
        add_rank1_term({ma(2): 1, ma(0): -1}, {mb(0): 1, mb(1): 1}, {mc(3): 1})
        # M7 = (A01-A11)(B10+B11)
        add_rank1_term({ma(1): 1, ma(3): -1}, {mb(2): 1, mb(3): 1}, {mc(0): 1})
    
    def add_schoolbook_2x2_1(a_i0, a_j0, b_i0, b_j0, c_i0, c_j0):
        """Add schoolbook decomposition for 2x2*2x1 block."""
        def ma(loc):
            r, c = loc // 2, loc % 2
            return (a_i0 + r) * m + (a_j0 + c)
        def mb(loc):
            return (b_i0 + loc) * p + b_j0
        def mc(loc):
            return c_j0 * n + (c_i0 + loc)
        
        # C[0] = A[0][0]*B[0] + A[0][1]*B[1]
        # C[1] = A[1][0]*B[0] + A[1][1]*B[1]
        add_rank1_term({ma(0): 1}, {mb(0): 1}, {mc(0): 1})
        add_rank1_term({ma(1): 1}, {mb(1): 1}, {mc(0): 1})
        add_rank1_term({ma(2): 1}, {mb(0): 1}, {mc(1): 1})
        add_rank1_term({ma(3): 1}, {mb(1): 1}, {mc(1): 1})
    
    # Block 1: A[:, 0:2] * B[0:2, :]
    add_strassen_2x2(0, 0, 0, 0, 0, 0)   # A[0:2,0:2] * B[0:2,0:2]
    add_strassen_2x2(0, 0, 0, 2, 0, 2)   # A[0:2,0:2] * B[0:2,2:4]
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)  # A[0:2,0:2] * B[0:2,4]
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    add_strassen_2x2(0, 2, 2, 0, 0, 0)   # A[0:2,2:4] * B[2:4,0:2]
    add_strassen_2x2(0, 2, 2, 2, 0, 2)   # A[0:2,2:4] * B[2:4,2:4]
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)  # A[0:2,2:4] * B[2:4,4]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
