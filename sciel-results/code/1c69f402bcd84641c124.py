"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    We use a block-Strassen structure:
    - Partition the 4 middle dimension into two 2x2 blocks: A = [A1, A2] (2x2 each)
    - Partition the 5 output columns into a 2x3 block and a 2x2 block
    - For the 2x2 * 2x2 sub-problem, use Strassen (rank 7)
    - For the 2x2 * 2x3 sub-problem, use schoolbook (rank 12) or optimized
    
    Actually, let's think more carefully. The matrix is A (2x4) * B (4x5) = C (2x5).
    
    We can write B as [B_left (4x2), B_right (4x3)] where B_left is the first 2 columns,
    B_right is the last 3 columns.
    
    C = A * B = [A * B_left (2x2), A * B_right (2x3)]
    
    For A * B_left: A is 2x4, B_left is 4x2. We can partition A into [A1, A2] where A1, A2 are 2x2.
    A * B_left = A1 * B1_left + A2 * B2_left where B1_left is top 2x2 of B_left, B2_left is bottom 2x2.
    Each 2x2 * 2x2 can be done with Strassen (rank 7), so this part costs 14.
    
    For A * B_right: A is 2x4, B_right is 4x3. Partition A into [A1, A2] (2x2 each).
    A * B_right = A1 * B1_right + A2 * B2_right where B1_right, B2_right are 2x3.
    Each 2x2 * 2x3 is schoolbook rank 12, so this part costs 24.
    
    Total: 14 + 24 = 38. That's better than 40.
    
    But we can do better. Let's use Strassen for the 2x2 * 2x3 case too.
    Actually, 2x2 * 2x3: we can write it as 2x2 * 2x2 + 2x2 * 2x1.
    The 2x2 * 2x1 part is rank 4 (schoolbook). So 2x2 * 2x3 = 7 + 4 = 11.
    So A * B_right costs 2 * 11 = 22.
    
    Total: 14 + 22 = 36.
    
    Let me verify this is correct. Actually, let me just implement the block-Strassen approach
    and see what rank we get.
    """
    import time
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # We'll build the decomposition using block-Strassen
    # A is n x m = 2 x 4, B is m x p = 4 x 5, C is n x p = 2 x 5
    
    # Partition strategy:
    # Split B's columns: first 2 columns (B1) and last 3 columns (B2)
    # Split A's rows (middle dim): first 2 (A_top) and last 2 (A_bot)
    
    # For C = A * B:
    # C[:, :2] = A * B[:, :2] = A_top * B_top1 + A_bot * B_bot1  (each 2x2*2x2, Strassen rank 7)
    # C[:, 2:] = A * B[:, 2:] = A_top * B_top2 + A_bot * B_bot2  (each 2x2*2x3)
    
    # For 2x2 * 2x3, we can split the 3 columns into 2+1:
    # 2x2 * 2x3 = 2x2 * 2x2 + 2x2 * 2x1
    # Strassen for 2x2*2x2 (rank 7) + schoolbook for 2x2*2x1 (rank 4) = rank 11
    
    # Total rank: 2*7 + 2*(7+4) = 14 + 22 = 36
    
    # Let's construct this explicitly
    # We need to track which entries of A, B, C are involved in each rank-1 term
    
    # Helper: Strassen for 2x2 * 2x2
    # Given matrices X (2x2), Y (2x2), compute Z = X*Y using 7 products
    # X = [[x11, x12], [x21, x22]], Y = [[y11, y12], [y21, y22]]
    # Z = [[z11, z12], [z21, z22]]
    # 
    # Strassen:
    # m1 = (x11+x22)*(y11+y22)
    # m2 = (x21+x22)*y11
    # m3 = x11*(y12-y22)
    # m4 = x22*(y21-y11)
    # m5 = (x11+x12)*y22
    # m6 = (x21-x11)*(y11+y12)
    # m7 = (x12-x22)*(y21+y22)
    # 
    # z11 = m1 + m4 - m5 + m7
    # z12 = m3 + m5
    # z21 = m2 + m4
    # z22 = m1 - m2 + m3 + m6
    
    # For each rank-1 term in Strassen, we need to express it as a product of 
    # linear forms in A and B, contributing to linear forms in C.
    
    # Let's define the rank-1 terms as tuples (u_coeffs, v_coeffs, w_coeffs)
    # where u_coeffs is a dict mapping (i,j) -> coeff for A[i,j]
    # v_coeffs is a dict mapping (j,k) -> coeff for B[j,k]  
    # w_coeffs is a dict mapping (i,k) -> coeff for C[i,k]
    
    # But we need to be careful about the indexing convention.
    # U has shape (n*m, rank) = (8, rank), indexed by [i*m + j]
    # V has shape (m*p, rank) = (20, rank), indexed by [j*p + k]
    # W has shape (p*n, rank) = (10, rank), indexed by [k*n + i]
    
    # So for A[i,j], the row index in U is i*m + j = i*4 + j
    # For B[j,k], the row index in V is j*p + k = j*5 + k
    # For C[i,k], the column index in W is k*n + i = k*2 + i
    
    # Let's build the decomposition step by step
    
    rank_terms = []  # Each term is (u_dict, v_dict, w_dict)
    
    def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
        """
        Add Strassen terms for multiplying two 2x2 matrices.
        a_offset: (i0, j0) - top-left corner of the 2x2 block of A
        b_offset: (j0, k0) - top-left corner of the 2x2 block of B  
        c_offset: (i0, k0) - top-left corner of the 2x2 block of C
        sign: multiplier for all coefficients
        """
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        # A block: A[i0:i0+2, j0:j0+2]
        # B block: B[j0:j0+2, k0:k0+2]
        # C block: C[i0:i0+2, k0:k0+2]
        
        # Define the 7 Strassen products
        # m1 = (x11+x22)*(y11+y22)
        m1_u = {(i0, j0): 1, (i0+1, j0+1): 1}
        m1_v = {(j0, k0): 1, (j0+1, k0+1): 1}
        m1_w = {(i0, k0): 1, (i0+1, k0+1): -1, (i0+1, k0): 1, (i0, k0+1): 1}
        # z11 = m1 + m4 - m5 + m7, z22 = m1 - m2 + m3 + m6
        # So m1 contributes: +1 to z11, +1 to z22
        # Wait, let me redo this.
        
        # z11 = m1 + m4 - m5 + m7
        # z12 = m3 + m5
        # z21 = m2 + m4
        # z22 = m1 - m2 + m3 + m6
        
        # m1 contributes: z11 += 1, z22 += 1
        # m2 contributes: z21 += 1, z22 -= 1
        # m3 contributes: z12 += 1, z22 += 1
        # m4 contributes: z11 += 1, z21 += 1
        # m5 contributes: z11 -= 1, z12 += 1
        # m6 contributes: z22 += 1
        # m7 contributes: z11 += 1
        
        terms = [
            # (u_dict, v_dict, w_dict)
            # m1
            ({(i0, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            # m2
            ({(i0+1, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1}, 
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            # m3
            ({(i0, j0): 1}, 
             {(j0, k0+1): 1, (j0+1, k0+1): -1}, 
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            # m4
            ({(i0+1, j0+1): 1}, 
             {(j0+1, k0): 1, (j0, k0): -1}, 
             {(i0, k0): 1, (i0+1, k0): 1}),
            # m5
            ({(i0, j0): 1, (i0, j0+1): 1}, 
             {(j0+1, k0+1): 1}, 
             {(i0, k0): -1, (i0, k0+1): 1}),
            # m6
            ({(i0+1, j0): 1, (i0, j0): -1}, 
             {(j0, k0): 1, (j0, k0+1): 1}, 
             {(i0+1, k0+1): 1}),
            # m7
            ({(i0, j0+1): 1, (i0+1, j0+1): -1}, 
             {(j0+1, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
    
    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
        """
        Add schoolbook terms for multiplying 2x2 * 2x1.
        """
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        # A is 2x2: A[i0:i0+2, j0:j0+2]
        # B is 2x1: B[j0:j0+2, k0]
        # C is 2x1: C[i0:i0+2, k0]
        
        # C[i0, k0] = A[i0, j0]*B[j0, k0] + A[i0, j0+1]*B[j0+1, k0]
        # C[i0+1, k0] = A[i0+1, j0]*B[j0, k0] + A[i0+1, j0+1]*B[j0+1, k0]
        
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
    
    # Now build the full decomposition
    # A is 2x4: indices i in {0,1}, j in {0,1,2,3}
    # B is 4x5: indices j in {0,1,2,3}, k in {0,1,2,3,4}
    # C is 2x5: indices i in {0,1}, k in {0,1,2,3,4}
    
    # Split B's columns: B[:, 0:2] and B[:, 2:5]
    # Split A's middle dim: A[:, 0:2] and A[:, 2:4]
    
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    # C[:, 2:5] = A[:, 0:2] * B[0:2, 2:5] + A[:, 2:4] * B[2:4, 2:5]
    
    # For C[:, 0:2]: two 2x2 * 2x2 multiplications, each Strassen (rank 7)
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))  # A_top * B_top_left
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))  # A_bot * B_bot_left
    
    # For C[:, 2:5]: two 2x2 * 2x3 multiplications
    # 2x2 * 2x3 = 2x2 * 2x2 (Strassen, rank 7) + 2x2 * 2x1 (schoolbook, rank 4)
    
    # A_top * B_top_right (2x2 * 2x3, columns 2,3,4)
    # Split into 2x2 * 2x2 (cols 2,3) + 2x2 * 2x1 (col 4)
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))  # A_top * B[0:2, 2:4]
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))  # A_top * B[0:2, 4:5]
    
    # A_bot * B_bot_right (2x2 * 2x3, columns 2,3,4)
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))  # A_bot * B[2:4, 2:4]
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))  # A_bot * B[2:4, 4:5]
    
    # Total rank: 7 + 7 + 7 + 4 + 7 + 4 = 36
    rank = len(rank_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
