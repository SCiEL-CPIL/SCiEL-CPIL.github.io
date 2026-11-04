"""Rewrite for matrix multiplication 2x3x7 rank-minimal exact tensor decomposition.

Strategy: Use a hybrid decomposition combining schoolbook multiplication for
parts of the matrix and Strassen's algorithm for the 2x2x2 sub-blocks to
reduce the rank from 42 to 40.

Decomposition:
1. Split B (3x7) into B1 (3x4) and B2 (3x3).
2. Split A (2x3) into A1 (2x2) and A2 (2x1).
3. C = A1*B1 + A2*B1 + A1*B2 + A2*B2.
   - A2*B1 is (2x1)*(3x4) -> effectively (2x1)*(1x4) for the shared index?
     No, A2 is col 2 of A. B1 is rows 0-2, cols 0-3 of B.
     Term A[2, :] * B[:, 2] * B[2, :]? No.
     C[i,k] = sum_j A[i,j] B[j,k].
     
     Let's re-verify the split logic.
     C = A[:, 0:2] * B[0:2, :] + A[:, 2:3] * B[2:3, :]
     
     Part 1: A2 (2x1) * B2 (1x7). Rank 2*1*7 = 14. (Schoolbook)
     Part 2: A1 (2x2) * B1 (2x7).
       Split B1 into B1a (2x4) and B1b (2x3).
       Part 2a: A1 (2x2) * B1b (2x3). Rank 2*2*3 = 12. (Schoolbook)
       Part 2b: A1 (2x2) * B1a (2x4).
         Split B1a into B1a1 (2x2) and B1a2 (2x2).
         Part 2b1: A1 (2x2) * B1a1 (2x2). Rank 7 (Strassen).
         Part 2b2: A1 (2x2) * B1a2 (2x2). Rank 7 (Strassen).
         
     Total Rank: 14 + 12 + 7 + 7 = 40.
"""

DIMENSIONS = [2, 3, 7]

def construct():
    """Return {'U','V','W'} with rank-40 exact tensor decomposition."""
    n, m, p = DIMENSIONS
    rank = 40

    U = [[0] * rank for _ in range(n * m)] # 6 x 40
    V = [[0] * rank for _ in range(m * p)] # 21 x 40
    W = [[0] * rank for _ in range(p * n)] # 14 x 40

    col = 0

    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Part 1: A2 (col 2 of A) * B2 (row 2 of B)
    # A2 is 2x1 (indices i=0,1; j=2). B2 is 1x7 (indices j=2; k=0..6).
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(2):
        for k in range(7):
            add_term(i * 3 + 2, 2 * 7 + k, k * 2 + i, 1, 1, 1)
    # col is now 14

    # Part 2a: A1 (cols 0,1 of A) * B1b (rows 0,1 of B, cols 4,5,6)
    # A1 is 2x2. B1b is 2x3.
    # C[i,k] += sum_{j=0,1} A[i,j] * B[j,k]
    for i in range(2):
        for j in range(2):
            for k in range(4, 7):
                add_term(i * 3 + j, j * 7 + k, k * 2 + i, 1, 1, 1)
    # col is now 14 + 12 = 26

    # Part 2b: A1 (cols 0,1 of A) * B1a (rows 0,1 of B, cols 0,1,2,3)
    # Split into two 2x2x2 Strassen blocks.
    # Block 1: B cols 0,1. Output C cols 0,1.
    # Block 2: B cols 2,3. Output C cols 2,3.
    
    def strassen_block(b_col_offset, w_col_offset):
        nonlocal col
        # A indices: i in 0,1; j in 0,1. U index: i*3 + j.
        # B indices: j in 0,1; k in b_col_offset, b_col_offset+1. V index: j*7 + k.
        # W indices: k in w_col_offset, w_col_offset+1; i in 0,1. W index: k*2 + i.
        
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # We need to add 7 terms for each multiplication M_m.
        # Each M_m is a scalar product of a linear combo of A and a linear combo of B.
        # This results in 7 rank-1 terms? No.
        # A rank-1 term in tensor decomposition is u * v * w.
        # Strassen's algorithm computes C using 7 multiplications.
        # Each multiplication M_m corresponds to a rank-1 component in the bilinear complexity sense?
        # Actually, each M_m is a scalar. The final C entries are sums of these scalars.
        # So we need to represent the 7 multiplications as 7 rank-1 terms in the decomposition of the 2x2x2 tensor.
        
        # Let's define the 7 multiplications.
        # For each mult m, we have:
        # U_vector (length 4, for A00,A01,A10,A11)
        # V_vector (length 4, for B00,B01,B10,B11)
        # W_vector (length 4, for C00,C01,C10,C11)
        
        # M1: A: [1, 0, 0, 1], B: [1, 0, 0, 1]
        #    C00 += 1, C11 += 1
        # M2: A: [0, 0, 1, 1], B: [1, 0, 0, 0]
        #    C10 += 1
        # M3: A: [1, 0, 0, 0], B: [0, 1, 0, -1]
        #    C01 += 1
        # M4: A: [0, 0, 0, 1], B: [0, 0, 1, -1]
        #    C00 += 1, C10 += 1
        # M5: A: [1, 1, 0, 0], B: [0, 0, 0, 1]
        #    C00 -= 1, C01 += 1
        # M6: A: [0, 0, 1, -1], B: [1, 1, 0, 0]
        #    C11 += 1
        # M7: A: [0, 1, 0, -1], B: [0, 0, 1, 1]
        #    C00 += 1, C11 -= 1
        
        # We add 7 terms.
        
        # M1
        add_term(0*3+0, 0*7+b_col_offset, w_col_offset*2+0, 1, 1, 1)
        add_term(1*3+1, 1*7+b_col_offset+1, w_col_offset*2+0, 1, 1, 0) # Wait, W index for C11 is (w_col_offset+1)*2 + 1.
        
        # This manual mapping is error-prone. Let's use a helper to add a full 2x2x2 Strassen block.
        
        pass

    # Let's implement the Strassen block correctly.
    def add_strassen_block(b_col_offset, w_col_offset):
        nonlocal col
        
        # Indices for A (2x2):
        # A00: i=0, j=0 -> U[0]
        # A01: i=0, j=1 -> U[1]
        # A10: i=1, j=0 -> U[3]
        # A11: i=1, j=1 -> U[4]
        
        # Indices for B (2x2):
        # B00: j=0, k=b_col_offset -> V[0*7 + b_col_offset]
        # B01: j=0, k=b_col_offset+1 -> V[0*7 + b_col_offset+1]
        # B10: j=1, k=b_col_offset -> V[1*7 + b_col_offset]
        # B11: j=1, k=b_col_offset+1 -> V[1*7 + b_col_offset+1]
        
        # Indices for C (2x2):
        # C00: i=0, k=w_col_offset -> W[w_col_offset*2 + 0]
        # C01: i=0, k=w_col_offset+1 -> W[w_col_offset*2 + 1]
        # C10: i=1, k=w_col_offset -> W[w_col_offset*2 + 2]
        # C11: i=1, k=w_col_offset+1 -> W[w_col_offset*2 + 3]
        
        # We will add 7 terms.
        
        # M1 = (A00 + A11) * (B00 + B11)
        # Contribution to C: C00 += 1, C11 += 1
        # Term 1: A00 * B00 -> C00
        add_term(0, 0*7+b_col_offset, w_col_offset*2+0, 1, 1, 1)
        # Term 2: A00 * B11 -> C11? No, M1 is one multiplication.
        # The rank-1 decomposition of the 2x2x2 tensor using Strassen has 7 components.
        # Component 1: u1, v1, w1.
        # u1 = [1, 0, 0, 1] (A00, A01, A10, A11)
        # v1 = [1, 0, 0, 1] (B00, B01, B10, B11)
        # w1 = [1, 0, 0, 1] (C00, C01, C10, C11)
        
        # So for M1, we add 1 term to U, V, W?
        # No, U is a vector. U[0] += 1, U[4] += 1.
        # V[0] += 1, V[1*7+b_col_offset+1] += 1.
        # W[w_col_offset*2+0] += 1, W[w_col_offset*2+3] += 1.
        
        # Let's rewrite add_term to handle vectors or just accumulate.
        # Since U, V, W are lists of lists, we can just update the entries.
        
        def update(u_indices, v_indices, w_indices, val):
            for u_idx in u_indices:
                U[u_idx][col] += val
            for v_idx in v_indices:
                V[v_idx][col] += val
            for w_idx in w_indices:
                W[w_idx][col] += val
        
        # M1
        update([0, 4], [0*7+b_col_offset, 1*7+b_col_offset+1], [w_col_offset*2+0, w_col_offset*2+3], 1)
        col += 1
        
        # M2 = (A10 + A11) * B00
        # u2 = [0, 0, 1, 1]
        # v2 = [1, 0, 0, 0]
        # w2 = [0, 0, 1, 0] (C10 += 1)
        update([3, 4], [0*7+b_col_offset], [w_col_offset*2+2], 1)
        col += 1
        
        # M3 = A00 * (B01 - B11)
        # u3 = [1, 0, 0, 0]
        # v3 = [0, 1, 0, -1]
        # w3 = [0, 1, 0, 0] (C01 += 1)
        update([0], [0*7+b_col_offset+1, 1*7+b_col_offset+1], [w_col_offset*2+1], 1)
        # Wait, v3 has -1 for B11.
        V[1*7+b_col_offset+1][col-1] = -1 # Correction for previous line
        col += 0 # col already incremented? No, I did col+=1 after update.
        # Let's restart the column management to be cleaner.
        
        pass

    # The above is getting messy. Let's just construct the 7 terms explicitly for each block.
    
    def add_strassen_block_v2(b_col_offset, w_col_offset):
        nonlocal col
        
        # M1
        U[0][col] = 1; U[4][col] = 1
        V[0*7+b_col_offset][col] = 1; V[1*7+b_col_offset+1][col] = 1
        W[w_col_offset*2+0][col] = 1; W[w_col_offset*2+3][col] = 1
        col += 1
        
        # M2
        U[3][col] = 1; U[4][col] = 1
        V[0*7+b_col_offset][col] = 1
        W[w_col_offset*2+2][col] = 1
        col += 1
        
        # M3
        U[0][col] = 1
        V[0*7+b_col_offset+1][col] = 1; V[1*7+b_col_offset+1][col] = -1
        W[w_col_offset*2+1][col] = 1
        col += 1
        
        # M4
        U[4][col] = 1
        V[1*7+b_col_offset][col] = 1; V[0*7+b_col_offset][col] = -1
        W[w_col_offset*2+0][col] = 1; W[w_col_offset*2+2][col] = 1
        col += 1
        
        # M5
        U[0][col] = 1; U[1][col] = 1
        V[1*7+b_col_offset+1][col] = 1
        W[w_col_offset*2+0][col] = -1; W[w_col_offset*2+1][col] = 1
        col += 1
        
        # M6
        U[3][col] = 1; U[0][col] = -1
        V[0*7+b_col_offset][col] = 1; V[0*7+b_col_offset+1][col] = 1
        W[w_col_offset*2+3][col] = 1
        col += 1
        
        # M7
        U[1][col] = 1; U[4][col] = -1
        V[1*7+b_col_offset][col] = 1; V[1*7+b_col_offset+1][col] = 1
        W[w_col_offset*2+0][col] = 1; W[w_col_offset*2+3][col] = -1
        col += 1

    # Call for Block 1 (B cols 0,1 -> C cols 0,1)
    add_strassen_block_v2(0, 0)
    
    # Call for Block 2 (B cols 2,3 -> C cols 2,3)
    add_strassen_block_v2(2, 2)

    return {"U": U, "V": V, "W": W}
