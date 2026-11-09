"""Matrix multiplication 2x4 by 4x4 - rank-minimal decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x4 * 4x4 matrix multiplication into rank-28 tensor factors.
    
    We treat the 4x4 intermediate as four 2x2 blocks (by columns of B),
    and the 2x4 * 4x4 as four independent 2x2 * 2x2 multiplications
    (one for each 2x2 column-block of B).
    
    For each of the 4 blocks, we use Strassen's rank-7 algorithm for 2x2x2.
    Total rank = 4 * 7 = 28.
    
    U shape: (n*m, rank) = (2*4, 28) = (8, 28)
    V shape: (m*p, rank) = (4*4, 28) = (16, 28)
    W shape: (p*n, rank) = (4*2, 28) = (8, 28)
    
    Indexing:
    U[i*m + j, c] where i in {0,1}, j in {0,1,2,3}
    V[j*p + k, c] where j in {0,1,2,3}, k in {0,1,2,3}
    W[k*n + i, c] where k in {0,1,2,3}, i in {0,1}
    
    For block b (b in {0,1,2,3}), the 2x2 sub-matrix of A uses rows:
      A_row0: i=0, j = 2*b, 2*b+1
      A_row1: i=1, j = 2*b, 2*b+1
    The 2x2 sub-matrix of B uses columns:
      B_col0: k = 2*b, 2*b+1 (wait, no...)
    
    Actually, let me re-think the block structure.
    
    C = A * B where A is 2x4 and B is 4x4.
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    We can partition by the column index k of C (which ranges 0..3).
    For a fixed k, C[:, k] = A * B[:, k] where B[:, k] is a 4x1 vector.
    
    Alternatively, partition by j. For j in {0,1,2,3}:
    C[:, :] += A[:, j] * B[j, :]
    
    That's the schoolbook view with rank 2*4*4 = 32.
    
    To use Strassen, we need 2x2x2 structure. Let's partition the j-index into
    pairs: (j0, j1) = (2b, 2b+1) for b in {0,1,2,3}.
    
    For block b, define:
    A_b = A[:, [2b, 2b+1]]  (2x2 matrix)
    B_b = B[[2b, 2b+1], :]   (2x4 matrix)
    
    Then C += A_b * B_b.
    
    Now A_b is 2x2 and B_b is 2x4. We can further partition B_b by columns:
    B_b = [B_{b,0}, B_{b,1}] where B_{b,c} is 2x2 for c in {0,1},
    corresponding to output columns k = 2c, 2c+1.
    
    So for each (b, c), we have a 2x2 * 2x2 multiplication:
    C[:, [2c, 2c+1]] += A_b * B_{b,c}
    
    That's 4*2 = 8 independent 2x2x2 multiplications, each of rank 7.
    Total rank = 8 * 7 = 56. That's worse than schoolbook!
    
    Hmm, that's not right. Let me reconsider.
    
    Actually, A is 2x4 and B is 4x4. The natural block structure for Strassen
    would be to view A as [A_left | A_right] where each is 2x2, and B as
    [[B_TL, B_TR], [B_BL, B_BR]] where each block is 2x2.
    
    Then C = A * B = [A_left * B_TL + A_right * B_BL, A_left * B_TR + A_right * B_BR]
    where C is 2x4 = [C_left | C_right] with each 2x2.
    
    So C_left = A_left * B_TL + A_right * B_BL
    C_right = A_left * B_TR + A_right * B_BR
    
    Each of these is a sum of two 2x2x2 products. Using Strassen for each:
    - C_left needs two Strassen decompositions: one for A_left*B_TL, one for A_right*B_BL
    - C_right needs two Strassen decompositions: one for A_left*B_TR, one for A_right*B_BR
    
    Total: 4 * 7 = 28. 
    
    Let me verify the indexing:
    A is 2x4. A_left = A[:, 0:2], A_right = A[:, 2:4].
    B is 4x4. B_TL = B[0:2, 0:2], B_TR = B[0:2, 2:4], B_BL = B[2:4, 0:2], B_BR = B[2:4, 2:4].
    
    C is 2x4. C_left = C[:, 0:2], C_right = C[:, 2:4].
    
    C_left = A_left @ B_TL + A_right @ B_BL
    C_right = A_left @ B_TR + A_right @ B_BR
    
    Each term is a 2x2 * 2x2 product. Using Strassen (rank 7) for each:
    Total rank = 4 * 7 = 28.
    
    Now let's construct the factors.
    
    Strassen 2x2x2 decomposition (rank 7):
    For X (2x2) * Y (2x2) = Z (2x2):
    
    Local U indices (X indices): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    Local V indices (Y indices): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    Local W indices (Z indices): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    (W[k*n+i] with k=row, i=col in 2x2: W_0=(0,0), W_1=(0,1), W_2=(1,0), W_3=(1,1))
    
    Wait, the task says W[k*n+i, r] where the output is p x n = 4x2.
    So W has shape (4*2, rank) = (8, rank).
    W[k*n + i, c] where k in {0,1,2,3} (column of C) and i in {0,1} (row of C).
    
    For a 2x2 output block at rows i in {0,1} and columns k in {k0, k0+1}:
    Local W index (a, b) maps to global W index (k0+b) * 2 + a.
    Where a is the local row (0 or 1) and b is the local column (0 or 1).
    
    Strassen coefficients for 2x2x2:
    Let X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]].
    
    M1 = (x00 + x11)(y00 + y11)
    M2 = (x10 + x11) y00
    M3 = x00 (y01 - y11)
    M4 = x11 (y10 - y00)
    M5 = (x00 + x01) y11
    M6 = (x10 - x00)(y00 + y01)
    M7 = (x01 - x11)(y10 + y11)
    
    Z00 = M1 + M4 - M5 + M7
    Z01 = M3 + M5
    Z10 = M2 + M4
    Z11 = M1 - M2 + M3 + M6
    
    So the U, V, W coefficients for each of the 7 terms:
    
    Term 1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    Term 2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,-1]
    Term 3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,1]
    Term 4: U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,1,0]
    Term 5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]
    Term 6: U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
    Term 7: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]
    
    Now for each of the 4 blocks:
    Block (side, col_pair): side in {left, right}, col_pair in {0, 1}
    
    For side=left (A_left), the A indices are:
      Local (0,0) -> global U index 0*4 + 0 = 0
      Local (0,1) -> global U index 0*4 + 1 = 1
      Local (1,0) -> global U index 1*4 + 0 = 4
      Local (1,1) -> global U index 1*4 + 1 = 5
    
    For side=right (A_right), the A indices are:
      Local (0,0) -> global U index 0*4 + 2 = 2
      Local (0,1) -> global U index 0*4 + 3 = 3
      Local (1,0) -> global U index 1*4 + 2 = 6
      Local (1,1) -> global U index 1*4 + 3 = 7
    
    For B blocks:
    B_TL (col_pair=0, row_pair=0):
      Local (0,0) -> V[0*4+0] = 0
      Local (0,1) -> V[0*4+1] = 1
      Local (1,0) -> V[0*4+2] = 2
      Local (1,1) -> V[0*4+3] = 3
    
    B_TR (col_pair=1, row_pair=0):
      Local (0,0) -> V[0*4+2] = 2
      Local (0,1) -> V[0*4+3] = 3
      Local (1,0) -> V[0*4+4] = 4
      Local (1,1) -> V[0*4+5] = 5
    
    B_BL (col_pair=0, row_pair=1):
      Local (0,0) -> V[2*4+0] = 8
      Local (0,1) -> V[2*4+1] = 9
      Local (1,0) -> V[2*4+2] = 10
      Local (1,1) -> V[2*4+3] = 11
    
    B_BR (col_pair=1, row_pair=1):
      Local (0,0) -> V[2*4+2] = 10
      Local (0,1) -> V[2*4+3] = 11
      Local (1,0) -> V[2*4+4] = 12
      Local (1,1) -> V[2*4+5] = 13
    
    For C blocks:
    C_left, col_pair=0 (columns 0,1):
      Local (0,0) -> W[0*2+0] = 0
      Local (0,1) -> W[0*2+1] = 1
      Local (1,0) -> W[1*2+0] = 2
      Local (1,1) -> W[1*2+1] = 3
    
    C_left, col_pair=1 (columns 2,3):
      Local (0,0) -> W[2*2+0] = 4
      Local (0,1) -> W[2*2+1] = 5
      Local (1,0) -> W[3*2+0] = 6
      Local (1,1) -> W[3*2+1] = 7
    
    So the 4 blocks are:
    1. A_left * B_TL -> C[:, 0:2] (col_pair=0)
    2. A_right * B_BL -> C[:, 0:2] (col_pair=0)
    3. A_left * B_TR -> C[:, 2:4] (col_pair=1)
    4. A_right * B_BR -> C[:, 2:4] (col_pair=1)
    
    Let's build the factors.
    """
    import numpy as np
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)  # (8, 28)
    V = np.zeros((m * p, rank), dtype=float)  # (16, 28)
    W = np.zeros((p * n, rank), dtype=float)  # (8, 28)
    
    # Strassen 2x2x2 coefficients
    strassen_U = np.array([
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ])  # shape (7, 4)
    
    strassen_V = np.array([
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [0, 0, 1, -1],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ])  # shape (7, 4)
    
    strassen_W = np.array([
        [1, 0, 0, 1],   # M1: Z00+=1, Z11+=1
        [0, 0, 1, -1],  # M2: Z10+=1, Z11-=1
        [0, 1, 0, 1],   # M3: Z01+=1, Z11+=1
        [1, 0, 1, 0],   # M4: Z00+=1, Z10+=1
        [-1, 1, 0, 0],  # M5: Z00-=1, Z01+=1
        [0, 0, 0, 1],   # M6: Z11+=1
        [1, 0, 0, 0],   # M7: Z00+=1
    ])  # shape (7, 4)
    
    # Define the 4 blocks: (A_side, B_block, C_col_pair)
    # A_side: 0=left (cols 0,1 of A), 1=right (cols 2,3 of A)
    # B_block: 0=TL, 1=TR, 2=BL, 3=BR
    # C_col_pair: 0=cols 0,1 of C, 1=cols 2,3 of C
    
    blocks = [
        # (A_side, B_block, C_col_pair)
        (0, 0, 0),  # A_left * B_TL -> C[:, 0:2]
        (1, 2, 0),  # A_right * B_BL -> C[:, 0:2]
        (0, 1, 1),  # A_left * B_TR -> C[:, 2:4]
        (1, 3, 1),  # A_right * B_BR -> C[:, 2:4]
    ]
    
    def a_local_to_global(side, local_idx):
        """Map local A index (0..3) to global U index."""
        # Local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # side=0: cols 0,1; side=1: cols 2,3
        local_row = local_idx // 2
        local_col = local_idx % 2
        global_row = local_row
        global_col = side * 2 + local_col
        return global_row * m + global_col
    
    def b_local_to_global(b_block, local_idx):
        """Map local B index (0..3) to global V index."""
        # Local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # b_block: 0=TL, 1=TR, 2=BL, 3=BR
        local_row = local_idx // 2
        local_col = local_idx % 2
        # Row pair: 0 for TL/TR (rows 0,1), 1 for BL/BR (rows 2,3)
        row_pair = b_block // 2  # 0 for b_block 0,1; 1 for b_block 2,3
        # Col pair: 0 for TL/BL (cols 0,1), 1 for TR/BR (cols 2,3)
        col_pair = b_block % 2
        global_row = row_pair * 2 + local_row
        global_col = col_pair * 2 + local_col
        return global_row * p + global_col
    
    def c_local_to_global(c_col_pair, local_idx):
        """Map local C index (0..3) to global W index."""
        # Local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # W[k*n + i] where k=col, i=row
        local_row = local_idx // 2  # This is actually... wait.
        # W local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # In Strassen, W indices correspond to Z rows/cols.
        # Let me use: local_idx = z_row * 2 + z_col
        z_row = local_idx // 2
        z_col = local_idx % 2
        global_row = z_row  # i in {0,1}
        global_col = c_col_pair * 2 + z_col  # k
        return global_col * n + global_row
    
    col = 0
    for (a_side, b_block, c_col_pair) in blocks:
        for t in range(7):
            for li in range(4):
                if strassen_U[t, li] != 0:
                    U[a_local_to_global(a_side, li), col] = strassen_U[t, li]
            for li in range(4):
                if strassen_V[t, li] != 0:
                    V[b_local_to_global(b_block, li), col] = strassen_V[t, li]
            for li in range(4):
                if strassen_W[t, li] != 0:
                    W[c_local_to_global(c_col_pair, li), col] = strassen_W[t, li]
            col += 1
    
    # Convert to lists of lists as expected by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
