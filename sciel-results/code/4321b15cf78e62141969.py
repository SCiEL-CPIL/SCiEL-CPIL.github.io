import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Define 2x2x2 Strassen decomposition
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # n=2, m=2, p=2
    # U is 4x7, V is 4x7, W is 4x7
    
    # Strassen formulas for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to express this as sum of rank-1 tensors.
    # Tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Note: The task says W[k*n+i, r].
    # For n=2, W[k*2+i, r].
    
    # Let's construct U, V, W for 2x2x2.
    # Indices: i,j,k in 0,1.
    # U row index: i*2+j.
    # V row index: j*2+k.
    # W row index: k*2+i.
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0: M1
    # A part: A11 + A22 -> U[0,0]=1, U[3,0]=1 (i=0,j=0 and i=1,j=1)
    # B part: B11 + B22 -> V[0,0]=1, V[3,0]=1 (j=0,k=0 and j=1,k=1)
    # C part: C11 + C22 -> W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
    U2[0,0] = 1; U2[3,0] = 1
    V2[0,0] = 1; V2[3,0] = 1
    W2[0,0] = 1; W2[3,0] = 1
    
    # r=1: M2
    # A part: A21 + A22 -> U[2,1]=1 (i=1,j=0), U[3,1]=1 (i=1,j=1)
    # B part: B11 -> V[0,1]=1 (j=0,k=0)
    # C part: C21 -> W[1,1]=1 (k=0,i=1)
    U2[2,1] = 1; U2[3,1] = 1
    V2[0,1] = 1
    W2[1,1] = 1
    
    # r=2: M3
    # A part: A11 -> U[0,2]=1
    # B part: B12 - B22 -> V[1,2]=1 (j=0,k=1), V[3,2]=-1 (j=1,k=1)
    # C part: C12 -> W[2,2]=1 (k=1,i=0)
    U2[0,2] = 1
    V2[1,2] = 1; V2[3,2] = -1
    W2[2,2] = 1
    
    # r=3: M4
    # A part: A22 -> U[3,3]=1
    # B part: B21 - B11 -> V[2,3]=1 (j=1,k=0), V[0,3]=-1 (j=0,k=0)
    # C part: C11 + C21 -> W[0,3]=1 (k=0,i=0), W[1,3]=1 (k=0,i=1)
    U2[3,3] = 1
    V2[2,3] = 1; V2[0,3] = -1
    W2[0,3] = 1; W2[1,3] = 1
    
    # r=4: M5
    # A part: A11 + A12 -> U[0,4]=1, U[1,4]=1
    # B part: B22 -> V[3,4]=1
    # C part: -C11 + C12 -> W[0,4]=-1 (k=0,i=0), W[2,4]=1 (k=1,i=0)
    U2[0,4] = 1; U2[1,4] = 1
    V2[3,4] = 1
    W2[0,4] = -1; W2[2,4] = 1
    
    # r=5: M6
    # A part: A21 - A11 -> U[2,5]=1, U[0,5]=-1
    # B part: B11 + B12 -> V[0,5]=1, V[1,5]=1
    # C part: C22 -> W[3,5]=1 (k=1,i=1)
    U2[2,5] = 1; U2[0,5] = -1
    V2[0,5] = 1; V2[1,5] = 1
    W2[3,5] = 1
    
    # r=6: M7
    # A part: A12 - A22 -> U[1,6]=1, U[3,6]=-1
    # B part: B21 + B22 -> V[2,6]=1, V[3,6]=1
    # C part: C11 -> W[0,6]=1
    U2[1,6] = 1; U2[3,6] = -1
    V2[2,6] = 1; V2[3,6] = 1
    W2[0,6] = 1
    
    # 2. Construct 4x4x4 using 2x2x2
    # Rank 49.
    # U444 is 16x49, V444 is 16x49, W444 is 16x49.
    
    U444 = np.zeros((16, 49))
    V444 = np.zeros((16, 49))
    W444 = np.zeros((16, 49))
    
    # Mapping of block indices (a,b) for A, (c,d) for B, (a,d) for C.
    # A is 4x4, split into 2x2 blocks of 2x2.
    # A[2a+ia, 2b+ib]
    # B[2c+ic, 2d+id]
    # C[2a+ia, 2d+id]
    
    # We use Strassen on the block level.
    # M1 = (A11+A22)(B11+B22) -> blocks (0,0)+(1,1) and (0,0)+(1,1)
    # M2 = (A21+A22)B11 -> blocks (1,0)+(1,1) and (0,0)
    # M3 = A11(B12-B22) -> blocks (0,0) and (0,1)-(1,1)
    # M4 = A22(B21-B11) -> blocks (1,1) and (1,0)-(0,0)
    # M5 = (A11+A12)B22 -> blocks (0,0)+(0,1) and (1,1)
    # M6 = (A21-A11)(B11+B12) -> blocks (1,0)-(0,0) and (0,0)+(0,1)
    # M7 = (A12-A22)(B21+B22) -> blocks (0,1)-(1,1) and (1,0)+(1,1)
    
    # Each block product is a 2x2x2 multiplication.
    # We iterate over the 7 block products r_block in 0..6.
    # For each, we use the 7 scalar products r_sub in 0..6.
    # Total 49.
    
    # Block definitions:
    # A blocks: 00, 01, 10, 11 corresponding to (a,b)
    # B blocks: 00, 01, 10, 11 corresponding to (c,d)
    # C blocks: 00, 01, 10, 11 corresponding to (a,d)
    
    # We need to add contributions to U444, V444, W444.
    
    # Helper to add a 2x2x2 term
    # U2, V2, W2 are 4x7.
    # We need to place them into the 4x4x4 structure.
    
    # For a given block product M_block, defined by A_blocks and B_blocks,
    # and a sub-product r_sub,
    # we add to U444, V444, W444.
    
    # A_block (a,b) corresponds to rows 2a..2a+1, cols 2b..2b+1 of A.
    # In U444, index is (2a+ia)*4 + (2b+ib).
    # In U2, index is ia*2+ib.
    # So U444[2a+ia, 2b+ib, r] += U2[ia, ib, r_sub] * coeff_A
    
    # Similarly for V and W.
    
    # Let's define the block combinations for each M_block.
    # M1: A(00,11), B(00,11)
    # M2: A(10,11), B(00)
    # M3: A(00), B(01,-11)
    # M4: A(11), B(10,-00)
    # M5: A(00,01), B(11)
    # M6: A(10,-00), B(00,01)
    # M7: A(01,-11), B(10,11)
    
    block_defs = [
        # (A_blocks, coeffs), (B_blocks, coeffs), (C_blocks, coeffs)
        # C is determined by Strassen formula.
        # M1 -> C11 + C22
        # M2 -> C21
        # M3 -> C12
        # M4 -> C11 + C21
        # M5 -> -C11 + C12
        # M6 -> C22
        # M7 -> C11
    ]
    
    # C blocks:
    # C11: (0,0)
    # C12: (0,1)
    # C21: (1,0)
    # C22: (1,1)
    
    # We will iterate r_block in 0..6, r_sub in 0..6.
    
    # A block contributions
    A_blocks = [
        [(0,0,1), (1,1,1)], # M1
        [(1,0,1), (1,1,1)], # M2
        [(0,0,1)],           # M3
        [(1,1,1)],           # M4
        [(0,0,1), (0,1,1)],  # M5
        [(1,0,1), (0,0,-1)], # M6
        [(0,1,1), (1,1,-1)]  # M7
    ]
    
    B_blocks = [
        [(0,0,1), (1,1,1)], # M1
        [(0,0,1)],           # M2
        [(0,1,1), (1,1,-1)], # M3
        [(1,0,1), (0,0,-1)], # M4
        [(1,1,1)],           # M5
        [(0,0,1), (0,1,1)],  # M6
        [(1,0,1), (1,1,1)]   # M7
    ]
    
    C_blocks = [
        [(0,0,1), (1,1,1)], # M1 -> C11 + C22
        [(1,0,1)],           # M2 -> C21
        [(0,1,1)],           # M3 -> C12
        [(0,0,1), (1,0,1)], # M4 -> C11 + C21
        [(0,0,-1), (0,1,1)],# M5 -> -C11 + C12
        [(1,1,1)],           # M6 -> C22
        [(0,0,1)]            # M7 -> C11
    ]
    
    for r_block in range(7):
        for r_sub in range(7):
            r = r_block * 7 + r_sub
            
            # Add U contributions
            for a, b, coeff in A_blocks[r_block]:
                for ia in range(2):
                    for ib in range(2):
                        u_idx = (2*a + ia) * 4 + (2*b + ib)
                        u2_idx = ia * 2 + ib
                        U444[u_idx, r] += coeff * U2[u2_idx, r_sub]
                        
            # Add V contributions
            for c, d, coeff in B_blocks[r_block]:
                for ic in range(2):
                    for id_ in range(2):
                        v_idx = (2*c + ic) * 4 + (2*d + id_)
                        v2_idx = ic * 2 + id_
                        V444[v_idx, r] += coeff * V2[v2_idx, r_sub]
                        
            # Add W contributions
            for a, d, coeff in C_blocks[r_block]:
                for ia in range(2):
                    for id_ in range(2):
                        # W index is k*n + i.
                        # Here n=4.
                        # k is column index of C, which is d (block) + id_ (local).
                        # i is row index of C, which is a (block) + ia (local).
                        # So k = 2*d + id_, i = 2*a + ia.
                        w_idx = (2*d + id_) * 4 + (2*a + ia)
                        w2_idx = id_ * 2 + ia # Note: W2 index is k*n+i for n=2. k=id_, i=ia.
                        W444[w_idx, r] += coeff * W2[w2_idx, r_sub]

    # 3. Construct 4x4x2
    # Rank 28.
    # A is 4x4, B is 4x2.
    # U442 is 16x28, V442 is 8x28, W442 is 8x28.
    # W index: k*n + i. n=4. k in 0..1 (2 cols), i in 0..3.
    # W size 8x28.
    
    U442 = np.zeros((16, 28))
    V442 = np.zeros((8, 28))
    W442 = np.zeros((8, 28))
    
    # B is 4x2. Split into B_top (2x2) and B_bot (2x2).
    # B_top corresponds to rows 0,1 of B.
    # B_bot corresponds to rows 2,3 of B.
    
    # C_top = A_top B_top + A_bot B_bot? No.
    # C is 4x2.
    # C[0:2, :] = A[0:2, :] B.
    # A[0:2, :] = [A11, A12].
    # B = [B_top; B_bot].
    # C_top = A11 B_top + A12 B_bot.
    # C_bot = A21 B_top + A22 B_bot.
    
    # 4 products:
    # P1: A11 B_top
    # P2: A12 B_bot
    # P3: A21 B_top
    # P4: A22 B_bot
    
    # Each is 2x2x2.
    
    # P1: A11 (0,0), B_top (0,0 in B's row block? No, B_top is rows 0,1).
    # B index: j in 0..3, k in 0..1.
    # B_top: j in 0,1.
    # B_bot: j in 2,3.
    
    # V index: j*2 + k.
    # B_top: j=0,1. k=0,1.
    # B_bot: j=2,3. k=0,1.
    
    # P1: A11 (a=0,b=0) * B_top (j=0,1)
    # P2: A12 (a=0,b=1) * B_bot (j=2,3)
    # P3: A21 (a=1,b=0) * B_top (j=0,1)
    # P4: A22 (a=1,b=1) * B_bot (j=2,3)
    
    # C indices:
    # C_top: i=0,1. k=0,1.
    # C_bot: i=2,3. k=0,1.
    
    # W index: k*4 + i.
    
    products = [
        # (A_block, B_row_block, C_row_block)
        ((0,0), 0, 0), # P1: A11 * B_top -> C_top
        ((0,1), 1, 0), # P2: A12 * B_bot -> C_top
        ((1,0), 0, 1), # P3: A21 * B_top -> C_bot
        ((1,1), 1, 1)  # P4: A22 * B_bot -> C_bot
    ]
    
    for p_idx in range(4):
        a, b = products[p_idx][0]
        b_row_block = products[p_idx][1] # 0 for top, 1 for bot
        c_row_block = products[p_idx][2] # 0 for top, 1 for bot
        
        for r_sub in range(7):
            r = p_idx * 7 + r_sub
            
            # U
            for ia in range(2):
                for ib in range(2):
                    u_idx = (2*a + ia) * 4 + (2*b + ib)
                    u2_idx = ia * 2 + ib
                    U442[u_idx, r] += U2[u2_idx, r_sub]
                    
            # V
            # B row block b_row_block.
            # j = 2*b_row_block + ic.
            for ic in range(2):
                for id_ in range(2):
                    j = 2*b_row_block + ic
                    v_idx = j * 2 + id_
                    v2_idx = ic * 2 + id_
                    V442[v_idx, r] += V2[v2_idx, r_sub]
                    
            # W
            # C row block c_row_block.
            # i = 2*c_row_block + ia.
            # k = id_ (since B has 2 cols, k is 0 or 1).
            for ia in range(2):
                for id_ in range(2):
                    i = 2*c_row_block + ia
                    k = id_
                    w_idx = k * 4 + i
                    w2_idx = id_ * 2 + ia
                    W442[w_idx, r] += W2[w2_idx, r_sub]

    # 4. Combine
    # U is 16x77
    # V is 16x77 (for 4x4x4 part) + 8x77 (for 4x4x2 part)?
    # No, V must be consistent.
    # For 4x4x4, V is 16x49.
    # For 4x4x2, V is 8x28.
    # We need to map V442 into the 16-row space.
    # V442 rows correspond to j=0..3, k=0..1.
    # V444 rows correspond to j=0..3, k=0..3.
    # So V442 rows 0..7 correspond to V444 rows 0..7 (k=0,1) and 8..15 (k=2,3)?
    # No, V index is j*p + k.
    # For 4x4x4, p=4. V size 16x49.
    # For 4x4x2, p=2. V size 8x28.
    # We need to embed V442 into V444's row space.
    # V442[j*2+k, r] corresponds to V444[j*4+k, r]?
    # Yes, if we consider the 4x4x2 part as using only the first 2 columns of B.
    # But B is 4x6.
    # Columns 0,1,2,3 are handled by 4x4x4.
    # Columns 4,5 are handled by 4x4x2.
    
    # So for the 4x4x2 part, the columns of B are 4,5.
    # In the 4x4x2 decomposition, k is 0,1.
    # These correspond to global k=4,5.
    # So V442[j*2+k, r] should be placed at V444[j*4+(k+4), r]?
    # No, V index is j*p + k.
    # For the combined tensor, p=6.
    # V size 24x77.
    # For 4x4x4 part (cols 0-3), V uses k=0..3.
    # V444[j*4+k, r] -> V[j*6+k, r].
    
    # For 4x4x2 part (cols 4-5), V uses k=4,5.
    # V442[j*2+k_local, r] where k_local in 0,1.
    # Global k = 4 + k_local.
    # So V[j*6 + (4+k_local), r] += V442[j*2+k_local, r].
    
    # Similarly for W.
    # W444 is 16x49. W index k*4+i. k in 0..3.
    # W442 is 8x28. W index k*4+i. k in 0,1.
    # Global W size 24x77.
    # W444[k*4+i, r] -> W[k*4+i, r] (since k<4).
    # W442[k*4+i, r] -> W[(k+4)*4+i, r]?
    # No, W index is k_global * n + i. n=4.
    # k_global in 0..5.
    # For 4x4x4 part, k_global = k (0..3).
    # For 4x4x2 part, k_global = 4 + k_local.
    # So W[(4+k_local)*4+i, r] += W442[k_local*4+i, r].
    
    U = np.zeros((16, 77))
    V = np.zeros((24, 77))
    W = np.zeros((24, 77))
    
    # Copy 4x4x4 part
    U[:, :49] = U444
    for j in range(4):
        for k in range(4):
            V[j*6+k, :49] = V444[j*4+k, :49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, :49] = W444[k*4+i, :49]
            
    # Copy 4x4x2 part
    U[:, 49:] = U442
    for j in range(4):
        for k_local in range(2):
            V[j*6 + (4+k_local), 49:] = V442[j*2+k_local, 49:] # Wait, V442 is 8x28.
            # V442 row index is j*2+k_local.
            # V442[j*2+k_local, :]
            # V[j*6 + 4 + k_local, 49:]
            pass # Already handled in loop above? No, need to assign.
    
    # Let's do it properly
    for r in range(28):
        for j in range(4):
            for k_local in range(2):
                V[j*6 + 4 + k_local, 49+r] = V442[j*2+k_local, r]
                
    for r in range(28):
        for k_local in range(2):
            for i in range(4):
                W[(4+k_local)*4+i, 49+r] = W442[k_local*4+i, r]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
