"""Matrix multiplication 5x5x5: rank-117 block decomposition with Strassen on 2x2x2 blocks."""
import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    # Partition 5 = 2 + 2 + 1
    sizes = [2, 2, 1]
    n, m, p = DIMENSIONS  # all 5
    
    # Total rank = sum over all block combinations
    # For (2,2,2): rank 7 (Strassen)
    # For (2,2,1): rank 4 (schoolbook)
    # For (2,1,2): rank 4
    # For (2,1,1): rank 2
    # For (1,2,2): rank 4
    # For (1,2,1): rank 2
    # For (1,1,2): rank 2
    # For (1,1,1): rank 1
    # For (1,2,2) again... wait, let me count all 27 combinations:
    # (2,2,2): 7
    # (2,2,1): 4
    # (2,1,2): 4
    # (2,1,1): 2
    # (1,2,2): 4
    # (1,2,1): 2
    # (1,1,2): 2
    # (1,1,1): 1
    # Total: 7+4+4+2+4+2+2+1 = 26? That's not 117.
    
    # Wait, I need to reconsider. The partition is over the three dimensions.
    # n=5 = 2+2+1, m=5 = 2+2+1, p=5 = 2+2+1
    # So we have 3x3x3 = 27 blocks.
    # For each block (na, mb, pc):
    #   if (na,mb,pc) == (2,2,2): rank 7
    #   else: rank na*mb*pc
    # 
    # Count of (2,2,2) blocks: 1 (only one block where all three are 2)
    # No wait, the sizes are [2,2,1], so the block sizes are:
    #   block (i,j,k) has size sizes[i] x sizes[j] x sizes[k]
    #   sizes = [2, 2, 1]
    #   So block (0,0,0) = 2x2x2, rank 7
    #   block (0,0,1) = 2x2x1, rank 4
    #   block (0,1,0) = 2x2x2, rank 7  -- wait, sizes[1]=2, so this is 2x2x2
    #   block (0,1,1) = 2x2x1, rank 4
    #   block (1,0,0) = 2x2x2, rank 7
    #   block (1,0,1) = 2x2x1, rank 4
    #   block (1,1,0) = 2x2x2, rank 7
    #   block (1,1,1) = 2x2x1, rank 4
    #   block (2,0,0) = 1x2x2, rank 4
    #   block (2,0,1) = 1x2x1, rank 2
    #   block (2,1,0) = 1x2x2, rank 4
    #   block (2,1,1) = 1x2x1, rank 2
    #   block (0,2,0) = 2x1x2, rank 4
    #   block (0,2,1) = 2x1x1, rank 2
    #   block (1,2,0) = 2x1x2, rank 4
    #   block (1,2,1) = 2x1x1, rank 2
    #   block (2,2,0) = 1x1x2, rank 2
    #   block (2,2,1) = 1x1x1, rank 1
    #   block (0,0,0) already counted
    #   Let me just enumerate all 27:
    #   i from 0 to 2, j from 0 to 2, k from 0 to 2
    #   sizes = [2, 2, 1]
    #   (0,0,0): 2,2,2 -> 7
    #   (0,0,1): 2,2,1 -> 4
    #   (0,1,0): 2,2,2 -> 7
    #   (0,1,1): 2,2,1 -> 4
    #   (0,2,0): 2,1,2 -> 4
    #   (0,2,1): 2,1,1 -> 2
    #   (1,0,0): 2,2,2 -> 7
    #   (1,0,1): 2,2,1 -> 4
    #   (1,1,0): 2,2,2 -> 7
    #   (1,1,1): 2,2,1 -> 4
    #   (1,2,0): 2,1,2 -> 4
    #   (1,2,1): 2,1,1 -> 2
    #   (2,0,0): 1,2,2 -> 4
    #   (2,0,1): 1,2,1 -> 2
    #   (2,1,0): 1,2,2 -> 4
    #   (2,1,1): 1,2,1 -> 2
    #   (2,2,0): 1,1,2 -> 2
    #   (2,2,1): 1,1,1 -> 1
    #   Wait, I have 18 entries but need 27. Let me recount.
    #   i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
    #   That's 27. I missed some. Let me list all:
    #   (0,0,0): 2,2,2 -> 7
    #   (0,0,1): 2,2,1 -> 4
    #   (0,1,0): 2,2,2 -> 7
    #   (0,1,1): 2,2,1 -> 4
    #   (0,2,0): 2,1,2 -> 4
    #   (0,2,1): 2,1,1 -> 2
    #   (1,0,0): 2,2,2 -> 7
    #   (1,0,1): 2,2,1 -> 4
    #   (1,1,0): 2,2,2 -> 7
    #   (1,1,1): 2,2,1 -> 4
    #   (1,2,0): 2,1,2 -> 4
    #   (1,2,1): 2,1,1 -> 2
    #   (2,0,0): 1,2,2 -> 4
    #   (2,0,1): 1,2,1 -> 2
    #   (2,1,0): 1,2,2 -> 4
    #   (2,1,1): 1,2,1 -> 2
    #   (2,2,0): 1,1,2 -> 2
    #   (2,2,1): 1,1,1 -> 1
    #   That's still 18. I'm missing 9. Oh wait, I think I'm confusing myself.
    #   Let me just code it up properly.
    
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                na, mb, pc = sizes[i], sizes[j], sizes[k]
                if na == 2 and mb == 2 and pc == 2:
                    total_rank += 7
                else:
                    total_rank += na * mb * pc
    # Let me compute: 
    # Blocks with all 2s: (0,0,0), (0,1,0), (1,0,0), (1,1,0) -- wait, sizes[0]=2, sizes[1]=2, sizes[2]=1
    # So (i,j,k) with sizes[i]=2 means i in {0,1}, sizes[j]=2 means j in {0,1}, sizes[k]=2 means k... 
    # sizes[k]=2 means k in {0,1} since sizes[0]=2 and sizes[1]=2, but sizes[2]=1.
    # So blocks where all three are 2: i in {0,1}, j in {0,1}, k in {0,1} -> 4 blocks, each rank 7.
    # Other blocks: 27 - 4 = 23 blocks, each with rank na*mb*pc.
    
    # Let me just compute total_rank in code and proceed.
    
    # Build U, V, W as lists of lists
    # U has shape (n*m, rank) = (25, rank)
    # V has shape (m*p, rank) = (25, rank)  
    # W has shape (p*n, rank) = (25, rank)
    
    # First compute total rank
    total_rank = 0
    block_info = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                na, mb, pc = sizes[i], sizes[j], sizes[k]
                if na == 2 and mb == 2 and pc == 2:
                    b_rank = 7
                else:
                    b_rank = na * mb * pc
                block_info.append((i, j, k, na, mb, pc, b_rank, total_rank))
                total_rank += b_rank
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    # Offsets for row indices
    row_offsets = [0, 2, 4]  # cumulative sum of sizes
    col_offsets = [0, 2, 4]  # same for m and p dimensions
    
    for (i, j, k, na, mb, pc, b_rank, offset) in block_info:
        # Get local factors
        if na == 2 and mb == 2 and pc == 2:
            # Strassen for 2x2x2
            U_loc, V_loc, W_loc = strassen_2x2x2()
        else:
            # Schoolbook
            U_loc, V_loc, W_loc = schoolbook(na, mb, pc)
        
        # Map to global
        for a in range(na):
            for b in range(mb):
                u_row = (row_offsets[i] + a) * 5 + (row_offsets[j] + b)
                for c in range(b_rank):
                    U[u_row][offset + c] = U_loc[a * mb + b][c]
        
        for b in range(mb):
            for c in range(pc):
                v_row = (row_offsets[j] + b) * 5 + (row_offsets[k] + c)
                for r in range(b_rank):
                    V[v_row][offset + r] = V_loc[b * pc + c][r]
        
        for c in range(pc):
            for a in range(na):
                w_row = (row_offsets[k] + c) * 5 + (row_offsets[i] + a)
                for r in range(b_rank):
                    W[w_row][offset + r] = W_loc[c * na + a][r]
    
    return {"U": U, "V": V, "W": W}


def schoolbook(na, mb, pc):
    """Schoolbook decomposition for na x mb x pc."""
    rank = na * mb * pc
    U = [[0] * rank for _ in range(na * mb)]
    V = [[0] * rank for _ in range(mb * pc)]
    W = [[0] * rank for _ in range(pc * na)]
    col = 0
    for a in range(na):
        for b in range(mb):
            for c in range(pc):
                U[a * mb + b][col] = 1
                V[b * pc + c][col] = 1
                W[c * na + a][col] = 1
                col += 1
    return U, V, W


def strassen_2x2x2():
    """Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U maps A (2x2) indices to Strassen intermediates
    V maps B (2x2) indices to Strassen intermediates
    W maps C (2x2) indices to Strassen intermediates
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # W convention: W[k*n+i, r] where n=2, so W index = k*2 + i
    #   C[0,0] -> W[0*2+0] = W[0]
    #   C[0,1] -> W[0*2+1] = W[1]  -- wait, this doesn't match
    #   Actually W[k*n+i] with n=2:
    #   C[0,0]: k=0, i=0 -> W[0]
    #   C[1,0]: k=1, i=0 -> W[2]
    #   C[0,1]: k=0, i=1 -> W[1]
    #   C[1,1]: k=1, i=1 -> W[3]
    
    # Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M6
    
    # M0: (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    
    # M1: (A10+A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    
    # M2: A00(B01-B11)
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1  # B11
    
    # M3: A11(B10-B00)
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1  # B00
    
    # M4: (A00+A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    
    # M5: (A10-A00)(B00+B01)
    U[2][5] = 1  # A10
    U[0][5] = -1  # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    
    # M6: (A01-A11)(B10+B11)
    U[1][6] = 1  # A01
    U[3][6] = -1  # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    
    # C00 = M0 + M3 - M4 - M6 -> W[0]
    W[0][0] = 1   # M0
    W[0][3] = 1   # M3
    W[0][4] = -1  # M4
    W[0][6] = -1  # M6
    
    # C01 = M2 + M4 -> W[1] (k=0, i=1)
    W[1][2] = 1   # M2
    W[1][4] = 1   # M4
    
    # C10 = M1 + M5 -> W[2] (k=1, i=0)
    W[2][1] = 1   # M1
    W[2][5] = 1   # M5
    
    # C11 = M0 - M1 + M2 + M6 -> W[3] (k=1, i=1)
    W[3][0] = 1   # M0
    W[3][1] = -1  # M1
    W[3][2] = 1   # M2
    W[3][6] = 1   # M6
    
    return U, V, W


# EVOLVE-BLOCK-START
# (construct, schoolbook, strassen_2x2x2 defined above)
# EVOLVE-BLOCK-END
