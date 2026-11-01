"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    # We'll build a block-diagonal decomposition:
    # Split the 5 columns of A (and 5 rows of B) into blocks of size 2,2,1.
    # Split the 6 rows of B (and 6 cols of C) into blocks of size 2,2,2.
    # For each (A_block, B_block) pair, use a rank-minimal decomposition for that sub-multiplication.
    # The total rank is the sum of ranks of all block decompositions.
    
    # Block sizes for m=5: [2, 2, 1]
    # Block sizes for p=6: [2, 2, 2]
    # For each (m_block, p_block), we need a decomposition for n x m_block x p_block.
    # n is always 2.
    
    # We need decompositions for:
    # 2x2x2: rank 7 (Strassen)
    # 2x2x1: rank 4 (trivial, since p=1, it's just 2*2 scalar mults... wait)
    # Actually, for 2x2x1: C is 2x1. C[i,0] = sum_j A[i,j] B[j,0].
    # This is 2 dot products of length 2. Each dot product of length 2 needs 3 mults (Strassen for 1x2x1? No).
    # Actually, a dot product of length k needs k multiplications in the trivial algorithm.
    # But we can use Strassen-like for larger blocks.
    # For 1x2x1: rank 2 (just a*b + c*d, 2 mults).
    # So 2x2x1: two independent 1x2x1 dot products? No, they share the B column.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # This is a 2x2x1 tensor. The trivial rank is 4.
    # Can we do better? The tensor is T[i,j,k] = delta_{k,0} (since p=1, k is always 0).
    # Actually, T[i,j,0] = 1 for all i,j. So it's just the identity tensor for the i,j indices, with k fixed.
    # The rank of the 2x2x1 tensor (which is just a 2x2 matrix of 1s, repeated for k=0) is 4.
    # Because we need 4 independent multiplications: A[0,0]*B[0,0], A[0,1]*B[1,0], A[1,0]*B[0,0], A[1,1]*B[1,0].
    # These are all independent. So rank 4.
    
    # 2x1x2: C is 2x2. C[i,k] = A[i,0] * B[0,k].
    # This is an outer product of A[:,0] (2x1) and B[0,:] (1x2).
    # Rank is 4? No, the tensor T[i,j,k] = delta_{j,0}.
    # T[i,0,k] = 1. So C = A[:,0] * B[0,:]. This is a rank-1 matrix (outer product).
    # But in the tensor decomposition sense, we need to compute this with scalar multiplications.
    # C[i,k] = A[i,0] * B[0,k]. We need 4 multiplications: A[0,0]*B[0,0], A[0,0]*B[0,1], A[1,0]*B[0,0], A[1,0]*B[0,1].
    # These are all independent. So rank 4.
    
    # 2x1x1: rank 1.
    
    # So we need decompositions for:
    # 2x2x2: rank 7
    # 2x2x1: rank 4
    # 2x1x2: rank 4
    # 2x1x1: rank 1
    
    # Now, let's build the block decompositions.
    # m_blocks = [2, 2, 1]
    # p_blocks = [2, 2, 2]
    
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 2]
    
    # Precompute decompositions for each (mb, pb)
    # We'll return U_block, V_block, W_block for each, with appropriate shapes.
    # U_block: (n * mb, rank_block)
    # V_block: (mb * pb, rank_block)
    # W_block: (pb * n, rank_block)
    
    def get_2x2x2():
        # Strassen decomposition for 2x2x2
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = (a-b)g
        # P5 = (b+d)(h-g)
        # P6 = (a-c)(e+f)
        # P7 = (c-d)(f+h)
        # C = [[P1+P4+P5-P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
        
        # U: (4, 7) where A indices are 0:a, 1:b, 2:c, 3:d
        U = [
            [1, 0, 0, 1],  # P1: a+d
            [0, 0, 1, 1],  # P2: c+d
            [1, 0, 0, 0],  # P3: a
            [0, 0, 0, 1],  # P4: d
            [1, 1, 0, 0],  # P5: a+b
            [-1, 0, 1, 0], # P6: -a+c
            [0, 1, 0, -1]  # P7: b-d
        ]
        # V: (4, 7) where B indices are 0:e, 1:f, 2:g, 3:h
        V = [
            [1, 0, 0, 1],  # P1: e+h
            [1, 0, 0, 0],  # P2: e
            [0, 1, 0, -1], # P3: f-h
            [0, 0, 1, 0],  # P4: g
            [0, 0, -1, 1], # P5: -g+h
            [1, 1, 0, 0],  # P6: e+f
            [0, 1, 0, 1]   # P7: f+h
        ]
        # W: (4, 7) where C indices are 0:C00, 1:C01, 2:C10, 3:C11
        W = [
            [1, 0, 0, 1],  # P1: C00, C11
            [0, 0, 1, -1], # P2: C10, -C11
            [0, 1, 0, 1],  # P3: C01, C11
            [1, 0, 1, 0],  # P4: C00, C10
            [1, 1, 0, 0],  # P5: C00, C01
            [0, 0, 0, 1],  # P6: C11
            [-1, 0, 0, 0]  # P7: -C00
        ]
        return U, V, W
    
    def get_2x2x1():
        # Trivial decomposition for 2x2x1
        # C[i,0] = sum_j A[i,j] B[j,0]
        # 4 multiplications: (i,j) in {(0,0),(0,1),(1,0),(1,1)}
        U = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        V = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        W = [
            [1, 0],  # C00
            [0, 1],  # C01 (but p=1, so only C00? Wait, p=1 means C is 2x1)
            # Wait, for 2x2x1, C is 2x1. So W should be (1*2, 4) = (2, 4)
            # W[k*n+i, r] where k=0, i in {0,1}
            # So W has 2 rows.
            [1, 0, 0, 0],  # C00
            [0, 0, 1, 0]   # C10
        ]
        # Let me re-check the indexing.
        # W[k*n+i, r]
        # For p=1, k=0. i in {0,1}.
        # So W[0, r] is for C[0,0], W[1, r] is for C[1,0].
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # So:
        # r=0: A[0,0]*B[0,0] -> C[0,0]
        # r=1: A[0,1]*B[1,0] -> C[0,0]
        # r=2: A[1,0]*B[0,0] -> C[1,0]
        # r=3: A[1,1]*B[1,0] -> C[1,0]
        W = [
            [1, 1, 0, 0],  # C00
            [0, 0, 1, 1]   # C10
        ]
        return U, V, W
    
    def get_2x1x2():
        # Trivial decomposition for 2x1x2
        # C[i,k] = A[i,0] * B[0,k]
        # 4 multiplications: (i,k) in {(0,0),(0,1),(1,0),(1,1)}
        U = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        V = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        # W: (p*n, rank) = (2*2, 4) = (4, 4)
        # W[k*n+i, r]
        # r=0: A[0,0]*B[0,0] -> C[0,0]
        # r=1: A[0,0]*B[0,1] -> C[0,1]
        # r=2: A[1,0]*B[0,0] -> C[1,0]
        # r=3: A[1,0]*B[0,1] -> C[1,1]
        W = [
            [1, 0, 0, 0],  # C00 (k=0, i=0)
            [0, 1, 0, 0],  # C01 (k=0, i=1) -- wait, C[0,1] means i=0, k=1?
            # Let me re-check the indexing.
            # C[i,k] where i is row (0..n-1), k is col (0..p-1)
            # W[k*n+i, r]
            # So for C[0,0]: k=0, i=0 -> W[0, r]
            # For C[0,1]: k=1, i=0 -> W[2, r]
            # For C[1,0]: k=0, i=1 -> W[1, r]
            # For C[1,1]: k=1, i=1 -> W[3, r]
            [1, 0, 0, 0],  # W[0]: C00
            [0, 0, 1, 0],  # W[1]: C10
            [0, 1, 0, 0],  # W[2]: C01
            [0, 0, 0, 1]   # W[3]: C11
        ]
        return U, V, W
    
    def get_2x1x1():
        # Trivial decomposition for 2x1x1
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    
    # Now, build the full decomposition by combining blocks.
    # m_blocks = [2, 2, 1]
    # p_blocks = [2, 2, 2]
    
    # For each (mb_idx, pb_idx), get the decomposition.
    # The total U, V, W will be block-diagonal in the rank dimension.
    
    U_full = []
    V_full = []
    W_full = []
    
    # We'll accumulate the rank.
    # For each block, we append the U_block, V_block, W_block to the full matrices.
    # But we need to zero-pad the other blocks.
    
    # Actually, it's easier to build the full matrices directly.
    # U_full: (n*m, total_rank)
    # V_full: (m*p, total_rank)
    # W_full: (p*n, total_rank)
    
    # First, compute total_rank.
    total_rank = 0
    block_decomps = {}
    for mb_idx, mb in enumerate(m_blocks):
        for pb_idx, pb in enumerate(p_blocks):
            if mb == 2 and pb == 2:
                U_b, V_b, W_b = get_2x2x2()
                rank_b = 7
            elif mb == 2 and pb == 1:
                U_b, V_b, W_b = get_2x2x1()
                rank_b = 4
            elif mb == 1 and pb == 2:
                U_b, V_b, W_b = get_2x1x2()
                rank_b = 4
            else:  # mb == 1 and pb == 1
                U_b, V_b, W_b = get_2x1x1()
                rank_b = 1
            block_decomps[(mb_idx, pb_idx)] = (U_b, V_b, W_b, rank_b)
            total_rank += rank_b
    
    # Now, build the full matrices.
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    for mb_idx, mb in enumerate(m_blocks):
        for pb_idx, pb in enumerate(p_blocks):
            U_b, V_b, W_b, rank_b = block_decomps[(mb_idx, pb_idx)]
            
            # Compute the starting indices for this block.
            # m_start = sum of m_blocks[:mb_idx]
            m_start = sum(m_blocks[:mb_idx])
            # p_start = sum of p_blocks[:pb_idx]
            p_start = sum(p_blocks[:pb_idx])
            
            # Fill U_full.
            # U_b has shape (n * mb, rank_b)
            # U_full[i * m + j, rank_offset + r] = U_b[i * mb + j, r]
            # where i in 0..n-1, j in 0..mb-1, r in 0..rank_b-1
            for i in range(n):
                for j in range(mb):
                    for r in range(rank_b):
                        U_full[i * m + (m_start + j)][rank_offset + r] = U_b[i * mb + j][r]
            
            # Fill V_full.
            # V_b has shape (mb * pb, rank_b)
            # V_full[j * p + k, rank_offset + r] = V_b[j * pb + k, r]
            # where j in 0..mb-1, k in 0..pb-1, r in 0..rank_b-1
            for j in range(mb):
                for k in range(pb):
                    for r in range(rank_b):
                        V_full[(m_start + j) * p + (p_start + k)][rank_offset + r] = V_b[j * pb + k][r]
            
            # Fill W_full.
            # W_b has shape (pb * n, rank_b)
            # W_full[k * n + i, rank_offset + r] = W_b[k * n + i, r]
            # where k in 0..pb-1, i in 0..n-1, r in 0..rank_b-1
            for k in range(pb):
                for i in range(n):
                    for r in range(rank_b):
                        W_full[(p_start + k) * n + i][rank_offset + r] = W_b[k * n + i][r]
            
            rank_offset += rank_b
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
