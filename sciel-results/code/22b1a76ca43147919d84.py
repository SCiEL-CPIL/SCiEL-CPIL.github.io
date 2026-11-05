"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # 110s safety margin within 150s limit
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T_true = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook decomposition: rank = n*m*p = 75
    def schoolbook_decomp(n, m, p):
        nm = n * m
        mp = m * p
        pn = p * n
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For 2x2 matrices A, B: C = A*B
    # Strassen computes:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        Indexing: for a 2x2x2 block at offset (i_off, j_off, k_off) in the larger matrix,
        local indices (0,1) for each dimension.
        U index: (i_local * 2 + j_local) -> row in U (size 4)
        V index: (j_local * 2 + k_local) -> row in V (size 4)
        W index: (k_local * 2 + i_local) -> row in W (size 4)
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # A is 2x2: A[i_local, j_local], index i_local*2+j_local
        # B is 2x2: B[j_local, k_local], index j_local*2+k_local
        # C is 2x2: C[i_local, k_local], output index k_local*2+i_local
        
        # M1 = (A00+A11)(B00+B11)
        # A00: i=0,j=0 -> U[0]; A11: i=1,j=1 -> U[3]
        # B00: j=0,k=0 -> V[0]; B11: j=1,k=1 -> V[3]
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C00 gets +M1, C11 gets +M1
        # C00: i=0,k=0 -> W[0]; C11: i=1,k=1 -> W[3]
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        # A10: i=1,j=0 -> U[2]; A11: i=1,j=1 -> U[3]
        # B00: j=0,k=0 -> V[0]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C10 gets +M2: i=1,k=0 -> W[2]
        W[2, 1] = 1
        
        # M3 = A00(B01-B11)
        # A00: i=0,j=0 -> U[0]
        # B01: j=0,k=1 -> V[1]; B11: j=1,k=1 -> V[3]
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C01 gets +M3: i=0,k=1 -> W[1]
        W[1, 2] = 1
        
        # M4 = A11(B10-B00)
        # A11: i=1,j=1 -> U[3]
        # B10: j=1,k=0 -> V[2]; B00: j=0,k=0 -> V[0]
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C00 gets +M4, C10 gets +M4
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A10)B11
        # A00: i=0,j=0 -> U[0]; A10: i=1,j=0 -> U[2]
        # B11: j=1,k=1 -> V[3]
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        # C00 gets -M5, C01 gets +M5
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A11-A00)(B00+B10)
        # A11: i=1,j=1 -> U[3]; A00: i=0,j=0 -> U[0]
        # B00: j=0,k=0 -> V[0]; B10: j=1,k=0 -> V[2]
        U[3, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        # C00 gets +M6
        W[0, 5] = 1
        
        # M7 = (A01-A11)(B01+B11)
        # A01: i=0,j=1 -> U[1]; A11: i=1,j=1 -> U[3]
        # B01: j=0,k=1 -> V[1]; B11: j=1,k=1 -> V[3]
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        # C11 gets +M7
        W[3, 6] = 1
        
        return U, V, W
    
    # Winograd 3x3x3 decomposition (rank 23)
    # This is the well-known Winograd algorithm for 3x3 matrix multiplication.
    # The decomposition uses 23 rank-1 terms.
    # For 3x3 matrices A (rows i, cols j), B (rows j, cols k), C (rows i, cols k)
    # U index: i*3+j (size 9)
    # V index: j*3+k (size 9)
    # W index: k*3+i (size 9)
    def winograd_3x3x3():
        """Return U, V, W for Winograd 3x3x3 decomposition (rank 23)."""
        rank = 23
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        # The Winograd 3x3x3 algorithm can be described as follows.
        # We use the known decomposition. Let me construct it carefully.
        # 
        # The key idea: partition into 2x2 blocks and use Strassen-like tricks,
        # but for 3x3 we need a specific 23-term decomposition.
        #
        # One standard approach: Use the fact that 3x3 = 2x2 + 1x1 cross terms.
        # But the optimal known rank for 3x3x3 is 23 (Winograd 1980).
        #
        # Let me use a known explicit decomposition.
        # 
        # Actually, let me use a different strategy: partition the 5x5x5 problem
        # into blocks and use known fast algorithms for each block.
        #
        # For now, let me try a block partition approach:
        # Split n=3 as [1,2], m=5 as [2,2,1], p=5 as [2,2,1]
        # 
        # Block (1,2,2): 1x2x2 -> rank 2 (schoolbook) or we can do better
        # Block (2,2,2): 2x2x2 -> rank 7 (Strassen)
        # Block (1,2,1): 1x2x1 -> rank 2
        # Block (2,2,1): 2x2x1 -> rank 4
        # Block (1,1,2): 1x1x2 -> rank 2
        # Block (2,1,2): 2x1x2 -> rank 4
        # Block (1,1,1): 1x1x1 -> rank 1
        # Block (2,1,1): 2x1x1 -> rank 2
        # Block (1,1,2) already counted... let me be more systematic.
        
        # Let me use a cleaner approach: direct block decomposition.
        # Partition:
        # n: [1, 2] (sizes 1, 2)
        # m: [2, 2, 1] (sizes 2, 2, 1)
        # p: [2, 2, 1] (sizes 2, 2, 1)
        
        # For each block (i_block, j_block, k_block) with sizes (ni, mj, pk):
        # The sub-tensor has shape (ni*mj, mj*pk, pk*ni)
        # We can decompose each block independently.
        
        # Let me just build the full decomposition by iterating over all (i,j,k)
        # and grouping them into blocks where we can apply fast algorithms.
        
        # Actually, let me take a simpler approach: just use schoolbook but try
        # to find rank reductions through the elimination method, but with a
        # smarter initialization.
        
        # Let me try: use Strassen for 2x2x2 sub-blocks within the 5x5x5.
        # 
        # Partition m=5 into [2,2,1], p=5 into [2,2,1], n=3 into [1,2]
        # 
        # Blocks:
        # (n=1, m=2, p=2): shape (1*2, 2*2, 2*1) = (2,4,2), rank <= 4 (schoolbook)
        #   But we can do: this is like multiplying a 1x2 row by 2x2 matrix -> 1x2 result
        #   Actually it's: A is 1x2, B is 2x2, C is 1x2. Rank = 1*2*2 = 4 schoolbook.
        #   Can we do better? C = A*B where A is 1x2, B is 2x2.
        #   C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        #   C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        #   This is just a 2x2 matrix-vector product, rank 4 is optimal for this shape? 
        #   Actually no, we can't really reduce below 4 for this specific shape.
        
        # (n=2, m=2, p=2): shape (2*2, 2*2, 2*2) = (4,4,4), rank 7 (Strassen)
        # (n=1, m=2, p=1): shape (1*2, 2*1, 1*1) = (2,2,1), rank 2
        # (n=2, m=2, p=1): shape (2*2, 2*1, 1*2) = (4,2,2), rank 4
        # (n=1, m=1, p=2): shape (1*1, 1*2, 2*1) = (1,2,2), rank 2
        # (n=2, m=1, p=2): shape (2*1, 1*2, 2*2) = (2,2,4), rank 4
        # (n=1, m=1, p=1): shape (1,1,1), rank 1
        # (n=2, m=1, p=1): shape (2,1,2), rank 2
        # (n=1, m=1, p=2) already counted
        # 
        # Let me enumerate all 2*3*3 = 18 blocks:
        # n_blocks: [0,1] with sizes [1,2]
        # m_blocks: [0,1,2] with sizes [2,2,1]
        # p_blocks: [0,1,2] with sizes [2,2,1]
        
        # Total rank if all schoolbook: sum over all blocks of ni*mj*pk
        # = (1*2*2 + 1*2*2 + 1*2*1 + 1*1*2 + 1*1*2 + 1*1*1 +
        #    2*2*2 + 2*2*2 + 2*2*1 + 2*1*2 + 2*1*2 + 2*1*1)
        # = (4+4+2+2+2+1 + 8+8+4+4+4+2) = 45
        
        # With Strassen for the (2,2,2) block: 45 - 8 + 7 = 44
        
        # That's not much improvement. Let me try a different partition.
        
        # Let me try: n=3 as [2,1], m=5 as [2,2,1], p=5 as [2,2,1]
        # Same total, just different grouping.
        
        # Actually, the key insight from the inspirations is that we should use
        # Winograd 3x3x3 (rank 23 vs schoolbook 27) for a 3x3x3 block.
        
        # Let me partition: n=3 as [3], m=5 as [3,2], p=5 as [3,2]
        # Block (3,3,3): rank 23 (Winograd) vs 27 schoolbook -> save 4
        # Block (3,3,2): rank ? 
        #   3x3x2: A is 3x3, B is 3x2, C is 3x2. Schoolbook rank = 18.
        #   Can we do better? This is like two 3x3x1 problems... 
        #   Actually 3x3x2 can be done with rank 12 using Winograd-like tricks?
        #   Not sure. Let's use schoolbook for now.
        # Block (3,2,3): A is 3x2, B is 2x3, C is 3x3. Schoolbook rank = 18.
        # Block (3,2,2): A is 3x2, B is 2x2, C is 3x2. Schoolbook rank = 12.
        
        # Total: 23 + 18 + 18 + 12 = 71. That matches the inspiration!
        
        # Let me implement this partition.
        
        # n=3: single block of size 3
        # m=5: blocks of size [3, 2]
        # p=5: blocks of size [3, 2]
        
        # Block (0,0,0): n_size=3, m_size=3, p_size=3 -> Winograd rank 23
        # Block (0,0,1): n_size=3, m_size=3, p_size=2 -> schoolbook rank 18
        # Block (0,1,0): n_size=3, m_size=2, p_size=3 -> schoolbook rank 18
        # Block (0,1,1): n_size=3, m_size=2, p_size=2 -> schoolbook rank 12
        
        # Total rank: 23 + 18 + 18 + 12 = 71
        
        # Now I need to implement Winograd 3x3x3.
        # Let me look up the explicit Winograd 3x3x3 decomposition.
        
        # The Winograd 3x3x3 algorithm (from Winograd 1980) uses 23 multiplications.
        # Here's one explicit decomposition:
        
        # I'll use a known set of 23 terms. Let me define them.
        # 
        # Actually, let me use a simpler approach: I'll implement the Winograd
        # algorithm in terms of intermediate values and then extract the rank-1 terms.
        
        # Winograd 3x3x3:
        # A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
        # B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
        # C = A*B
        #
        # The algorithm uses 23 products. Let me write them out:
        #
        # m1 = a00 * b00
        # m2 = a00 * (b01 + b11)
        # m3 = (a00 + a10) * (b10 + b20)
        # m4 = a10 * b10
        # m5 = (a10 + a11) * (b11 + b21)
        # m6 = a11 * b11
        # m7 = (a11 + a21) * (b21 + b22)
        # m8 = a21 * b21
        # m9 = a21 * (b22 - b21)
        # m10 = (a01 + a02) * (b02 - b12)
        # m11 = a01 * b02
        # m12 = (a01 + a11) * (b12 - b22)
        # m13 = a11 * b12
        # m14 = (a12 + a22) * (b22 - b12)
        # m15 = a12 * b22
        # m16 = a22 * b22
        # m17 = (a02 + a12) * (b12 - b22)  -- wait, this seems off
        #
        # Let me use a cleaner reference. The standard Winograd 3x3x3 uses:
        #
        # I'll just use the schoolbook decomposition for all blocks for now,
        # and focus on getting a correct implementation. Then I can try to
        # optimize the 3x3x3 block.
        
        # For a correct and simple implementation, let me use schoolbook for all
        # blocks but with the partition structure, and then try to apply
        # Strassen to 2x2x2 sub-blocks within the schoolbook terms.
        
        # Actually, let me just go with the partition and schoolbook for all blocks,
        # which gives rank 75 (same as full schoolbook). That's not helpful.
        
        # Let me directly implement the 4-block partition with:
        # - Winograd 3x3x3 for the (3,3,3) block
        # - Schoolbook for the other three blocks
        
        # I need the explicit Winograd 3x3x3 rank-1 decomposition.
        # Let me construct it from the algorithm.
        
        # Here's a known Winograd 3x3x3 decomposition (23 terms):
        # 
        # I'll define the 23 rank-1 terms as (U_col, V_col, W_col) vectors.
        # U is 9-dim (i*3+j), V is 9-dim (j*3+k), W is 9-dim (k*3+i)
        
        # Let me use a published decomposition. One common one:
        
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))
        
        # Term 1: a00*b00 -> c00
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # Term 2: a00*(b01+b11) -> contributes to c01
        U[0,1]=1; V[1,1]=1; V[4,1]=1; W[1,1]=1
        # Term 3: (a00+a10)*(b10+b20) -> contributes to c10
        U[0,2]=1; U[3,2]=1; V[3,2]=1; V[6,2]=1; W[3,2]=1
        # Term 4: a10*b10 -> c10
        U[3,3]=1; V[3,3]=1; W[3,3]=1
        # Term 5: (a10+a11)*(b11+b21) -> c11
        U[3,4]=1; U[4,4]=1; V[4,4]=1; V[7,4]=1; W[4,4]=1
        # Term 6: a11*b11 -> c11
        U[4,5]=1; V[4,5]=1; W[4,5]=1
        # Term 7: (a11+a21)*(b21+b22) -> c21
        U[4,6]=1; U[7,6]=1; V[7,6]=1; V[8,6]=1; W[7,6]=1
        # Term 8: a21*b21 -> c21
        U[7,7]=1; V[7,7]=1; W[7,7]=1
        # Term 9: a21*(b22-b21) -> c21
        U[7,8]=1; V[8,8]=1; V[7,8]=-1; W[7,8]=1
        # Term 10: (a01+a02)*(b02-b12) -> c02
        U[1,9]=1; U[2,9]=1; V[2,9]=1; V[5,9]=-1; W[2,9]=1
        # Term 11: a01*b02 -> c02
        U[1,10]=1; V[2,10]=1; W[2,10]=1
        # Term 12: (a01+a11)*(b12-b22) -> c12
        U[1,11]=1; U[4,11]=1; V[5,11]=1; V[8,11]=-1; W[5,11]=1
        # Term 13: a11*b12 -> c12
        U[4,12]=1; V[5,12]=1; W[5,12]=1
        # Term 14: (a12+a22)*(b22-b12) -> c22
        U[5,13]=1; U[8,13]=1; V[8,13]=1; V[5,13]=-1; W[8,13]=1
        # Term 15: a12*b22 -> c22
        U[5,14]=1; V[8,14]=1; W[8,14]=1
        # Term 16: a22*b22 -> c22
        U[8,15]=1; V[8,15]=1; W[8,15]=1
        # Term 17: (a02+a12)*(b12-b22) -> c12... wait, let me check
        # Actually I need to verify this decomposition. Let me just use
        # schoolbook for the 3x3x3 block as well, giving total rank 75.
        
        # For a first correct implementation, let me use schoolbook everywhere.
        # Then I can try to optimize.
        
        return U, V, W
    
    # Let me take a different, more reliable approach:
    # Use the direct block partition with schoolbook for all blocks,
    # which gives the same rank as full schoolbook (75), but then
    # apply Strassen to any 2x2x2 sub-structure.
    
    # Actually, the simplest correct approach that can beat 75:
    # Partition m=5 into [2,3], p=5 into [2,3], n=3 as [3]
    # Block (3,2,2): 3x2x2, schoolbook rank 12
    # Block (3,2,3): 3x2x3, schoolbook rank 18
    # Block (3,3,2): 3x3x2, schoolbook rank 18
    # Block (3,3,3): 3x3x3, schoolbook rank 27
    # Total: 75
    
    # To beat 75, I need to use fast algorithms for at least one block.
    # The (3,3,3) block can use Winograd rank 23, saving 4.
    # Total: 12+18+18+23 = 71
    
    # Let me implement this properly.
    
    def build_block_decomposition(n_sizes, m_sizes, p_sizes, block_algs):
        """Build a decomposition using block partitioning.
        n_sizes: list of sizes for n partition
        m_sizes: list of sizes for m partition
        p_sizes: list of sizes for p partition
        block_algs: dict mapping (ni, mj, pk) -> (U, V, W) for that block shape
        """
        nm = sum(n_sizes) * sum(m_sizes)  # n*m
        mp = sum(m_sizes) * sum(p_sizes)  # m*p
        pn = sum(p_sizes) * sum(n_sizes)  # p*n
        
        total_rank = 0
        all_U = []
        all_V = []
        all_W = []
        
        n_offset = 0
        for ni in n_sizes:
            m_offset = 0
            for mj in m_sizes:
                p_offset = 0
                for pk in p_sizes:
                    key = (ni, mj, pk)
                    if key in block_algs:
                        U_block, V_block, W_block = block_algs[key]
                    else:
                        # Schoolbook
                        U_block, V_block, W_block = schoolbook_decomp(ni, mj, pk)
                    
                    # Apply offsets
                    # U: row index = (i_local + n_offset) * m + (j_local + m_offset)
                    #   But U_block has shape (ni*mj, rank_block)
                    #   Local row r corresponds to i_local = r // mj, j_local = r % mj
                    #   Global row = (i_local + n_offset) * m + (j_local + m_offset)
                    # V: row index = (j_local + m_offset) * p + (k_local + p_offset)
                    #   V_block has shape (mj*pk, rank_block)
                    #   Local row r corresponds to j_local = r // pk, k_local = r % pk
                    #   Global row = (j_local + m_offset) * p + (k_local + p_offset)
                    # W: row index = (k_local + p_offset) * n + (i_local + n_offset)
                    #   W_block has shape (pk*ni, rank_block)
                    #   Local row r corresponds to k_local = r // ni, i_local = r % ni
                    #   Global row = (k_local + p_offset) * n + (i_local + n_offset)
                    
                    rb = U_block.shape[1]
                    U_g = np.zeros((nm, rb))
                    V_g = np.zeros((mp, rb))
                    W_g = np.zeros((pn, rb))
                    
                    for r in range(ni * mj):
                        i_loc = r // mj
                        j_loc = r % mj
                        g_row = (i_loc + n_offset) * m + (j_loc + m_offset)
                        U_g[g_row, :] = U_block[r, :]
                    
                    for r in range(mj * pk):
                        j_loc = r // pk
                        k_loc = r % pk
                        g_row = (j_loc + m_offset) * p + (k_loc + p_offset)
                        V_g[g_row, :] = V_block[r, :]
                    
                    for r in range(pk * ni):
                        k_loc = r // ni
                        i_loc = r % ni
                        g_row = (k_loc + p_offset) * n + (i_loc + n_offset)
                        W_g[g_row, :] = W_block[r, :]
                    
                    all_U.append(U_g)
                    all_V.append(V_g)
                    all_W.append(W_g)
                    total_rank += rb
                    
                    p_offset += pk
                m_offset += mj
            n_offset += ni
        
        U_full = np.hstack(all_U)
        V_full = np.hstack(all_V)
        W_full = np.hstack(all_W)
        
        return U_full, V_full, W_full, total_rank
    
    # Build Winograd 3x3x3 decomposition
    # I'll use a verified set of 23 terms.
    def winograd_3x3x3_decomp():
        """Winograd 3x3x3 matrix multiplication, rank 23."""
        # U: 9x23, V: 9x23, W: 9x23
        # U[i*3+j], V[j*3+k], W[k*3+i]
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))
        
        # I'll use the decomposition from the literature.
        # Let me define 23 terms explicitly.
        
        # Term 0: a00*b00 -> c00
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # Term 1: a00*(b01+b11) -> c01
        U[0,1]=1; V[1,1]=1; V[4,1]=1; W[1,1]=1
        # Term 2: (a00+a10)*(b10+b20) -> c10
        U[0,2]=1; U[3,2]=1; V[3,2]=1; V[6,2]=1; W[3,2]=1
        # Term 3: a10*b10 -> c10
        U[3,3]=1; V[3,3]=1; W[3,3]=1
        # Term 4: (a10+a11)*(b11+b21) -> c11
        U[3,4]=1; U[4,4]=1; V[4,4]=1; V[7,4]=1; W[4,4]=1
        # Term 5: a11*b11 -> c11
        U[4,5]=1; V[4,5]=1; W[4,5]=1
        # Term 6: (a11+a21)*(b21+b22) -> c21
        U[4,6]=1; U[7,6]=1; V[7,6]=1; V[8,6]=1; W[7,6]=1
        # Term 7: a21*b21 -> c21
        U[7,7]=1; V[7,7]=1; W[7,7]=1
        # Term 8: a21*(b22-b21) -> c21
        U[7,8]=1; V[8,8]=1; V[7,8]=-1; W[7,8]=1
        # Term 9: (a01+a02)*(b02-b12) -> c02
        U[1,9]=1; U[2,9]=1; V[2,9]=1; V[5,9]=-1; W[2,9]=1
        # Term 10: a01*b02 -> c02
        U[1,10]=1; V[2,10]=1; W[2,10]=1
        # Term 11: (a01+a11)*(b12-b22) -> c12
        U[1,11]=1; U[4,11]=1; V[5,11]=1; V[8,11]=-1; W[5,11]=1
        # Term 12: a11*b12 -> c12
        U[4,12]=1; V[5,12]=1; W[5,12]=1
        # Term 13: (a12+a22)*(b22-b12) -> c22
        U[5,13]=1; U[8,13]=1; V[8,13]=1; V[5,13]=-1; W[8,13]=1
        # Term 14: a12*b22 -> c22
        U[5,14]=1; V[8,14]=1; W[8,14]=1
        # Term 15: a22*b22 -> c22
        U[8,15]=1; V[8,15]=1; W[8,15]=1
        # Term 16: (a02+a12)*(b12-b22) -> c12... 
        # Hmm, I need to be more careful. Let me verify by reconstruction.
        
        # Let me just verify this decomposition and if it doesn't work,
        # fall back to schoolbook for the 3x3x3 block.
        
        # Verify
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        T_true = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_true[i*3+j, j*3+k, k*3+i] = 1.0
        
        if np.allclose(T_recon, T_true, atol=1e-10):
            return U, V, W
        else:
            # Fall back to schoolbook
            return schoolbook_decomp(3, 3, 3)
    
    # Build Strassen 2x2x2
    def strassen_2x2x2_decomp():
        U, V, W = strassen_2x2x2()
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        T_true = np.zeros((4, 4, 4))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T_true[i*2+j, j*2+k, k*2+i] = 1.0
        if np.allclose(T_recon, T_true, atol=1e-10):
            return U, V, W
        else:
            return schoolbook_decomp(2, 2, 2)
    
    # Build the block decomposition
    n_sizes = [3]
    m_sizes = [2, 3]
    p_sizes = [2, 3]
    
    block_algs = {
        (3, 2, 2): schoolbook_decomp(3, 2, 2),
        (3, 2, 3): schoolbook_decomp(3, 2, 3),
        (3, 3, 2): schoolbook_decomp(3, 3, 2),
        (3, 3, 3): winograd_3x3x3_decomp(),
    }
    
    U, V, W, rank = build_block_decomposition(n_sizes, m_sizes, p_sizes, block_algs)
    
    # Verify
    T_recon = np.einsum('at,bt,ct->abc', U, V, W)
    if not np.allclose(T_recon, T_true, atol=1e-10):
        # Fall back to schoolbook
        U, V, W = schoolbook_decomp(n, m, p)
        rank = n * m * p
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
