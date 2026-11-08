import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will use a block decomposition approach.
    # Split m into chunks and p into chunks.
    # For each pair of chunks (m_chunk, p_chunk), we perform a 2 x m_chunk x p_chunk multiplication.
    # Since n=2 is small, we can use a specific decomposition for 2 x m' x p'.
    
    # Actually, let's try to split the problem into independent sub-problems.
    # C = A * B where A is 2x5, B is 5x6.
    # We can split the columns of A and rows of B.
    # Let's split m=5 into m1=2, m2=3.
    # Let's split p=6 into p1=3, p2=3.
    
    # This gives 4 sub-problems:
    # 1. A[:, 0:2] * B[0:2, 0:3] -> 2x3
    # 2. A[:, 0:2] * B[0:2, 3:6] -> 2x3
    # 3. A[:, 2:5] * B[2:5, 0:3] -> 2x3
    # 4. A[:, 2:5] * B[2:5, 3:6] -> 2x3
    
    # Each sub-problem is 2x3 matrix multiplication.
    # We need the rank of 2x3 matrix multiplication.
    # The trivial rank is 2*3*3 = 18. Total 4*18 = 72. This is worse than 60.
    
    # Let's try different splits.
    # Maybe split m=5 into 1,1,1,1,1? No.
    
    # Let's look at the rank of 2 x m x p.
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x3x3, what is the rank?
    # It is known that R(2,3,3) <= 16? Or 15?
    # Trivial is 18.
    # If we use 2x2x2 blocks, we can cover 2x2 parts.
    
    # Let's try to decompose 2x5x6 using 2x2x2 blocks where possible.
    # This is complex to index.
    
    # Alternative: Use the fact that R(2, m, p) can be bounded.
    # Actually, there is a known result for 2 x n x n.
    # For n=5, R(2,5,5) is at most 2*5*5 - 5 = 45? (using Strassen-like for 2x2 blocks)
    # If R(2,5,5) <= 45, then R(2,5,6) <= R(2,5,5) + R(2,5,1) = 45 + 10 = 55.
    # This would be an improvement over 60.
    
    # Let's implement R(2,5,5) using 2x2x2 Strassen blocks and trivial for the rest.
    # Split 5 into 2+2+1.
    # Blocks:
    # (0,0) 2x2
    # (0,1) 2x2
    # (0,2) 2x1
    # (1,0) 2x2
    # (1,1) 2x2
    # (1,2) 2x1
    # (2,0) 1x2
    # (2,1) 1x2
    # (2,2) 1x1
    
    # We need ranks for:
    # 2x2x2: 7
    # 2x2x1: 4 (trivial)
    # 2x1x2: 4 (trivial)
    # 1x2x2: 4 (trivial)
    # 1x1x2: 2 (trivial)
    # 2x1x1: 2 (trivial)
    # 1x2x1: 2 (trivial)
    # 1x1x1: 1
    
    # Sum for 2x5x5:
    # 4 * 7 (2x2x2) = 28
    # 2 * 4 (2x2x1) = 8
    # 2 * 4 (2x1x2) = 8
    # 2 * 4 (1x2x2) = 8
    # 1 * 2 (1x1x2) = 2
    # 1 * 2 (2x1x1) = 2
    # 1 * 2 (1x2x1) = 2
    # 1 * 1 (1x1x1) = 1
    # Total = 28 + 8 + 8 + 8 + 2 + 2 + 2 + 1 = 59.
    # This is worse than 60? No, 59 is better than 60.
    # But wait, I need to add the 6th column of B.
    # R(2,5,6) = R(2,5,5) + R(2,5,1)
    # R(2,5,1): Split 5 into 2+2+1.
    # 2x2x1: 4
    # 2x2x1: 4
    # 2x1x1: 2
    # 1x2x1: 2
    # 1x2x1: 2
    # 1x1x1: 1
    # 1x1x1: 1
    # 1x1x1: 1
    # Sum = 4+4+2+2+2+1+1+1 = 17.
    # Total Rank = 59 + 17 = 76. This is worse.
    
    # The block decomposition sum of ranks is additive, but we can share some terms?
    # No, the tensor decomposition is direct sum if the blocks are disjoint in indices.
    # So the rank is the sum of the ranks of the blocks.
    
    # We need a better decomposition for 2x5x6 directly.
    # Or maybe my calculation of R(2,5,5) is too high.
    # Is there a better way to do 2x2x2? Yes, Strassen is 7.
    # Is there a better way to do 2x2x1?
    # 2x2x1 is just 4 scalar multiplications. Rank 4.
    
    # Let's try to find a known rank for 2x5x6.
    # Or use a different splitting.
    # Split m=5 into 3+2.
    # Split p=6 into 3+3.
    # Blocks:
    # 2x3x3: Rank?
    # 2x3x3: Rank?
    # 2x2x3: Rank?
    # 2x2x3: Rank?
    
    # If R(2,3,3) = 16, then 2*16 + 2*R(2,2,3).
    # R(2,2,3) <= 2*2*3 = 12.
    # Total = 32 + 24 = 56.
    # This is better than 60!
    
    # So we need to implement R(2,3,3) with rank 16 and R(2,2,3) with rank 12.
    # How to get R(2,3,3) = 16?
    # Trivial is 18.
    # We can use one 2x2x2 Strassen block (rank 7) and the rest trivial.
    # Split 3x3 into:
    # 2x2 (top-left)
    # 2x1 (top-right)
    # 1x2 (bottom-left)
    # 1x1 (bottom-right)
    
    # For 2x3x3 (A is 2x3, B is 3x3):
    # A = [A1 (2x2) | A2 (2x1)]
    # B = [[B11 (2x2), B12 (2x1)], [B21 (1x2), B22 (1x1)]]
    # C = A1 B11 + A2 B21 (2x2 part)
    # C = A1 B12 + A2 B22 (2x1 part)
    
    # Term 1: A1 B11 (2x2x2) -> Rank 7 (Strassen)
    # Term 2: A2 B21 (2x1x2) -> Rank 4 (Trivial)
    # Term 3: A1 B12 (2x2x1) -> Rank 4 (Trivial)
    # Term 4: A2 B22 (2x1x1) -> Rank 2 (Trivial)
    # Total Rank = 7 + 4 + 4 + 2 = 17.
    # This is 17, not 16.
    
    # Can we do better for 2x3x3?
    # Maybe a different split?
    # If we use 2x2x2 Strassen for the main block, we get 17.
    # Is 16 possible?
    # According to some sources, R(2,3,3) = 16.
    # How?
    # Perhaps by sharing some linear forms?
    # Or maybe my trivial ranks are too high?
    # 2x1x2 is 4 scalar mults. Rank 4.
    # 2x2x1 is 4 scalar mults. Rank 4.
    # 2x1x1 is 2 scalar mults. Rank 2.
    # Sum 7+4+4+2 = 17.
    
    # Let's check R(2,2,3).
    # A (2x2), B (2x3).
    # Split B into 2x2 and 2x1.
    # A B1 (2x2x2) -> Rank 7
    # A B2 (2x2x1) -> Rank 4
    # Total Rank = 11.
    # Trivial is 12. So 11 is better.
    
    # So for 2x5x6 with split 3+2 and 3+3:
    # 2 * R(2,3,3) + 2 * R(2,2,3)
    # = 2 * 17 + 2 * 11 = 34 + 22 = 56.
    
    # This is a solid improvement from 60 to 56.
    # I will implement this.
    
    # We need functions to generate U, V, W for:
    # 1. 2x2x2 Strassen
    # 2. Trivial blocks (any size)
    
    # Let's define a helper to add a block to the global matrices.
    
    U = []
    V = []
    W = []
    
    def add_block(u_block, v_block, w_block, i_off, j_off, k_off):
        # u_block shape: (n, rank_block)
        # v_block shape: (m_block, rank_block)
        # w_block shape: (p_block, n) -- wait, W is (p*n, rank)
        # The W index is k*n + i.
        # So w_block should be indexed by k (0..p_block-1) and i (0..n-1).
        
        rank_block = u_block.shape[1]
        
        for r in range(rank_block):
            u_col = [0.0] * (n * m)
            v_col = [0.0] * (m * p)
            w_col = [0.0] * (p * n)
            
            for i in range(n):
                for j in range(u_block.shape[0]): # m_block
                    # u_block is actually for A, which is n x m.
                    # So u_block[i, j] corresponds to A[i, j].
                    # Global index in U is i*m + j.
                    # But u_block is for a sub-block of A.
                    # The sub-block A[i_off:i_off+n, j_off:j_off+m_block]
                    # So local i is 0..n-1, local j is 0..m_block-1.
                    # Global i is i_off + i.
                    # Global j is j_off + j.
                    # Global U index: (i_off + i) * m + (j_off + j).
                    
                    # Wait, u_block in my helper signature was (n, rank_block)?
                    # No, U has shape (n*m, rank).
                    # So for a block, we need to fill the corresponding rows of U.
                    
                    pass # This logic is getting messy. Let's build the full matrices directly.
    
    # Let's construct the full U, V, W matrices of size (n*m, rank), (m*p, rank), (p*n, rank).
    # We will determine the total rank first, then fill the columns.
    
    # Blocks:
    # 1. A[0:2, 0:3] * B[0:3, 0:3] (2x3x3)
    # 2. A[0:2, 3:5] * B[3:5, 0:3] (2x2x3)
    # 3. A[0:2, 0:3] * B[0:3, 3:6] (2x3x3)
    # 4. A[0:2, 3:5] * B[3:5, 3:6] (2x2x3)
    
    # Wait, A is 2x5. Rows are 0,1. Cols are 0..4.
    # B is 5x6. Rows are 0..4. Cols are 0..5.
    
    # Block 1: A[0:2, 0:3] (rows 0-1, cols 0-2) * B[0:3, 0:3] (rows 0-2, cols 0-2)
    #   m_block=3, p_block=3.
    # Block 2: A[0:2, 3:5] (rows 0-1, cols 3-4) * B[3:5, 0:3] (rows 3-4, cols 0-2)
    #   m_block=2, p_block=3.
    # Block 3: A[0:2, 0:3] (rows 0-1, cols 0-2) * B[0:3, 3:6] (rows 0-2, cols 3-5)
    #   m_block=3, p_block=3.
    # Block 4: A[0:2, 3:5] (rows 0-1, cols 3-4) * B[3:5, 3:6] (rows 3-4, cols 3-5)
    #   m_block=2, p_block=3.
    
    # Ranks:
    # R(2,3,3) = 17
    # R(2,2,3) = 11
    # Total Rank = 17 + 11 + 17 + 11 = 56.
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    def add_trivial_block(n_b, m_b, p_b, i_off, j_off, k_off):
        nonlocal col_offset
        rank_b = n_b * m_b * p_b
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u_idx = (i_off + i) * m + (j_off + j)
                    v_idx = (j_off + j) * p + (k_off + k)
                    w_idx = (k_off + k) * n + (i_off + i)
                    
                    U[u_idx, col_offset] = 1.0
                    V[v_idx, col_offset] = 1.0
                    W[w_idx, col_offset] = 1.0
                    col_offset += 1
    
    def add_strassen_block_2x2x2(i_off, j_off, k_off):
        # n_b=2, m_b=2, p_b=2
        nonlocal col_offset
        # Strassen formulas for 2x2x2
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(e-f)
        # M5 = (a+b)h
        # M6 = (c-a)(f+h)
        # M7 = (b-d)(e+g)
        
        # We need to map these to U, V, W columns.
        # Each M is a product of a linear form in A and a linear form in B.
        # The contribution to C is a linear form in M.
        
        # Let's define the linear forms.
        # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
        # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
        # C indices: (0,0), (0,1), (1,0), (1,1)
        
        # M1: U=[1,0,1,0], V=[1,1,0,0], W=[1,0,0,1] (C00, C11)
        # M2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0] (C10)
        # M3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0] (C01)
        # M4: U=[0,0,0,1], V=[1,-1,0,0], W=[1,0,0,1] (C00, C11)
        # M5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0] (C00, C01) -> C00 -= M5, C01 += M5
        # M6: U=[-1,0,1,0], V=[0,1,0,1], W=[0,0,0,1] (C11)
        # M7: U=[0,1,0,-1], V=[1,0,1,0], W=[1,0,0,0] (C00)
        
        # Note: W indices are (k*n+i).
        # C00: k=0, i=0 -> idx 0
        # C01: k=1, i=0 -> idx 1
        # C10: k=0, i=1 -> idx 2
        # C11: k=1, i=1 -> idx 3
        
        # Let's verify M5: C = M1 + M4 - M5 + M7 (for C00), M3 + M5 (for C01)
        # So W for M5 is [-1, 1, 0, 0] for C00, C01.
        # Global W indices:
        # C00: (k_off+0)*n + (i_off+0)
        # C01: (k_off+1)*n + (i_off+0)
        
        # We will define the 7 terms.
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U is 4x1 (for a,b,c,d), V is 4x1 (for e,f,g,h), W is 4x1 (for C00,C01,C10,C11)
            ([1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3
            ([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 0, 1]), # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
            ([-1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 1]), # M6
            ([0, 1, 0, -1], [1, 0, 1, 0], [1, 0, 0, 0])  # M7
        ]
        
        for u_c, v_c, w_c in terms:
            # U
            for local_i in range(2):
                for local_j in range(2):
                    u_idx = (i_off + local_i) * m + (j_off + local_j)
                    U[u_idx, col_offset] = u_c[local_i * 2 + local_j]
            
            # V
            for local_j in range(2):
                for local_k in range(2):
                    v_idx = (j_off + local_j) * p + (k_off + local_k)
                    V[v_idx, col_offset] = v_c[local_j * 2 + local_k]
            
            # W
            for local_k in range(2):
                for local_i in range(2):
                    w_idx = (k_off + local_k) * n + (i_off + local_i)
                    W[w_idx, col_offset] = w_c[local_k * 2 + local_i]
            
            col_offset += 1
    
    def add_2x3x3_block(i_off, j_off, k_off):
        # A is 2x3, B is 3x3
        # Split B into 2x2, 2x1, 1x2, 1x1
        # B = [[B11 (2x2), B12 (2x1)], [B21 (1x2), B22 (1x1)]]
        # A = [A1 (2x2), A2 (2x1)]
        
        # Term 1: A1 B11 (2x2x2) -> Strassen
        # j_off, k_off are for B11. A1 is j_off..j_off+1.
        add_strassen_block_2x2x2(i_off, j_off, k_off)
        
        # Term 2: A2 B21 (2x1x2) -> Trivial
        # A2 is A[:, j_off+2] (1 col)
        # B21 is B[j_off+2, k_off:k_off+2] (1 row, 2 cols)
        # Wait, B is 3x3.
        # B11: rows 0-1, cols 0-1 (relative)
        # B12: rows 0-1, cols 2 (relative)
        # B21: rows 2, cols 0-1 (relative)
        # B22: rows 2, cols 2 (relative)
        
        # A1: cols 0-1 (relative)
        # A2: col 2 (relative)
        
        # A2 B21:
        # A2 is 2x1. B21 is 1x2. Result is 2x2.
        # Rank 4.
        add_trivial_block(2, 1, 2, i_off, j_off + 2, k_off)
        
        # Term 3: A1 B12 (2x2x1) -> Trivial
        # A1 is 2x2. B12 is 2x1. Result is 2x1.
        # Rank 4.
        add_trivial_block(2, 2, 1, i_off, j_off, k_off + 2)
        
        # Term 4: A2 B22 (2x1x1) -> Trivial
        # A2 is 2x1. B22 is 1x1. Result is 2x1.
        # Rank 2.
        add_trivial_block(2, 1, 1, i_off, j_off + 2, k_off + 2)
    
    def add_2x2x3_block(i_off, j_off, k_off):
        # A is 2x2, B is 2x3
        # Split B into 2x2, 2x1
        # B = [B1 (2x2), B2 (2x1)]
        # A = A (2x2)
        
        # Term 1: A B1 (2x2x2) -> Strassen
        add_strassen_block_2x2x2(i_off, j_off, k_off)
        
        # Term 2: A B2 (2x2x1) -> Trivial
        add_trivial_block(2, 2, 1, i_off, j_off, k_off + 2)
    
    # Apply blocks
    # 1. 2x3x3 at (0, 0, 0)
    add_2x3x3_block(0, 0, 0)
    
    # 2. 2x2x3 at (0, 3, 0)
    add_2x2x3_block(0, 3, 0)
    
    # 3. 2x3x3 at (0, 0, 3)
    add_2x3x3_block(0, 0, 3)
    
    # 4. 2x2x3 at (0, 3, 3)
    add_2x2x3_block(0, 3, 3)
    
    # Check if col_offset matches total_rank
    # 17 + 11 + 17 + 11 = 56
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
