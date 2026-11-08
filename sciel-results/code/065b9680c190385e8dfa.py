import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We'll build a rank-14 decomposition using two 7-term Strassen decompositions
    # for the two 4x4 halves of the 4x8 matrix B.
    
    rank = 14
    U = [[0.0] * rank for _ in range(n * m)]   # (16, 14)
    V = [[0.0] * rank for _ in range(m * p)]   # (32, 14)
    W = [[0.0] * rank for _ in range(p * n)]   # (32, 14)
    
    # Define the 7 Strassen terms for a 2x2 block matrix multiply (2x2 x 2x2 -> 2x2)
    # Each term is defined by which 2x2 blocks of A and B are involved, and which
    # blocks of the output C get the contribution.
    
    # For a 4x4 matrix, we partition into 2x2 blocks:
    # Block (bi, bj) where bi, bj in {0,1} covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # Entry (i,j) belongs to block (i//2, j//2)
    
    # Strassen terms for 2x2 block multiply:
    # M1: (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # M2: (A10+A11)*B00       -> C10 += M2, C11 -= M2
    # M3: A00*(B01-B11)       -> C01 += M3, C11 += M3
    # M4: A11*(B10-B00)       -> C00 += M4, C10 += M4
    # M5: (A00+A01)*B11       -> C00 -= M5, C01 += M5
    # M6: (A10-A00)*(B00+B01) -> C11 += M6
    # M7: (A01-A11)*(B10+B11) -> C00 += M7
    
    # For each Strassen term t, define:
    #   a_blocks[t]: list of (block_row, block_col, sign) for A
    #   b_blocks[t]: list of (block_row, block_col, sign) for B
    #   c_blocks[t]: list of (block_row, block_col, sign) for C
    
    a_blocks = [
        [(0,0,1), (1,1,1)],           # M1: A00 + A11
        [(1,0,1), (1,1,1)],           # M2: A10 + A11
        [(0,0,1)],                    # M3: A00
        [(1,1,1)],                    # M4: A11
        [(0,0,1), (0,1,1)],           # M5: A00 + A01
        [(1,0,1), (0,0,-1)],          # M6: A10 - A00
        [(0,1,1), (1,1,-1)],          # M7: A01 - A11
    ]
    b_blocks = [
        [(0,0,1), (1,1,1)],           # M1: B00 + B11
        [(0,0,1)],                    # M2: B00
        [(0,1,1), (1,1,-1)],          # M3: B01 - B11
        [(1,0,1), (0,0,-1)],          # M4: B10 - B00
        [(1,1,1)],                    # M5: B11
        [(0,0,1), (0,1,1)],           # M6: B00 + B01
        [(1,0,1), (1,1,1)],           # M7: B10 + B11
    ]
    c_blocks = [
        [(0,0,1), (1,1,1)],           # M1: C00 + C11
        [(1,0,1), (1,1,-1)],          # M2: C10 - C11
        [(0,1,1), (1,1,1)],           # M3: C01 + C11
        [(0,0,1), (1,0,1)],           # M4: C00 + C10
        [(0,0,-1), (0,1,1)],          # M5: -C00 + C01
        [(1,1,1)],                    # M6: C11
        [(0,0,1)],                    # M7: C00
    ]
    
    def fill_blocks(U, V, W, a_blk, b_blk, c_blk, rank_offset, b_col_offset, c_col_offset):
        """
        Fill in U, V, W for one Strassen term.
        a_blk, b_blk, c_blk: lists of (block_row, block_col, sign)
        rank_offset: which column of U, V, W to fill
        b_col_offset: offset in the column index of B (for the 4x8 case)
        c_col_offset: offset in the column index of C (for the 4x8 case)
        """
        # Fill U: for each (br, bc, s) in a_blk, add s to U entries in block (br, bc)
        for (br, bc, s) in a_blk:
            for di in range(2):
                for dj in range(2):
                    i = 2 * br + di
                    j = 2 * bc + dj
                    U[i * m + j][rank_offset] += s
        
        # Fill V: for each (br, bc, s) in b_blk
        # B is 4x8, block (br, bc) covers rows 2*br..2*br+1, cols (b_col_offset + 2*bc)..(b_col_offset + 2*bc+1)
        for (br, bc, s) in b_blk:
            for di in range(2):
                for dj in range(2):
                    i = 2 * br + di
                    j = b_col_offset + 2 * bc + dj
                    V[i * p + j][rank_offset] += s
        
        # Fill W: for each (br, bc, s) in c_blk
        # C is 4x8, block (br, bc) covers rows 2*br..2*br+1, cols (c_col_offset + 2*bc)..(c_col_offset + 2*bc+1)
        # W index = k*n + i where k is col, i is row
        for (br, bc, s) in c_blk:
            for di in range(2):
                for dj in range(2):
                    i = 2 * br + di
                    k = c_col_offset + 2 * bc + dj
                    W[k * n + i][rank_offset] += s
    
    # First 7 terms: multiply A (4x4) by B_left (4x4, cols 0-3)
    for t in range(7):
        fill_blocks(U, V, W, a_blocks[t], b_blocks[t], c_blocks[t], t, 0, 0)
    
    # Last 7 terms: multiply A (4x4) by B_right (4x4, cols 4-7)
    for t in range(7):
        fill_blocks(U, V, W, a_blocks[t], b_blocks[t], c_blocks[t], 7 + t, 4, 4)
    
    # Verify
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        _verify(U, V, W, n, m, p)
    
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify the decomposition by testing with random matrices."""
    np.random.seed(12345)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        rank = len(U[0])
        for r in range(rank):
            a_proj = sum(U[i * m + j][r] * A[i][j] for i in range(n) for j in range(m))
            b_proj = sum(V[j * p + k][r] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_computed[i][k] += W[k * n + i][r] * a_proj * b_proj
        
        err = np.max(np.abs(C_computed - C_expected))
        if err > 1e-9:
            raise ValueError(f"Verification failed: max error = {err}")
