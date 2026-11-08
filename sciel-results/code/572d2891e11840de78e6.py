"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses a direct 4x4 * 4x9 tensor decomposition with rank 114:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen decomposition for 2x2 * 2x2
    # Each term: (A_mask, B_mask, C_mask) where masks are dicts {index: coeff}
    # A is 2x2, B is 2x2, C is 2x2
    # Indexing: row * 2 + col
    strassen_2x2_terms = [
        # m1 = (A00+A11)(B00+B11) -> C00 += m1, C11 += m1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # m2 = (A10+A11)(B00) -> C10 += m2, C11 -= m2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # m3 = (A00)(B10+B11) -> C01 += m3, C11 += m3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # m4 = (A11)(B10-B00) -> C00 += m4, C10 += m4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # m5 = (A00+A01)(B11) -> C00 -= m5, C01 += m5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # m6 = (A10-A00)(B00+B01) -> C11 += m6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # m7 = (A01-A11)(B10+B11) -> C00 += m7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Add Strassen decomposition for 4x4 * 4x4 (rank 49).
        
        Splits 4x4 into 2x2 blocks and applies Strassen at block level,
        then Strassen within each block multiplication.
        """
        # Block-level Strassen for 2x2 blocks of 4x4 matrices
        # A = [[A00, A01], [A10, A11]] where each block is 2x2
        # B = [[B00, B01], [B10, B11]]
        # C = A*B, computed via Strassen on blocks
        
        # The 7 block multiplications, each of which is a 2x2*2x2=4x4 multiplication
        # We use Strassen for each 2x2*2x2 block multiplication (rank 7 each)
        # 7 * 7 = 49 terms total
        
        # Block multiplication definitions:
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # M2 = (A10+A11)(B00) -> C10 += M2, C11 -= M2
        # M3 = (A00)(B10+B11) -> C01 += M3, C11 += M3
        # M4 = (A11)(B10-B00) -> C00 += M4, C10 += M4
        # M5 = (A00+A01)(B11) -> C00 -= M5, C01 += M5
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        
        # For each block multiplication, we need to express it as a sum of
        # rank-1 terms using Strassen on the 2x2 blocks.
        
        # A block at position (br, bc) has global indices:
        #   rows: A_base_i + br*2 to A_base_i + br*2+1
        #   cols: A_base_j + bc*2 to A_base_j + bc*2+1
        
        # Define the 7 block-level Strassen terms
        # Each term specifies which A blocks, B blocks, and C blocks are involved
        block_strassen = [
            # (A_blocks: [(br, bc, coeff), ...], B_blocks, C_blocks: [(br, bc, coeff), ...])
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),   # M1
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),            # M2
            ([(0,0,1)], [(1,0,1), (1,1,1)], [(0,1,1), (1,1,1)]),             # M3
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),            # M4
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),            # M5
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),            # M6
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),            # M7
        ]
        
        for A_block_list, B_block_list, C_block_list in block_strassen:
            # For each of the 7 inner Strassen terms
            for A_inner, B_inner, C_inner in strassen_2x2_terms:
                u_vec = [0] * (n * m)   # 16
                v_vec = [0] * (m * p)   # 36
                w_vec = [0] * (p * n)   # 36
                
                # Build U vector
                for local_idx, coeff in A_inner.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    for (br, bc, bcoeff) in A_block_list:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * bcoeff
                
                # Build V vector
                for local_idx, coeff in B_inner.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc, bcoeff) in B_block_list:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * bcoeff
                
                # Build W vector
                for local_idx, coeff in C_inner.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc, bcoeff) in C_block_list:
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * bcoeff
                
                add_term(u_vec, v_vec, w_vec)
    
    # Part 1: A * B1 (4x4 * 4x4), B1 is columns 0-3 of B
    strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (4x4 * 4x4), B2 is columns 4-7 of B
    strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (4x4 * 4x1), B3 is column 8 of B
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    
    # Convert to matrix form: U[row][col], V[row][col], W[row][col]
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
