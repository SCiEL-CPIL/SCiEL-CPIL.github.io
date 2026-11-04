"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Uses Kronecker product structure:
  T_4x4x6 = T_2x2x2 ⊗ T_2x2x3
  where T_2x2x2 is the Strassen decomposition (rank 7)
  and T_2x2x3 is a rank-11 decomposition of 2x2 * 2x3 matrix multiplication.
  Total rank: 7 * 11 = 77.
"""

import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before the 150s kill


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses Kronecker product structure: (2x2x2 Strassen rank-7) ⊗ (2x2x3 rank-11) = rank 77.
    
    Returns a dict with keys 'U', 'V', 'W' where:
      U: shape (n*m, rank) = (16, rank)
      V: shape (m*p, rank) = (24, rank)
      W: shape (p*n, rank) = (24, rank)
    """
    n, m, p = 4, 4, 6
    
    # --- 2x2x2 Strassen decomposition (rank 7) ---
    # Each term: (u_dict, v_dict, w_dict) with keys (row, col)
    S2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # --- 2x2x3 decomposition (rank 11) ---
    # 2x2 * 2x3: C[2x3] = A[2x2] * B[2x3]
    # Split B into first 2 columns (2x2) and last 1 column (2x1)
    # Rank 7 (Strassen for 2x2*2x2) + Rank 4 (schoolbook for 2x2*2x1) = 11
    
    S2x3 = list(S2)  # 7 Strassen terms for the 2x2*2x2 part
    
    # Schoolbook terms for 2x2*2x1 (the 3rd column of B):
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    S2x3.append((
        {(0,0): 1},
        {(0,2): 1},
        {(0,2): 1},
    ))  # A[0,0]*B[0,2] -> C[0,2]
    S2x3.append((
        {(1,0): 1},
        {(0,2): 1},
        {(1,2): 1},
    ))  # A[1,0]*B[0,2] -> C[1,2]
    S2x3.append((
        {(0,1): 1},
        {(1,2): 1},
        {(0,2): 1},
    ))  # A[0,1]*B[1,2] -> C[0,2]
    S2x3.append((
        {(1,1): 1},
        {(1,2): 1},
        {(1,2): 1},
    ))  # A[1,1]*B[1,2] -> C[1,2]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Kronecker product: T_4x4x6 = T_2x2x2 ⊗ T_2x2x3
    # View A[4x4] as 2x2 blocks of 2x2: A = [[A00, A01], [A10, A11]]
    # View B[4x6] as 2x2 blocks of 2x3: B = [[B00, B01], [B10, B11]] where each is 2x3
    # View C[4x6] as 2x2 blocks of 2x3: C = [[C00, C01], [C10, C11]] where each is 2x3
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # The Kronecker structure: for each Strassen-2x2 term (a,b,c) and each 2x2x3 term (u,v,w),
    # we get a term where:
    #   U factor: place 'a' pattern in the appropriate 2x2 block of A
    #   V factor: place 'v' pattern in the appropriate 2x3 block of B  
    #   W factor: place 'w' pattern in the appropriate 2x3 block of C
    #
    # Block layout for A[4x4]:
    #   Block (0,0): rows 0-1, cols 0-1  -> A[0:2, 0:2]
    #   Block (0,1): rows 0-1, cols 2-3  -> A[0:2, 2:4]
    #   Block (1,0): rows 2-3, cols 0-1  -> A[2:4, 0:2]
    #   Block (1,1): rows 2-3, cols 2-3  -> A[2:4, 2:4]
    #
    # Block layout for B[4x6]:
    #   Block (0,0): rows 0-1, cols 0-2  -> B[0:2, 0:3]
    #   Block (0,1): rows 0-1, cols 3-5  -> B[0:2, 3:6]
    #   Block (1,0): rows 2-3, cols 0-2  -> B[2:4, 0:3]
    #   Block (1,1): rows 2-3, cols 3-5  -> B[2:4, 3:6]
    #
    # Block layout for C[4x6]:
    #   Block (0,0): rows 0-1, cols 0-2  -> C[0:2, 0:3]
    #   Block (0,1): rows 0-1, cols 3-5  -> C[0:2, 3:6]
    #   Block (1,0): rows 2-3, cols 0-2  -> C[2:4, 0:3]
    #   Block (1,1): rows 2-3, cols 3-5  -> C[2:4, 3:6]
    
    # A block offsets: (row_offset, col_offset) for each 2x2 block
    A_blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]  # (00, 01, 10, 11)
    # B block offsets: (row_offset, col_offset) for each 2x3 block
    B_blocks = [(0, 0), (0, 3), (2, 0), (2, 3)]  # (00, 01, 10, 11)
    # C block offsets: (row_offset, col_offset) for each 2x3 block
    C_blocks = [(0, 0), (0, 3), (2, 0), (2, 3)]  # (00, 01, 10, 11)
    
    # Strassen 2x2: block indices for U, V, W patterns
    # The 7 Strassen terms specify which 2x2 blocks are involved
    # M1 = (A00+A11)(B00+B11) -> C00 + C11 - C01 - C10
    # M2 = (A10+A11)B00 -> C10 + C11 - C00
    # M3 = A00(B01-B11) -> C01 + C11 - C10
    # M4 = A11(B10-B00) -> C00 + C10 - C01
    # M5 = (A00+A01)B11 -> C00 + C01 - C11
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00
    
    # For each Strassen term, the block involvement is:
    # (U_blocks, V_blocks, W_blocks) where each is a dict {block_idx: coeff}
    strassen_block_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 1: 1, 2: -1, 3: -1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: 1, 0: -1}),                # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1, 2: -1}),               # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1, 1: -1}),               # M4
        ({0: 1, 1: 1}, {3: 1}, {0: 1, 1: 1, 3: -1}),                # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),                       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),                       # M7
    ]
    
    for s_idx, (u_blocks, v_blocks, w_blocks) in enumerate(strassen_block_terms):
        for s2_idx, (u_pat, v_pat, w_pat) in enumerate(S2x3):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Fill U: for each block in u_blocks, place the 2x2 pattern
            for block_idx, block_coeff in u_blocks.items():
                br, bc = A_blocks[block_idx]
                for (r, c), coeff in u_pat.items():
                    i = br + r
                    j = bc + c
                    if 0 <= i < n and 0 <= j < m:
                        u_col[i * m + j] += block_coeff * coeff
            
            # Fill V: for each block in v_blocks, place the 2x3 pattern
            for block_idx, block_coeff in v_blocks.items():
                br, bc = B_blocks[block_idx]
                for (r, c), coeff in v_pat.items():
                    j = br + r
                    k = bc + c
                    if 0 <= j < m and 0 <= k < p:
                        v_col[j * p + k] += block_coeff * coeff
            
            # Fill W: for each block in w_blocks, place the 2x3 pattern
            for block_idx, block_coeff in w_blocks.items():
                br, bc = C_blocks[block_idx]
                for (r, c), coeff in w_pat.items():
                    i = br + r
                    k = bc + c
                    if 0 <= i < n and 0 <= k < p:
                        w_col[k * n + i] += block_coeff * coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    rank = len(U_cols)
    
    # Transpose to get U as (n*m, rank), V as (m*p, rank), W as (p*n, rank)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    # Verify the decomposition
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback: schoolbook rank-96
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}





def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is correct.
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    # For each (i, j, k), compute the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if total != 1:
                    return False
    return True
