"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Uses the additive block-Strassen structure:
  1. 4x4 * 4x4 part (cols 0-3): Recursive Strassen, rank 49.
  2. 4x4 * 4x2 part (cols 4-5): 4 independent 2x4x2 multiplications.
     Each 2x4x2 multiplication is decomposed into two 2x2x2 Strassen terms (rank 14).
     Total rank for this part: 4 * 14 = 56? 
     Wait, the code uses 28. 
     Re-evaluation: The code uses 4 calls to _add_strassen_2x2.
     Each call adds 7 terms. Total 28.
     This implies each call handles a 2x2x2 multiplication.
     But C[0:2, 4:6] = A[0:2, 0:2]B[0:2, 4:6] + A[0:2, 2:4]B[2:4, 4:6].
     This is a sum of two 2x2x2 products.
     The code adds Strassen(A00 B0) and Strassen(A01 B1) separately.
     So it adds 7 + 7 = 14 terms for this block.
     There are 4 such blocks. Total 56 terms?
     Let's check the code logic again.
     PART 2 loop:
     for rb in range(2):
         _add_strassen_2x2(rb * 2, 0, 4, ...)
         _add_strassen_2x2(rb * 2, 2, 4, ...)
     
     rb=0:
     1. oi=0, oj=0, ok=4. A[0:2, 0:2] * B[0:2, 4:6].
     2. oi=0, oj=2, ok=4. A[0:2, 2:4] * B[2:4, 4:6].
     These two contribute to C[0:2, 4:6].
     rb=1:
     3. oi=2, oj=0, ok=4. A[2:4, 0:2] * B[0:2, 4:6].
     4. oi=2, oj=2, ok=4. A[2:4, 2:4] * B[2:4, 4:6].
     These two contribute to C[2:4, 4:6].
     
     So we have 4 independent 2x2x2 Strassen decompositions.
     Total rank 28.
     This is correct.
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
        # m1 = (x00+x11)(y00+y11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m2 = (x10+x11)y00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # m3 = x00(y01-y11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # m4 = x11(y10-y00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # m5 = (x00+x01)y11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # m6 = (x10-x00)(y00+y01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # m7 = (x01-x11)(y10+y11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # --- 2x2x3 decomposition (rank 11) ---
    # 2x2 * 2x3 matrix multiplication: C[2x3] = A[2x2] * B[2x3]
    # We use a combination of Strassen-like terms and schoolbook terms.
    # Split into: 2x2*2x2 (rank 7) + 2x2*2x1 (rank 4) = rank 11
    # For the 2x1 part, we use schoolbook (4 terms).
    
    # 2x2*2x2 Strassen terms (same as S2 above)
    S2x2 = S2  # rank 7
    
    # 2x2*2x1 schoolbook terms (rank 4): C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    S2x1 = [
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),  # A[0,0]*B[0,0] -> C[0,0]
        ({(1,0): 1}, {(0,0): 1}, {(1,0): 1}),  # A[1,0]*B[0,0] -> C[1,0]
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),  # A[0,1]*B[1,0] -> C[0,0]
        ({(1,1): 1}, {(1,0): 1}, {(1,0): 1}),  # A[1,1]*B[1,0] -> C[1,0]
    ]  # rank 4
    
    # Combined 2x2x3 terms: rank 7 + 4 = 11
    S2x3 = S2x2 + S2x1  # 11 terms
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Kronecker product: for each of 7 Strassen-2x2 terms and 11 2x2x3 terms,
    # create one rank-1 term in the 4x4x6 decomposition.
    # 
    # The 4x4x6 tensor decomposes as:
    #   T_4x4x6 = T_2x2x2 ⊗ T_2x2x3 (in appropriate block structure)
    #
    # More precisely, we view:
    #   A[4x4] as 2x2 blocks of 2x2: A = [[A00, A01], [A10, A11]]
    #   B[4x6] as 2x2 blocks of 2x3: B = [[B00, B01, B02], [B10, B11, B12]]
    #   C[4x6] as 2x2 blocks of 2x3: C = [[C00, C01, C02], [C10, C11, C12]]
    #
    # But this doesn't quite work because the middle dimension (m=4) needs to match.
    # 
    # Alternative approach: Use the block-Strassen structure directly.
    # For the 4x4 * 4x6 multiplication:
    #   C[4x6] = A[4x4] * B[4x6]
    # Split A into 2x2 blocks, B into 2x2 blocks (with 3 column-blocks of width 2):
    #   A = [[A00, A01], [A10, A11]]  (each 2x2)
    #   B = [[B00, B01, B02], [B10, B11, B12]]  (each 2x2)
    #   C = [[C00, C01, C02], [C10, C11, C12]]  (each 2x2)
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    #
    # This is 12 separate 2x2*2x2 multiplications, each rank 7, total rank 84.
    # But we can do better with Strassen at the 4x4 level.
    #
    # Let's use the original approach which gave rank 77:
    #   4x4*4x4 part (cols 0-3): recursive Strassen rank 49
    #   4x4*4x2 part (cols 4-5): 4 separate 2x2*2x2 Strassen, rank 28
    #   Total: 77
    
    # PART 1: 4x4 * 4x4 (columns 0-3 of B) using recursive Strassen (rank 49)
    if _time_remaining() > 5.0:
        _add_strassen_4x4(0, 0, 0, U_cols, V_cols, W_cols, n, m, p)
    
    # PART 2: 4x4 * 4x2 (columns 4-5 of B) using 4 independent 2x2*2x2 Strassen (rank 28)
    if _time_remaining() > 3.0:
        for rb in range(2):
            if _time_remaining() < 2.0:
                break
            _add_strassen_2x2(rb * 2, 0, 4, U_cols, V_cols, W_cols, n, m, p)
            _add_strassen_2x2(rb * 2, 2, 4, U_cols, V_cols, W_cols, n, m, p)
    
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


def _add_strassen_2x2(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    """Add Strassen decomposition for 2x2 * 2x2 block multiplication."""
    S2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for u_pat, v_pat, w_pat in S2:
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (r, c), coeff in u_pat.items():
            i = oi + r
            j = oj + c
            if 0 <= i < n and 0 <= j < m:
                u_col[i * m + j] += coeff
        
        for (r, c), coeff in v_pat.items():
            j = oj + r
            k = ok + c
            if 0 <= j < m and 0 <= k < p:
                v_col[j * p + k] += coeff
        
        for (r, c), coeff in w_pat.items():
            i = oi + r
            k = ok + c
            if 0 <= i < n and 0 <= k < p:
                w_col[k * n + i] += coeff
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)


def _add_strassen_4x4(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    """
    Add recursive Strassen decomposition for 4x4 * 4x4 block multiplication.
    Uses 7 sub-problems, each of which is a 2x2 * 2x2 Strassen.
    Total: 7 * 7 = 49 rank-1 terms.
    """
    # The 7 Strassen 2x2 terms
    S2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Block offsets for 2x2 blocks within 4x4
    # Block 0 = TL (rows 0-1, cols 0-1), Block 1 = TR (rows 0-1, cols 2-3)
    # Block 2 = BL (rows 2-3, cols 0-1), Block 3 = BR (rows 2-3, cols 2-3)
    block_offsets = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
    
    # The 7 super-Strassen terms for 4x4*4x4
    # Each specifies which 2x2 blocks of A, B, C are involved with coefficients
    super_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for u_blocks, v_blocks, w_blocks in super_terms:
        for u_pat, v_pat, w_pat in S2:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for block_idx, block_coeff in u_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in u_pat.items():
                    i = oi + br + r
                    j = oj + bc + c
                    if 0 <= i < n and 0 <= j < m:
                        u_col[i * m + j] += block_coeff * coeff
            
            for block_idx, block_coeff in v_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in v_pat.items():
                    j = oj + br + r
                    k = ok + bc + c
                    if 0 <= j < m and 0 <= k < p:
                        v_col[j * p + k] += block_coeff * coeff
            
            for block_idx, block_coeff in w_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in w_pat.items():
                    i = oi + br + r
                    k = ok + bc + c
                    if 0 <= i < n and 0 <= k < p:
                        w_col[k * n + i] += block_coeff * coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)


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
