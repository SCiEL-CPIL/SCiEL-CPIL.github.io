"""Matrix multiplication 4x4 x 4x5: rank-65 decomposition (49 Strassen + 16 schoolbook)."""
import time

DIMENSIONS = [4, 4, 5]
TIME_LIMIT = 150.0

def _expand_strassen_2x2():
    """Return (U, V, W) lists of 7 terms for 2x2x2 Strassen.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    where C = A*B, A is 2x2, B is 2x2, C is 2x2.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A00-A10)(B00+B01)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # U: A coefficients, V: B coefficients, W: C coefficients
    # Term 0: M1
    U[0][0] = 1; U[3][0] = 1          # A00 + A11
    V[0][0] = 1; V[3][0] = 1          # B00 + B11
    W[0][0] = 1                        # C00
    W[3][0] = 1                        # C11

    # Term 1: M2
    U[2][1] = 1; U[3][1] = 1          # A10 + A11
    V[0][1] = 1                        # B00
    W[2][1] = 1                        # C10
    W[3][1] = -1                       # C11

    # Term 2: M3
    U[0][2] = 1                        # A00
    V[1][2] = 1; V[3][2] = -1          # B01 - B11
    W[1][2] = 1                        # C01
    W[3][2] = 1                        # C11

    # Term 3: M4
    U[3][3] = 1                        # A11
    V[2][3] = 1; V[0][3] = -1          # B10 - B00
    W[0][3] = 1                        # C00
    W[2][3] = 1                        # C10

    # Term 4: M5
    U[0][4] = 1; U[2][4] = 1          # A00 + A10
    V[3][4] = 1                        # B11
    W[0][4] = -1                       # C00
    W[1][4] = 1                        # C01

    # Term 5: M6
    U[1][5] = 1; U[3][5] = -1          # A01 - A11
    V[2][5] = 1; V[3][5] = 1          # B10 + B11
    W[3][5] = 1                        # C11

    # Term 6: M7
    U[0][6] = 1; U[2][6] = -1          # A00 - A10
    V[0][6] = 1; V[1][6] = 1          # B00 + B01
    W[0][6] = 1                        # C00

    return U, V, W


def _expand_strassen_4x4():
    """Return (U, V, W) lists of 49 terms for 4x4x4 via Strassen.
    U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    base_U, base_V, base_W = _expand_strassen_2x2()
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    for a in range(7):
        for b in range(7):
            c = a * 7 + b
            # Block (a//... no, a indexes the 7 terms of outer Strassen on 2x2 blocks)
            # Actually: outer Strassen has 7 terms M1..M7 applied to 2x2 sub-blocks.
            # Each outer term corresponds to an inner Strassen term.
            # The outer Strassen operates on 2x2 blocks, so:
            # - Outer term a uses specific 2x2 sub-matrices of A (4x4)
            # - Inner term b uses specific 2x2 sub-matrices within those
            pass
    # Simpler: just hardcode via recursion
    # 4x4 = 2x2 blocks of 2x2.
    # Outer Strassen: 7 multiplications of 2x2 blocks.
    # Each 2x2 multiplication uses 7 inner terms -> 49 total.
    
    # Let's define the outer Strassen structure for 2x2 blocks:
    # A is 4x4, split into A00,A01,A10,A11 (each 2x2)
    # B is 4x4, split into B00,B01,B10,B11 (each 2x2)
    # C is 4x4, split into C00,C01,C10,C11 (each 2x2)
    #
    # Outer M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
    # Outer M2 = (A10+A11)*B00 -> contributes to C10, C11
    # Outer M3 = A00*(B01-B11) -> contributes to C01, C11
    # Outer M4 = A11*(B10-B00) -> contributes to C00, C10
    # Outer M5 = (A00+A10)*B11 -> contributes to C00, C01
    # Outer M6 = (A01-A11)*(B10+B11) -> contributes to C11
    # Outer M7 = (A00-A10)*(B00+B01) -> contributes to C00
    
    # For each outer term, we multiply two 2x2 matrices using inner Strassen (7 terms).
    # So we need to map: for outer term t, which A-blocks and B-blocks are involved?
    # Then for each inner term, which elements of those blocks?
    
    # Let me define outer terms as (a_indices, b_indices, sign) for each factor:
    # a_indices: list of (block_row, block_col) for A
    # b_indices: list of (block_row, block_col) for B
    
    outer_terms = [
        # (a_list, b_list, c_contribs) where c_contribs is list of (block_row, block_col, sign)
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5
        ([(0,0,1),(1,0,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(1,1,1)]),
        # M7
        ([(0,0,1),(1,0,-1)], [(0,0,1),(0,1,1)], [(0,0,1)]),
    ]
    
    # Inner Strassen terms for 2x2:
    # Returns (u_idx, v_idx, w_idx, sign) for each of 7 terms
    # u_idx: (row, col) in 2x2 A-block
    # v_idx: (row, col) in 2x2 B-block  
    # w_idx: (row, col) in 2x2 C-block
    inner_U = _expand_strassen_2x2()[0]  # 4x7
    inner_V = _expand_strassen_2x2()[1]  # 4x7
    inner_W = _expand_strassen_2x2()[2]  # 4x7
    
    for t_idx, (a_list, b_list, c_list) in enumerate(outer_terms):
        for inner_t in range(7):
            c = t_idx * 7 + inner_t
            # Process A factor
            for (br, bc, sign) in a_list:
                for i in range(2):
                    for j in range(2):
                        val = inner_U[i*2+j][inner_t]
                        if val != 0:
                            U[br*4 + i, bc*4 + j][c] += val * sign
            # Process B factor
            for (br, bc, sign) in b_list:
                for i in range(2):
                    for j in range(2):
                        val = inner_V[i*2+j][inner_t]
                        if val != 0:
                            V[br*4 + i, bc*4 + j][c] += val * sign
            # Process C factor
            for (br, bc, sign) in c_list:
                for i in range(2):
                    for j in range(2):
                        val = inner_W[i*2+j][inner_t]
                        if val != 0:
                            W[br*4 + i, bc*4 + j][c] += val * sign
    
    return U, V, W


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Get 4x4x4 Strassen decomposition (49 terms)
    S_U, S_V, S_W = _expand_strassen_4x4()
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy Strassen terms for first 4 columns of B
    for i in range(n * m):
        for c in range(49):
            U[i][c] = S_U[i][c]
    for j in range(m * 4):
        for c in range(49):
            V[j][c] = S_V[j][c]
    for k in range(4 * n):
        for c in range(49):
            W[k][c] = S_W[k][c]
    
    # Add 16 schoolbook terms for the 5th column (k=4)
    for i in range(n):
        for j in range(m):
            c = 49 + i * m + j
            U[i * m + j][c] = 1
            V[j * p + 4][c] = 1
            W[4 * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
