"""
Tensor decomposition for 4x4 x 4x7 matrix multiplication.

Approach:
  - Split the 7 columns into 4 + 3.
  - Use recursive Strassen for the 4x4x4 part (rank 49).
  - Use schoolbook for the 4x4x3 part (rank 48).
  - Total rank: 97.

The Strassen decomposition is built by applying the 7-term identity
twice: once to split 4x4x4 into 2x(2x2)x(2x2) blocks, then again
inside each 2x2x2 block multiplication.

Convention:
  U: (n*m, rank) = (16, 97)
  V: (m*p, rank) = (28, 97)
  W: (p*n, rank) = (28, 97)
  W[k*n + i, r]  (trace convention)

  The tensor is: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
  which corresponds to C[i,k] = sum_j A[i,j] * B[j,k]
"""

DIMENSIONS = [4, 4, 7]

# Precompute the 2x2x2 Strassen rank-7 coefficients.
# Each term is (a_coeff, b_coeff, c_coeff) where:
#   contribution to C[i,k] += a_coeff * b_coeff * c_coeff
# and a, b, c are linear combinations of A and B entries.
#
# For 2x2x2: A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
# C = A*B = [[c00, c01], [c10, c11]]
#
# Strassen:
# M1 = (a00+a11)(b00+b11)
# M2 = (a10+a11)*b00
# M3 = a00*(b01-b11)
# M4 = a11*(b10-b00)
# M5 = (a00+a01)*b11
# M6 = (a10-a00)*(b00+b01)
# M7 = (a01-a11)*(b10+b11)
#
# C[0,0] = M1 - M4 + M5 - M7
# C[0,1] = M3 + M4
# C[1,0] = M2 + M7
# C[1,1] = M1 - M3 + M6 + M7

# Represent each of the 7 terms as:
#   (list of (row_idx, col_idx, coeff) for A) x (list of (row_idx, col_idx, coeff) for B)
# and the contribution to each C element.

def _strassen_2x2x2_terms():
    """
    Return a list of 7 tuples. Each tuple is:
      (a_factors, b_factors, c_map)
    where a_factors is a list of (i, j, coeff) meaning coeff * A[i,j],
          b_factors is a list of (j, k, coeff) meaning coeff * B[j,k],
          c_map is a list of (i, k, coeff) meaning this term contributes coeff * (a_prod * b_prod) to C[i,k].
    """
    terms = []
    # M1 = (a00+a11)(b00+b11)
    terms.append((
        [(0,0,1),(1,1,1)],
        [(0,0,1),(1,1,1)],
        [(0,0,1),(1,1,-1)]  # C[0,0]+=1*M1, C[1,1]-=1*M1 ... wait
    ))
    # Let me be more careful.
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M4
    # C[1,0] = M2 + M7
    # C[1,1] = M1 - M3 + M6 + M7
    
    # M1: contributes +1 to C[0,0], -1 to C[1,1]
    terms = []
    # M1
    terms.append((
        [(0,0,1),(1,1,1)],
        [(0,0,1),(1,1,1)],
        [(0,0,1),(1,1,-1)]
    ))
    # M2 = (a10+a11)*b00
    # C[1,0] += M2
    terms.append((
        [(1,0,1),(1,1,1)],
        [(0,0,1)],
        [(1,0,1)]
    ))
    # M3 = a00*(b01-b11)
    # C[0,1] += M3, C[1,1] -= M3
    terms.append((
        [(0,0,1)],
        [(0,1,1),(1,1,-1)],
        [(0,1,1),(1,1,-1)]
    ))
    # M4 = a11*(b10-b00)
    # C[0,0] -= M4, C[0,1] += M4
    terms.append((
        [(1,1,1)],
        [(1,0,1),(0,0,-1)],
        [(0,0,-1),(0,1,1)]
    ))
    # M5 = (a00+a01)*b11
    # C[0,0] += M5
    terms.append((
        [(0,0,1),(0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ))
    # M6 = (a10-a00)*(b00+b01)
    # C[1,1] += M6
    terms.append((
        [(1,0,1),(0,0,-1)],
        [(0,0,1),(0,1,1)],
        [(1,1,1)]
    ))
    # M7 = (a01-a11)*(b10+b11)
    # C[0,0] -= M7, C[1,0] += M7, C[1,1] += M7
    terms.append((
        [(0,1,1),(1,1,-1)],
        [(1,0,1),(1,1,1)],
        [(0,0,-1),(1,0,1),(1,1,1)]
    ))
    return terms


def _expand_2x2x4_to_4x4x4():
    """
    Build a rank-49 decomposition for 4x4 x 4x4.
    
    Split A (4x4) into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    Split B (4x4) into 2x2 blocks: B = [[B11, B12], [B21, B22]]
    
    Using Strassen at the block level:
    S1 = (A11+A22)(B11+B22)
    S2 = (A21+A22)B11
    S3 = A11(B12-B22)
    S4 = A22(B21-B11)
    S5 = (A11+A12)B22
    S6 = (A21-A11)(B11+B12)
    S7 = (A12-A22)(B21+B22)
    
    C11 = S1 - S4 + S5 - S7
    C12 = S3 + S4
    C21 = S2 + S7
    C22 = S1 - S3 + S6 + S7
    
    Each Sk is a product of two 2x2 matrices, which we decompose with rank 7.
    Total: 7 * 7 = 49 terms.
    """
    base_terms = _strassen_2x2x2_terms()
    
    # Block-level Strassen: for each of the 7 block-products, 
    # we need the A-block combination and B-block combination.
    # 
    # A block indices: 00=(0,0), 01=(0,1), 10=(1,0), 11=(1,1)
    # Each block is 2x2.
    # 
    # S1: A_part = A11+A22, B_part = B11+B22
    # S2: A_part = A21+A22, B_part = B11
    # S3: A_part = A11, B_part = B12-B22
    # S4: A_part = A22, B_part = B21-B11
    # S5: A_part = A11+A12, B_part = B22
    # S6: A_part = A21-A11, B_part = B11+B12
    # S7: A_part = A12-A22, B_part = B21+B22
    
    # For each Sk, A_part is a sum/diff of 1-2 blocks, B_part similarly.
    # We represent A_part as list of (block_row, block_col, coeff)
    # and B_part as list of (block_row, block_col, coeff)
    
    block_combos = [
        # S1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # S2
        ([(1,0,1),(1,1,1)], [(0,0,1)]),
        # S3
        ([(0,0,1)], [(0,1,1),(1,1,-1)]),
        # S4
        ([(1,1,1)], [(1,0,1),(0,0,-1)]),
        # S5
        ([(0,0,1),(0,1,1)], [(1,1,1)]),
        # S6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        # S7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
    ]
    
    # C block contributions:
    # C11 = S1 - S4 + S5 - S7
    # C12 = S3 + S4
    # C21 = S2 + S7
    # C22 = S1 - S3 + S6 + S7
    c_block_map = [
        [(0,0,1),(1,1,-1)],   # S1
        [(1,0,1)],            # S2
        [(0,1,1),(1,1,-1)],   # S3
        [(0,0,-1),(0,1,1)],   # S4
        [(0,0,1)],            # S5
        [(1,1,1)],            # S6
        [(0,0,-1),(1,0,1),(1,1,1)],  # S7
    ]
    
    # Now expand each of the 49 terms.
    # For term (s, t) where s is the block-level Strassen term index (0-6)
    # and t is the inner Strassen term index (0-6):
    
    all_terms = []
    for s in range(7):
        a_blocks, b_blocks = block_combos[s]
        c_blocks = c_block_map[s]
        
        for t in range(7):
            # Inner term t gives:
            # a_inner_factors: list of (i,j,coeff) within a 2x2 block
            # b_inner_factors: list of (j,k,coeff) within a 2x2 block
            # c_inner_map: list of (i,k,coeff) within a 2x2 block
            a_inner, b_inner, c_inner = base_terms[t]
            
            # Full A factors: for each block in a_blocks, for each inner factor
            full_a = []
            for (br, bc, bc_coeff) in a_blocks:
                for (i, j, coeff) in a_inner:
                    full_a.append((br*2 + i, bc*2 + j, bc_coeff * coeff))
            
            full_b = []
            for (br, bc, bc_coeff) in b_blocks:
                for (j, k, coeff) in b_inner:
                    full_b.append((br*2 + j, bc*2 + k, bc_coeff * coeff))
            
            # Full C map: for each block in c_blocks, for each inner c contribution
            full_c = []
            for (cr, cc, cc_coeff) in c_blocks:
                for (i, k, coeff) in c_inner:
                    full_c.append((cr*2 + i, cc*2 + k, cc_coeff * coeff))
            
            all_terms.append((full_a, full_b, full_c))
    
    return all_terms


def _build_4x4x3_schoolbook():
    """
    Build a rank-48 schoolbook decomposition for 4x4 x 4x3.
    A is 4x4, B is 4x3, C is 4x3.
    C[i,k] = sum_j A[i,j] * B[j,k]
    Rank = 4*4*3 = 48.
    """
    terms = []
    for i in range(4):
        for j in range(4):
            for k in range(3):
                a_factors = [(i, j, 1)]
                b_factors = [(j, k, 1)]
                c_map = [(i, k, 1)]
                terms.append((a_factors, b_factors, c_map))
    return terms


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Get the 49 Strassen terms for 4x4x4
    strassen_terms_4x4x4 = _expand_2x2x4_to_4x4x4()
    
    # Get the 48 schoolbook terms for 4x4x3
    schoolbook_terms_4x4x3 = _build_4x4x3_schoolbook()
    
    total_rank = len(strassen_terms_4x4x4) + len(schoolbook_terms_4x4x3)
    # 49 + 48 = 97
    
    U = [[0] * total_rank for _ in range(n * m)]   # 16 x 97
    V = [[0] * total_rank for _ in range(m * p)]   # 28 x 97
    W = [[0] * total_rank for _ in range(p * n)]   # 28 x 97
    
    col = 0
    # Add Strassen terms for columns 0-3
    for a_factors, b_factors, c_map in strassen_terms_4x4x4:
        for (i, j, coeff) in a_factors:
            U[i * m + j][col] += coeff
        for (j, k, coeff) in b_factors:
            V[j * p + k][col] += coeff
        for (i, k, coeff) in c_map:
            W[k * n + i][col] += coeff
        col += 1
    
    # Add schoolbook terms for columns 4-6
    for a_factors, b_factors, c_map in schoolbook_terms_4x4x3:
        for (i, j, coeff) in a_factors:
            U[i * m + j][col] += coeff
        for (j, k, coeff) in b_factors:
            V[j * p + (k + 4)][col] += coeff  # offset by 4 for the column split
        for (i, k, coeff) in c_map:
            W[(k + 4) * n + i][col] += coeff  # offset by 4
        col += 1
    
    return {"U": U, "V": V, "W": W}
