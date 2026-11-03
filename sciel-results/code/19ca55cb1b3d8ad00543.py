"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _add_2x2_strassen(U, V, W, 
                      a_rows, a_cols,   # 2x2 block of A: rows a_rows, cols a_cols
                      b_rows, b_cols,   # 2x2 block of B: rows b_rows, cols b_cols
                      c_rows, c_cols,   # 2x2 block of C: rows c_rows, cols c_cols
                      r_offset,
                      sign_a=1, sign_b=1, sign_c=1):
    """
    Add a 2x2x2 Strassen decomposition (7 rank-1 terms) for the sub-problem:
    C_block = A_block * B_block
    where each block is 2x2.
    
    U index: i*m + j  (A[i,j])
    V index: j*p + k  (B[j,k])
    W index: k*n + i  (C[i,k])
    
    a_rows = [r0, r1], a_cols = [c0, c1]  -> A indices: (r0,c0),(r0,c1),(r1,c0),(r1,c1)
    b_rows = [r0, r1], b_cols = [c0, c1]  -> B indices
    c_rows = [r0, r1], c_cols = [c0, c1]  -> C indices (but W uses k*n+i)
    """
    n, m, p = 4, 4, 9  # global dims for index computation
    
    # A block element indices in U: U[i*m+j]
    a00 = a_rows[0] * m + a_cols[0]
    a01 = a_rows[0] * m + a_cols[1]
    a10 = a_rows[1] * m + a_cols[0]
    a11 = a_rows[1] * m + a_cols[1]
    
    # B block element indices in V: V[j*p+k]
    b00 = b_rows[0] * p + b_cols[0]
    b01 = b_rows[0] * p + b_cols[1]
    b10 = b_rows[1] * p + b_cols[0]
    b11 = b_rows[1] * p + b_cols[1]
    
    # C block element indices in W: W[k*n+i]
    # c_rows = output row indices (i), c_cols = output col indices (k)
    c00 = c_cols[0] * n + c_rows[0]  # k=0, i=0
    c01 = c_cols[1] * n + c_rows[0]  # k=1, i=0
    c10 = c_cols[0] * n + c_rows[1]  # k=0, i=1
    c11 = c_cols[1] * n + c_rows[1]  # k=1, i=1
    
    # Strassen 2x2: 7 rank-1 terms
    # M1 = (a00 + a11)(b00 + b11) -> c00 += M1, c11 += M1
    # M2 = (a10 + a11)b00 -> c10 += M2
    # M3 = a00(b01 - b11) -> c01 += M3
    # M4 = a11(b10 - b00) -> c10 += M4
    # M5 = (a00 + a01)b11 -> c01 += M5
    # M6 = (a10 - a00)(b00 + b01) -> c11 += M6
    # M7 = (a01 - a11)(b10 + b11) -> c00 += M7
    
    # Term 1: M1
    r = r_offset
    U[a00][r] += sign_a
    U[a11][r] += sign_a
    V[b00][r] += sign_b
    V[b11][r] += sign_b
    W[c00][r] += sign_c
    W[c11][r] += sign_c
    
    # Term 2: M2
    r = r_offset + 1
    U[a10][r] += sign_a
    U[a11][r] += sign_a
    V[b00][r] += sign_b
    W[c10][r] += sign_c
    
    # Term 3: M3
    r = r_offset + 2
    U[a00][r] += sign_a
    V[b01][r] += sign_b
    V[b11][r] += -sign_b
    W[c01][r] += sign_c
    
    # Term 4: M4
    r = r_offset + 3
    U[a11][r] += sign_a
    V[b10][r] += sign_b
    V[b00][r] += -sign_b
    W[c10][r] += sign_c
    
    # Term 5: M5
    r = r_offset + 4
    U[a00][r] += sign_a
    U[a01][r] += sign_a
    V[b11][r] += sign_b
    W[c01][r] += sign_c
    
    # Term 6: M6
    r = r_offset + 5
    U[a10][r] += sign_a
    U[a00][r] += -sign_a
    V[b00][r] += sign_b
    V[b01][r] += sign_b
    W[c11][r] += sign_c
    
    # Term 7: M7
    r = r_offset + 6
    U[a01][r] += sign_a
    U[a11][r] += -sign_a
    V[b10][r] += sign_b
    V[b11][r] += sign_b
    W[c00][r] += sign_c


def _add_4x4_strassen(U, V, W, 
                      a_row_base, a_col_base,   # top-left of 4x4 A block
                      b_row_base, b_col_base,   # top-left of 4x4 B block
                      c_row_base, c_col_base,   # top-left of 4x4 C block
                      r_offset):
    """
    Add a 4x4x4 Strassen decomposition (49 rank-1 terms) using two levels of Strassen.
    
    Level 1: Split 4x4 into 2x2 blocks of 2x2 matrices.
    A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    B = [[B00, B01], [B10, B11]]
    C = A*B = [[C00, C01], [C10, C11]]
    
    7 block-level Strassen products:
    P1 = (A00 + A11)(B00 + B11) -> C00 += P1, C11 += P1
    P2 = (A10 + A11)B00 -> C10 += P2
    P3 = A00(B01 - B11) -> C01 += P3
    P4 = A11(B10 - B00) -> C10 += P4
    P5 = (A00 + A01)B11 -> C01 += P5
    P6 = (A10 - A00)(B00 + B01) -> C11 += P6
    P7 = (A01 - A11)(B10 + B11) -> C00 += P7
    
    Each Pi is a 2x2 * 2x2 = 2x2 product, decomposed with 7 scalar rank-1 terms.
    Total: 7 * 7 = 49 rank-1 terms.
    """
    n, m, p = 4, 4, 9
    
    # Block indices (each block is 2x2)
    # A blocks:
    A00_r, A00_c = a_row_base, a_col_base
    A01_r, A01_c = a_row_base, a_col_base + 2
    A10_r, A10_c = a_row_base + 2, a_col_base
    A11_r, A11_c = a_row_base + 2, a_col_base + 2
    
    # B blocks:
    B00_r, B00_c = b_row_base, b_col_base
    B01_r, B01_c = b_row_base, b_col_base + 2
    B10_r, B10_c = b_row_base + 2, b_col_base
    B11_r, B11_c = b_row_base + 2, b_col_base + 2
    
    # C blocks:
    C00_r, C00_c = c_row_base, c_col_base
    C01_r, C01_c = c_row_base, c_col_base + 2
    C10_r, C10_c = c_row_base + 2, c_col_base
    C11_r, C11_c = c_row_base + 2, c_col_base + 2
    
    r = r_offset
    
    # P1 = (A00 + A11)(B00 + B11) -> C00 += P1, C11 += P1
    # This is a 2x2*2x2 product with A = A00+A11, B = B00+B11, result added to C00 and C11
    # We need to decompose (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # But we can use Strassen on the "combined" matrices.
    # Actually, the cleanest way: P1 is itself a 2x2*2x2 multiplication where
    # the "A matrix" has entries A00[i,j]+A11[i,j] and "B matrix" has B00[i,j]+B11[i,j].
    # So we add 4 separate 2x2 Strassen decompositions? No, that would be 28 terms.
    # 
    # Better: P1 = (A00+A11)(B00+B11) is a single 2x2*2x2 product.
    # We can apply Strassen's 7-term decomposition to this product directly.
    # The "A" for Strassen is the 2x2 matrix with entries:
    #   a00 = A00[0,0] + A11[0,0], a01 = A00[0,1] + A11[0,1]
    #   a10 = A00[1,0] + A11[1,0], a11 = A00[1,1] + A11[1,1]
    # Similarly for B.
    # 
    # But in our tensor decomposition framework, we can't directly have "sum" entries.
    # Instead, we expand: (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Each of these is a 2x2*2x2 product = 7 rank-1 terms, total 28 terms.
    # That's too many. 
    #
    # Alternative: Use the fact that we can have multiple entries in U, V, W for a single rank-1 term.
    # For P1, the 7 Strassen terms for the product (A00+A11)*(B00+B11) can be written as:
    # M1_1 = ((A00+A11)[0,0] + (A00+A11)[1,1]) * ((B00+B11)[0,0] + (B00+B11)[1,1])
    #       = (A00[0,0]+A11[0,0]+A00[1,1]+A11[1,1]) * (B00[0,0]+B11[0,0]+B00[1,1]+B11[1,1])
    # This is a single rank-1 term with 4 entries in U, 4 in V, and contributes to C00 and C11.
    #
    # So for each of the 7 Strassen terms, we add ONE rank-1 term with multiple non-zero entries.
    
    # Helper to add a 2x2 Strassen product where A and B are "sums" of blocks
    def add_combined_2x2(a_blocks, b_blocks, c_blocks, r_cur, sign=1):
        """
        Add Strassen 2x2 for product of (sum of a_blocks) * (sum of b_blocks),
        result added to (sum of c_blocks).
        a_blocks: list of (row_base, col_base) for 2x2 A blocks to sum
        b_blocks: list of (row_base, col_base) for 2x2 B blocks to sum
        c_blocks: list of (row_base, col_base) for 2x2 C blocks to add result to
        """
        # Collect all A element indices with their signs
        a_entries = []  # (U_index, sign)
        for (rb, cb) in a_blocks:
            a_entries.append((rb * m + cb, sign))
            a_entries.append((rb * m + cb + 1, sign))
            a_entries.append(((rb + 1) * m + cb, sign))
            a_entries.append(((rb + 1) * m + cb + 1, sign))
        
        b_entries = []
        for (rb, cb) in b_blocks:
            b_entries.append((rb * p + cb, sign))
            b_entries.append((rb * p + cb + 1, sign))
            b_entries.append(((rb + 1) * p + cb, sign))
            b_entries.append(((rb + 1) * p + cb + 1, sign))
        
        c_entries = []
        for (rb, cb) in c_blocks:
            c_entries.append((cb * n + rb, sign))
            c_entries.append((cb * n + rb + 1, sign))
            c_entries.append(((cb + 1) * n + rb, sign))
            c_entries.append(((cb + 1) * n + rb + 1, sign))
        
        # Now apply Strassen 2x2 to the "combined" 2x2 matrices
        # The 2x2 A matrix has elements at positions (0,0),(0,1),(1,0),(1,1)
        # But we have multiple physical entries for each logical position.
        # For Strassen term 1: (a00+a11)(b00+b11)
        #   U: all a00 and a11 entries get +1
        #   V: all b00 and b11 entries get +1
        #   W: all c00 and c11 entries get +1
        
        # This is getting complex. Let me use a different approach.
        # I'll just expand each block product into schoolbook 4 terms and use 
        # the 7-term Strassen at the scalar level for each 2x2*2x2.
        
        pass  # Not used, see below
    
    # Simpler approach: For each of the 7 block-level Strassen products,
    # expand into 4 scalar 2x2*2x2 products and apply 7-term Strassen to each.
    # That's 7*4*7 = 196 terms, which is way too many.
    
    # Better approach: For each block-level Strassen product P = A_sum * B_sum,
    # where A_sum is a 2x2 matrix (sum of blocks) and B_sum is a 2x2 matrix,
    # apply the 7-term Strassen decomposition directly, where each "element" of A_sum
    # and B_sum is a sum of physical entries.
    
    # For P1 = (A00+A11)(B00+B11):
    # A_sum[0,0] = A00[0,0] + A11[0,0]
    # A_sum[0,1] = A00[0,1] + A11[0,1]
    # A_sum[1,0] = A00[1,0] + A11[1,0]
    # A_sum[1,1] = A00[1,1] + A11[1,1]
    # Similarly for B_sum.
    
    # Strassen term 1: (A_sum[0,0] + A_sum[1,1]) * (B_sum[0,0] + B_sum[1,1])
    # = (A00[0,0]+A11[0,0]+A00[1,1]+A11[1,1]) * (B00[0,0]+B11[0,0]+B00[1,1]+B11[1,1])
    # This is ONE rank-1 term with 4 U entries, 4 V entries, and W entries for C00 and C11.
    
    # Let me define a helper that adds a Strassen 2x2 decomposition where
    # each logical element is a list of (index, sign) pairs.
    
    def add_strassen_2x2_with_sums(a_elems, b_elems, c_elems, r_start):
        """
        a_elems: dict {(i,j): [(U_idx, sign), ...]} for i,j in {0,1}
        b_elems: dict {(i,j): [(V_idx, sign), ...]} for i,j in {0,1}
        c_elems: dict {(i,j): [(W_idx, sign), ...]} for i,j in {0,1}
        Returns number of rank-1 terms added (7).
        """
        # Strassen 2x2:
        # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
        # M2 = (a10+a11)b00 -> c10 += M2
        # M3 = a00(b01-b11) -> c01 += M3
        # M4 = a11(b10-b00) -> c10 += M4
        # M5 = (a00+a01)b11 -> c01 += M5
        # M6 = (a10-a00)(b00+b01) -> c11 += M6
        # M7 = (a01-a11)(b10+b11) -> c00 += M7
        
        terms = 0
        
        # M1
        r = r_start
        for (idx, s) in a_elems[(0,0)]:
            U[idx][r] += s
        for (idx, s) in a_elems[(1,1)]:
            U[idx][r] += s
        for (idx, s) in b_elems[(0,0)]:
            V[idx][r] += s
        for (idx, s) in b_elems[(1,1)]:
            V[idx][r] += s
        for (idx, s) in c_elems[(0,0)]:
            W[idx][r] += s
        for (idx, s) in c_elems[(1,1)]:
            W[idx][r] += s
        terms += 1
        
        # M2
        r = r_start + 1
        for (idx, s) in a_elems[(1,0)]:
            U[idx][r] += s
        for (idx, s) in a_elems[(1,1)]:
            U[idx][r] += s
        for (idx, s) in b_elems[(0,0)]:
            V[idx][r] += s
        for (idx, s) in c_elems[(1,0)]:
            W[idx][r] += s
        terms += 1
        
        # M3
        r = r_start + 2
        for (idx, s) in a_elems[(0,0)]:
            U[idx][r] += s
        for (idx, s) in b_elems[(0,1)]:
            V[idx][r] += s
        for (idx, s) in b_elems[(1,1)]:
            V[idx][r] += -s
        for (idx, s) in c_elems[(0,1)]:
            W[idx][r] += s
        terms += 1
        
        # M4
        r = r_start + 3
        for (idx, s) in a_elems[(1,1)]:
            U[idx][r] += s
        for (idx, s) in b_elems[(1,0)]:
            V[idx][r] += s
        for (idx, s) in b_elems[(0,0)]:
            V[idx][r] += -s
        for (idx, s) in c_elems[(1,0)]:
            W[idx][r] += s
        terms += 1
        
        # M5
        r = r_start + 4
        for (idx, s) in a_elems[(0,0)]:
            U[idx][r] += s
        for (idx, s) in a_elems[(0,1)]:
            U[idx][r] += s
        for (idx, s) in b_elems[(1,1)]:
            V[idx][r] += s
        for (idx, s) in c_elems[(0,1)]:
            W[idx][r] += s
        terms += 1
        
        # M6
        r = r_start + 5
        for (idx, s) in a_elems[(1,0)]:
            U[idx][r] += s
        for (idx, s) in a_elems[(0,0)]:
            U[idx][r] += -s
        for (idx, s) in b_elems[(0,0)]:
            V[idx][r] += s
        for (idx, s) in b_elems[(0,1)]:
            V[idx][r] += s
        for (idx, s) in c_elems[(1,1)]:
            W[idx][r] += s
        terms += 1
        
        # M7
        r = r_start + 6
        for (idx, s) in a_elems[(0,1)]:
            U[idx][r] += s
        for (idx, s) in a_elems[(1,1)]:
            U[idx][r] += -s
        for (idx, s) in b_elems[(1,0)]:
            V[idx][r] += s
        for (idx, s) in b_elems[(1,1)]:
            V[idx][r] += s
        for (idx, s) in c_elems[(0,0)]:
            W[idx][r] += s
        terms += 1
        
        return terms
    
    def make_block_elems(rb, cb, mat_type, sign=1):
        """Make element dict for a single 2x2 block."""
        elems = {}
        for di in range(2):
            for dj in range(2):
                if mat_type == 'A':
                    idx = (rb + di) * m + (cb + dj)
                elif mat_type == 'B':
                    idx = (rb + di) * p + (cb + dj)
                else:  # C
                    idx = (cb + dj) * n + (rb + di)
                elems[(di, dj)] = [(idx, sign)]
        return elems
    
    def combine_block_elems(*block_elems_list):
        """Combine multiple block element dicts by summing their entries."""
        result = {}
        for be in block_elems_list:
            for key in be:
                if key not in result:
                    result[key] = []
                result[key].extend(be[key])
        return result
    
    # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
    a_p1 = combine_block_elems(make_block_elems(A00_r, A00_c, 'A'), make_block_elems(A11_r, A11_c, 'A'))
    b_p1 = combine_block_elems(make_block_elems(B00_r, B00_c, 'B'), make_block_elems(B11_r, B11_c, 'B'))
    c_p1 = combine_block_elems(make_block_elems(C00_r, C00_c, 'C'), make_block_elems(C11_r, C11_c, 'C'))
    r += add_strassen_2x2_with_sums(a_p1, b_p1, c_p1, r)
    
    # P2 = (A10+A11)B00 -> C10 += P2
    a_p2 = combine_block_elems(make_block_elems(A10_r, A10_c, 'A'), make_block_elems(A11_r, A11_c, 'A'))
    b_p2 = make_block_elems(B00_r, B00_c, 'B')
    c_p2 = make_block_elems(C10_r, C10_c, 'C')
    r += add_strassen_2x2_with_sums(a_p2, b_p2, c_p2, r)
    
    # P3 = A00(B01-B11) -> C01 += P3
    a_p3 = make_block_elems(A00_r, A00_c, 'A')
    b_p3 = combine_block_elems(make_block_elems(B01_r, B01_c, 'B', sign=1), make_block_elems(B11_r, B11_c, 'B', sign=-1))
    c_p3 = make_block_elems(C01_r, C01_c, 'C')
    r += add_strassen_2x2_with_sums(a_p3, b_p3, c_p3, r)
    
    # P4 = A11(B10-B00) -> C10 += P4
    a_p4 = make_block_elems(A11_r, A11_c, 'A')
    b_p4 = combine_block_elems(make_block_elems(B10_r, B10_c, 'B', sign=1), make_block_elems(B00_r, B00_c, 'B', sign=-1))
    c_p4 = make_block_elems(C10_r, C10_c, 'C')
    r += add_strassen_2x2_with_sums(a_p4, b_p4, c_p4, r)
    
    # P5 = (A00+A01)B11 -> C01 += P5
    a_p5 = combine_block_elems(make_block_elems(A00_r, A00_c, 'A'), make_block_elems(A01_r, A01_c, 'A'))
    b_p5 = make_block_elems(B11_r, B11_c, 'B')
    c_p5 = make_block_elems(C01_r, C01_c, 'C')
    r += add_strassen_2x2_with_sums(a_p5, b_p5, c_p5, r)
    
    # P6 = (A10-A00)(B00+B01) -> C11 += P6
    a_p6 = combine_block_elems(make_block_elems(A10_r, A10_c, 'A', sign=1), make_block_elems(A00_r, A00_c, 'A', sign=-1))
    b_p6 = combine_block_elems(make_block_elems(B00_r, B00_c, 'B'), make_block_elems(B01_r, B01_c, 'B'))
    c_p6 = make_block_elems(C11_r, C11_c, 'C')
    r += add_strassen_2x2_with_sums(a_p6, b_p6, c_p6, r)
    
    # P7 = (A01-A11)(B10+B11) -> C00 += P7
    a_p7 = combine_block_elems(make_block_elems(A01_r, A01_c, 'A', sign=1), make_block_elems(A11_r, A11_c, 'A', sign=-1))
    b_p7 = combine_block_elems(make_block_elems(B10_r, B10_c, 'B'), make_block_elems(B11_r, B11_c, 'B'))
    c_p7 = make_block_elems(C00_r, C00_c, 'C')
    r += add_strassen_2x2_with_sums(a_p7, b_p7, c_p7, r)
    
    return r - r_offset  # number of terms added (should be 49)


def _schoolbook():
    """Return the standard rank-144 schoolbook decomposition for 4x4 * 4x9."""
    n, m, p = DIMENSIONS  # 4, 4, 9
    rank = n * m * p  # 144

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}


def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy: Split 9 output columns into 4+4+1.
    - First 4 columns: 4x4x4 Strassen (rank 49)
    - Next 4 columns: 4x4x4 Strassen (rank 49)
    - Last 1 column: 4x4x1 schoolbook (rank 16)
    Total rank: 49 + 49 + 16 = 114
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    deadline = time.time() + 140.0  # safety margin
    
    # Build U, V, W as lists of lists
    # U: (n*m) x rank = 16 x 114
    # V: (m*p) x rank = 36 x 114
    # W: (p*n) x rank = 36 x 114
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First 4x4x4 Strassen block: columns 0-3 of B, columns 0-3 of C
    r = _add_4x4_strassen(U, V, W,
                          a_row_base=0, a_col_base=0,
                          b_row_base=0, b_col_base=0,
                          c_row_base=0, c_col_base=0,
                          r_offset=0)
    
    # Second 4x4x4 Strassen block: columns 4-7 of B, columns 4-7 of C
    r2 = _add_4x4_strassen(U, V, W,
                           a_row_base=0, a_col_base=0,
                           b_row_base=0, b_col_base=4,
                           c_row_base=0, c_col_base=4,
                           r_offset=r)
    
    # Third block: 4x4x1 schoolbook for column 8
    # A[i,j] * B[j,8] -> C[i,8]
    for i in range(n):
        for j in range(m):
            U[i * m + j][r2 + i * m + j] = 1
            V[j * p + 8][r2 + i * m + j] = 1
            W[8 * n + i][r2 + i * m + j] = 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    valid = True
    for i in range(n):
        if not valid:
            break
        for j in range(m):
            if not valid:
                break
            for k in range(p):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
    
    if not valid:
        # Fallback to schoolbook
        return _schoolbook()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
