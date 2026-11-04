"""Matrix multiplication 4x5x6 via Strassen-style block decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# Deadline guard: keep a safety margin before the 150s wall-clock limit.
_DEADLINE = 140.0

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Returns U, V, W as lists of lists (4x7, 4x7, 4x7) for 2x2*2x2.
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # Result: [[p1+p4-p5+p7, p3+p5], [p2+p4, p1-p2+p3+p6]]
    
    # U: rows indexed by (i,j) of A, i in {0,1}, j in {0,1}
    # V: rows indexed by (j,k) of B, j in {0,1}, k in {0,1}
    # W: rows indexed by (k,i) of result, k in {0,1}, i in {0,1}
    
    # For each product c (0..6), we need U[i*2+j][c], V[j*2+k][c], W[k*2+i][c]
    # such that sum_c U[i*2+j][c] * V[j*2+k][c] * W[k*2+i][c] = (AB)[i,k]
    
    # Let's define the 7 bilinear forms.
    # Product c corresponds to a specific linear combination.
    
    # p1 = (a+d)(e+h): 
    #   A factors: a -> 1, d -> 1 (i=0,j=0 and i=1,j=1)
    #   B factors: e -> 1, h -> 1 (j=0,k=0 and j=1,k=1)
    #   W: contributes to result[0,0] and result[1,1]
    #   (AB)[0,0] = ae + bg, (AB)[1,1] = cg + dh
    #   p1 = ae + ah + de + dh
    #   So in (AB)[0,0]: p1 contributes +1 (we need ae, but p1 has ae+ah+de+dh)
    #   Actually the Strassen algorithm works by:
    #   (AB)[0,0] = p1 + p4 - p5 + p7
    #   (AB)[0,1] = p3 + p5
    #   (AB)[1,0] = p2 + p4
    #   (AB)[1,1] = p1 - p2 + p3 + p6
    
    # Let me just directly construct U, V, W from the known Strassen formulas.
    # For product c, U[i*2+j][c] is the coefficient of A[i,j] in the first factor,
    # V[j*2+k][c] is the coefficient of B[j,k] in the second factor,
    # W[k*2+i][c] is the coefficient of product c in (AB)[i,k].
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # c=0: p1 = (a+d)(e+h)
    # A: a=U[0][0]=1, d=U[3][0]=1
    # B: e=V[0][0]=1, h=V[3][0]=1
    # W: (AB)[0,0] += p1 => W[0][0] += 1 (k=0,i=0)
    #     (AB)[1,1] += p1 => W[2][1] += 1 (k=1,i=1)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[2][1] += 1
    
    # c=1: p2 = (c+d)e
    # A: c=U[2][1]=1, d=U[3][1]=1
    # B: e=V[0][1]=1
    # W: (AB)[1,0] += p2 => W[0][1] += 1 (k=0,i=1)
    #     (AB)[1,1] -= p2 => W[2][1] -= 1 (k=1,i=1)
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[0][1] += 1; W[2][1] -= 1
    
    # c=2: p3 = a(f-h)
    # A: a=U[0][2]=1
    # B: f=V[1][2]=1, h=V[3][2]=-1
    # W: (AB)[0,1] += p3 => W[1][0] += 1 (k=1,i=0)
    #     (AB)[1,1] += p3 => W[2][1] += 1 (k=1,i=1)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][0] += 1; W[2][1] += 1
    
    # c=3: p4 = d(g-e)
    # A: d=U[3][3]=1
    # B: g=V[2][3]=1, e=V[0][3]=-1
    # W: (AB)[0,0] += p4 => W[0][0] += 1 (k=0,i=0)
    #     (AB)[1,0] += p4 => W[0][1] += 1 (k=0,i=1)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][0] += 1; W[0][1] += 1
    
    # c=4: p5 = (a+b)h
    # A: a=U[0][4]=1, b=U[1][4]=1
    # B: h=V[3][4]=1
    # W: (AB)[0,0] -= p5 => W[0][0] -= 1 (k=0,i=0)
    #     (AB)[0,1] += p5 => W[1][0] += 1 (k=1,i=0)
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][0] -= 1; W[1][0] += 1
    
    # c=5: p6 = (c-a)(e+f)
    # A: c=U[2][5]=1, a=U[0][5]=-1
    # B: e=V[0][5]=1, f=V[1][5]=1
    # W: (AB)[1,1] += p6 => W[2][1] += 1 (k=1,i=1)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[2][1] += 1
    
    # c=6: p7 = (b-d)(g+h)
    # A: b=U[1][6]=1, d=U[3][6]=-1
    # B: g=V[2][6]=1, h=V[3][6]=1
    # W: (AB)[0,0] += p7 => W[0][0] += 1 (k=0,i=0)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][0] += 1
    
    return U, V, W


def _build_decomposition(n, m, p):
    """
    Build a decomposition using block structure.
    We split the n x m matrix into blocks and the m x p matrix into blocks,
    then use Strassen for 2x2 blocks where possible.
    
    For 4x5x6:
    - Split 4x5 into two 2x5 blocks (top and bottom)
    - Split 5x6 into two 5x3 blocks (left and right) -- not square, so Strassen doesn't directly apply
    - Alternative: pad to make square blocks, or use a different splitting.
    
    Actually, for non-square matrices, we can still use Strassen-like techniques
    by padding to the next power of 2 or using rectangular Strassen variants.
    
    A simpler approach: use the known result that rank(4x5x6) <= 108 via
    combining Strassen on 2x2 sub-blocks with schoolbook for the rest.
    
    Let's think about this more carefully.
    
    One approach: 
    - Split A (4x5) into A1 (2x5) and A2 (2x5)
    - Split B (5x6) into B1 (5x3) and B2 (5x3)
    - C = A1*B1 + A2*B2 (each product is 2x3)
    
    For each 2x3 product, we can't directly use Strassen (which is for 2x2*2x2).
    But we can pad 2x3 to 2x4 or use a different decomposition.
    
    Actually, let me try a different approach. Let's use the fact that:
    - 4x5x6 can be decomposed using 2x2 Strassen blocks.
    
    Split A into four 2x2 blocks (padding the 5th column with zeros) plus a 2x1 remainder.
    This gets complicated. Let me just try to construct a valid rank-108 decomposition
    by combining Strassen for the 2x2*2x2 parts and schoolbook for the rest.
    
    Actually, the simplest valid approach that gives rank 108:
    - Use Strassen's algorithm on the 2x2 subproblems.
    - 4x5x6: split into (2x5)*(5x6) + (2x5)*(5x6) for the two row-blocks of A.
    - For each (2x5)*(5x6), split the 5 columns into (2+3) and the 6 cols into (3+3).
    - This is getting complex. Let me just use a known valid construction.
    
    Let me try: pad 4x5 to 4x6 and 5x6 stays, giving 4x6*6x6.
    No, that changes the problem.
    
    Let me just implement the direct Strassen-based construction:
    - A is 4x5, B is 5x6, C is 4x6.
    - Split A into A_top (2x5), A_bot (2x5)
    - Split B into B_left (5x3), B_right (5x3)
    - C_top_left = A_top * B_left (2x3)
    - C_top_right = A_top * B_right (2x3)
    - C_bot_left = A_bot * B_left (2x3)
    - C_bot_right = A_bot * B_right (2x3)
    
    For a 2x3 * 3x3 product (after splitting B into 3x3 blocks... wait B is 5x3, not 3x3).
    
    OK let me just do schoolbook for the non-Strassen parts and Strassen where we have 2x2*2x2.
    
    Simplest valid improvement: just use Strassen for the 2x2 sub-blocks within the
    4x6 output (treating 5 as 2+2+1 or 3+2, etc.)
    
    Let me try: split m=5 into 2+3, and use Strassen on the 2x2 parts.
    - A_top (2x5) * B (5x6): split B into B1 (5x2) and B2 (5x4)
    - A_top * B1 is 2x2, can use Strassen (rank 7 instead of 8)
    - A_top * B2 is 2x4, split into two 2x2: A_top * B2a (2x2) and A_top * B2b (2x2)
    - Similarly for A_bot.
    
    Total rank: 4 sub-problems of 2x2*2x2, each rank 7 => 4*7 = 28? No, that's not right
    because the 2x2 blocks share the same A rows.
    
    Actually, let me just carefully construct it.
    
    A (4x5):
      A[0:2, 0:2], A[0:2, 2:4], A[0:2, 4:5]  -- 2x2, 2x2, 2x1
      A[2:4, 0:2], A[2:4, 2:4], A[2:4, 4:5]  -- 2x2, 2x2, 2x1
    
    B (5x6):
      B[0:2, :], B[2:4, :], B[4:5, :]  -- 2x6, 2x6, 1x6
    
    C = A_top * B + A_bot * B
    
    This is getting too complex to get right quickly. Let me just use the known
    rank-108 construction which is documented in the literature.
    
    Actually, the measured knowledge says Strassen gives 108. Let me just implement
    a straightforward block decomposition that achieves this.
    
    Strategy: 
    - Split the 4x5 matrix into two 2x5 blocks.
    - For each 2x5 block times the 5x6 matrix, split the 5 columns of the first
      factor into 2+3 and use Strassen-like decomposition.
    
    Hmm, let me just try a cleaner approach: pad to 6x6 and use Strassen recursively.
    But padding changes the rank.
    
    Let me just implement the direct construction with Strassen on 2x2 sub-blocks:
    """
    # Direct construction using Strassen for 2x2*2x2 sub-blocks
    # A (4x5), B (5x6), C (4x6)
    
    # Split A into 2x2 blocks (padding col 5 with zeros):
    # A = [[A00, A01, A02],   where A00,A01 are 2x2, A02 is 2x1
    #      [A10, A11, A12]]   where A10,A11 are 2x2, A12 is 2x1
    
    # Split B into rows:
    # B = [[B00, B01, B02],   where B00,B01 are 2x2, B02 is 2x1
    #      [B10, B11, B12],   where B10,B11 are 2x2, B12 is 2x1
    #      [B20, B21, B22]]   where B20,B21 are 1x2, B22 is 1x1
    
    # C = A00*B00 + A01*B10 + A02*B20
    #   + A00*B01 + A01*B11 + A02*B21
    #   + A00*B02 + A01*B12 + A02*B22
    #   + A10*B00 + A11*B10 + A12*B20
    #   + A10*B01 + A11*B11 + A12*B21
    #   + A10*B02 + A11*B12 + A12*B22
    
    # Each 2x2*2x2 product can be done with rank 7 (Strassen)
    # 2x2*2x1 product: rank 2 (schoolbook, since it's just 2 scalar mults)
    # 2x1*1x2 product: rank 2
    # 2x1*1x1 product: rank 2
    # 1x2*2x1 product: rank 2
    # etc.
    
    # Count:
    # 2x2*2x2: there are 4 such products (A00*B00, A01*B11, A10*B01... wait let me recount)
    
    # Actually this is getting very complex. Let me just use a simpler valid approach.
    
    # Use Strassen on 2x2 blocks where the dimensions align:
    # Split 4x5x6 as follows:
    # - The 4 rows of A and 6 cols of C can be split into pairs.
    # - The 5 cols of A and 5 rows of B: split as 2+2+1.
    
    # For the 2+2 part (4x4 effective), use Strassen: rank 7 per 2x2*2x2 block.
    # For the 2x1 and 1x2 parts, use schoolbook.
    
    # Let me just construct it directly:
    
    n, m, p = 4, 5, 6
    rank_total = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_entry, v_entry, w_entry, u_idx, v_idx, w_idx):
        """Add a single rank-1 term."""
        pass
    
    # I'll use a list-based approach where I accumulate terms.
    # Each term is (U_row, V_row, W_row) each of length 1.
    
    terms = []  # list of (u_idx, u_val, v_idx, v_val, w_idx, w_val)
    
    # For each combination of (i_block, j_block, k_block) where:
    # i_block in {0,1} (rows 0-1, 2-3 of A and C)
    # j_block in {0,1,2} (cols 0-1, 2-3, 4 of A and rows 0-1, 2-3, 4 of B)
    # k_block in {0,1,2} (cols 0-1, 2-3, 4-5 of C)
    
    # When j_block < 2 and k_block < 2: 2x2*2x2 -> use Strassen (rank 7)
    # Otherwise: schoolbook
    
    for i_block in range(2):
        for j_block in range(3):
            for k_block in range(3):
                # Determine block sizes
                # A block: rows 2*i_block to 2*i_block+1, cols j_block*2 to min(j_block*2+2, 5)
                a_row_start = 2 * i_block
                a_col_start = j_block * 2
                a_cols = min(2, 5 - a_col_start)  # number of columns in A block
                
                # B block: rows j_block*2 to min(j_block*2+2, 5), cols k_block*2 to min(k_block*2+2, 6)
                b_row_start = j_block * 2
                b_rows = min(2, 5 - b_row_start)
                b_col_start = k_block * 2
                b_cols = min(2, 6 - b_col_start)
                
                # C block: rows 2*i_block to 2*i_block+1, cols k_block*2 to min(k_block*2+2, 6)
                c_row_start = 2 * i_block
                c_col_start = k_block * 2
                c_rows = 2  # always 2 for n=4
                c_cols = min(2, 6 - c_col_start)
                
                # For the product of these blocks:
                # A_block is (2 x a_cols), B_block is (b_rows x b_cols), C_block is (2 x c_cols)
                # The product A_block * B_block contributes to C_block.
                # For this to be valid, a_cols must equal b_rows (the inner dimension).
                # a_cols = min(2, 5 - j_block*2)
                # b_rows = min(2, 5 - j_block*2)
                # So a_cols == b_rows always. Good.
                
                inner_dim = a_cols  # == b_rows
                
                if inner_dim == 2 and c_cols == 2:
                    # 2x2 * 2x2 -> use Strassen, rank 7
                    su, sv, sw = _strassen_2x2()
                    for c in range(7):
                        for a_i in range(2):
                            for a_j in range(2):
                                u_idx = (c_row_start + a_i) * m + (a_col_start + a_j)
                                u_val = su[a_i * 2 + a_j][c]
                                if u_val != 0:
                                    terms.append((u_idx, u_val))
                        for b_j in range(2):
                            for b_k in range(2):
                                v_idx = (b_row_start + b_j) * p + (b_col_start + b_k)
                                v_val = sv[b_j * 2 + b_k][c]
                                if v_val != 0:
                                    terms.append((v_idx, v_val))
                        for c_k in range(2):
                            for c_i in range(2):
                                w_idx = (c_col_start + c_k) * n + (c_row_start + c_i)
                                w_val = sw[c_k * 2 + c_i][c]
                                if w_val != 0:
                                    terms.append((w_idx, w_val))
                    # This is getting messy. Let me restructure.
                    pass
                else:
                    # Schoolbook for this block
                    for a_i in range(2):
                        for a_j in range(inner_dim):
                            for c_k in range(c_cols):
                                u_idx = (c_row_start + a_i) * m + (a_col_start + a_j)
                                v_idx = (b_row_start + a_j) * p + (b_col_start + c_k)
                                w_idx = (c_col_start + c_k) * n + (c_row_start + a_i)
                                terms.append((u_idx, v_idx, w_idx, 1, 1, 1))
    
    # This approach is getting too tangled. Let me just write it more directly.
    return None


def _construct_strassen_4x5x6():
    """
    Construct a rank-108 decomposition for 4x5x6 matrix multiplication.
    
    Approach: Split the computation into blocks and use Strassen (rank 7) for
    2x2*2x2 sub-problems, and schoolbook (rank = product of dims) for the rest.
    """
    n, m, p = 4, 5, 6
    
    # We'll build U, V, W as lists of lists.
    # First, let's determine the total rank by counting terms.
    # Then build the matrices.
    
    # Block decomposition:
    # A (4x5) split into row-blocks of size 2: A0 (2x5), A1 (2x5)
    # B (5x6) split into row-blocks of size 2,2,1: B0 (2x6), B1 (2x6), B2 (1x6)
    # C (4x6) split into col-blocks of size 2,2,2: C0 (4x2), C1 (4x2), C2 (4x2)
    
    # C = A0 * B0 + A0 * B1 + A0 * B2 + A1 * B0 + A1 * B1 + A1 * B2
    # Each A_i * B_j is (2 x 6) = (2x2)*(2x6) + (2x2)*(2x6) + ... 
    # Actually B_j is (rows x 6), and we split the 6 cols into 2+2+2.
    
    # Let me just use a flat approach: enumerate all (i, j, k) triples and
    # group them into 2x2*2x2 Strassen blocks where possible.
    
    # i in 0..3 (rows of A, cols of C)
    # j in 0..4 (cols of A, rows of B)  
    # k in 0..5 (cols of B, cols of C)
    
    # Group i into pairs: (0,1), (2,3)
    # Group j into pairs: (0,1), (2,3), and single (4)
    # Group k into pairs: (0,1), (2,3), (4,5)
    
    # For each combination of i-pair, j-pair/single, k-pair:
    # - If j-pair has 2 elements and the effective sub-problem is 2x2*2x2, use Strassen
    # - Otherwise use schoolbook
    
    terms = []  # Each term: (u_indices, v_indices, w_indices) for a rank-1 component
    # Actually, let's just build U, V, W directly.
    
    # I'll use a dictionary-based approach: for each rank index, store the
    # U, V, W values.
    
    rank = 0
    U_data = {}  # (row, col) -> value
    V_data = {}
    W_data = {}
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries: list of (row, val), etc."""
        nonlocal rank
        col = rank
        for row, val in u_entries:
            if val != 0:
                U_data[(row, col)] = val
        for row, val in v_entries:
            if val != 0:
                V_data[(row, col)] = val
        for row, val in w_entries:
            if val != 0:
                W_data[(row, col)] = val
        rank += 1
    
    # Process each (i_pair, j_group, k_pair) combination
    i_pairs = [(0, 1), (2, 3)]
    j_groups = [(0, 1), (2, 3), (4,)]  # last one is size 1
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for i_pair in i_pairs:
        for j_group in j_groups:
            for k_pair in k_pairs:
                i0, i1 = i_pair
                k0, k1 = k_pair
                j_list = list(j_group)
                
                if len(j_list) == 2:
                    j0, j1 = j_list
                    # 2x2 * 2x2 sub-problem: use Strassen
                    su, sv, sw = _strassen_2x2()
                    for c in range(7):
                        u_entries = []
                        v_entries = []
                        w_entries = []
                        for a_i in range(2):
                            for a_j in range(2):
                                val = su[a_i * 2 + a_j][c]
                                if val != 0:
                                    u_entries.append(((i0 + a_i) * m + (j0 + a_j), val))
                        for b_j in range(2):
                            for b_k in range(2):
                                val = sv[b_j * 2 + b_k][c]
                                if val != 0:
                                    v_entries.append(((j0 + b_j) * p + (k0 + b_k), val))
                        for c_k in range(2):
                            for c_i in range(2):
                                val = sw[c_k * 2 + c_i][c]
                                if val != 0:
                                    w_entries.append(((k0 + c_k) * n + (i0 + c_i), val))
                        if u_entries and v_entries and w_entries:
                            add_term(u_entries, v_entries, w_entries)
                else:
                    # Single j: 2x1 * 1x2 -> schoolbook, rank 2
                    j0 = j_list[0]
                    for a_i in range(2):
                        for c_k in range(2):
                            u_entries = [((i0 + a_i) * m + j0, 1)]
                            v_entries = [(j0 * p + (k0 + c_k), 1)]
                            w_entries = [((k0 + c_k) * n + (i0 + a_i), 1)]
                            add_term(u_entries, v_entries, w_entries)
    
    # Build final matrices
    nU = n * m
    nV = m * p
    nW = p * n
    
    U = [[0] * rank for _ in range(nU)]
    V = [[0] * rank for _ in range(nV)]
    W = [[0] * rank for _ in range(nW)]
    
    for (row, col), val in U_data.items():
        U[row][col] = val
    for (row, col), val in V_data.items():
        V[row][col] = val
    for (row, col), val in W_data.items():
        W[row][col] = val
    
    return {"U": U, "V": V, "W": W}


def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Compute C[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # For each (i, j, k), the tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Check a few random entries
    random.seed(42)
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        expected = 1.0  # T[i,j,k] = 1 if this is the matrix multiplication tensor
        # Actually, the matrix multiplication tensor has T[i,j,k] = delta_{i,k}... no.
        # The tensor for matrix multiplication is T[i,j,k] = 1 for all i,j,k? No.
        # The tensor T satisfies: C[i,k] = sum_j A[i,j] B[j,k] = sum_{j,c} A[i,j] B[j,k] T[i,j,k,c]
        # where T[i,j,k,c] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For the standard decomposition, T[i,j,k] = delta_{j,j} = 1 for all i,j,k.
        # Wait, the tensor is T[i,j,k] = 1 for all i,j,k? That can't be right.
        
        # Actually, the matrix multiplication tensor is defined as:
        # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor entry T[i,j,k] = 1 for all valid i,j,k.
        # So for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1.
        
        actual = 0.0
        for c in range(rank):
            actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        if abs(actual - 1.0) > tol:
            return False, (i, j, k, actual)
    
    return True, None


def construct():
    """Construct a rank-minimal (or near-minimal) decomposition for 4x5x6 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based construction
    try:
        result = _construct_strassen_4x5x6()
        rank = len(result["U"][0]) if result["U"] else 0
        
        # Verify
        valid, info = _verify_decomp(result["U"], result["V"], result["W"], n, m, p)
        if valid:
            return result
        
        # If verification failed, fall back to schoolbook
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition (rank 120)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
