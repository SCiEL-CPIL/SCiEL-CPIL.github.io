"""Matrix multiplication tensor decomposition: 3x4x5."""
from fractions import Fraction

def construct():
    n, m, p = 3, 4, 5
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Index conventions:
    #   U[i*m + j, c] corresponds to coefficient of A[i,j] in the c-th factor
    #   V[j*p + k, c] corresponds to coefficient of B[j,k] in the c-th factor
    #   W[k*n + i, c] corresponds to coefficient of C[i,k] output from the c-th factor

    # We use a hierarchical Strassen-based approach:
    # Split n=3 as 2+1, m=4 as 2+2, p=5 as 2+2+1
    # 
    # Strategy: For each pair of row-block (i1, i2) and col-block (j1, j2) and k-block (k1, k2),
    # we can use Strassen's 7-product algorithm for 2x2x2 blocks.
    # 
    # Let's think of this more carefully.
    #
    # A is 3x4, B is 4x5, C = A*B is 3x5.
    # 
    # Partition A into rows: A_top (2x4) and A_bottom (1x4)
    # Partition B into cols: B_left (4x2), B_mid (4x2), B_right (4x1)
    # 
    # Actually, let's use a different partitioning that's more amenable to Strassen:
    # 
    # Think of A as 3x4 = stack of 2x4 and 1x4
    # Think of B as 4x5
    # 
    # For the top 2 rows of C: C_top = A_top * B, which is 2x5.
    # For the bottom 1 row of C: C_bottom = A_bottom * B, which is 1x5.
    #
    # C_top = A_top * B where A_top is 2x4, B is 4x5 -> output 2x5
    # C_bottom = A_bottom * B where A_bottom is 1x4, B is 4x5 -> output 1x5
    #
    # For C_top (2x5 = 2 rows, 5 cols):
    #   We can partition the 4 intermediate columns as 2+2.
    #   A_top = [A1 | A2] where A1 is 2x2, A2 is 2x2
    #   B = [[B1],[B2]] where B1 is 2x5, B2 is 2x5
    #   C_top = A1*B1 + A2*B2
    #
    # For C_bottom (1x5):
    #   A_bottom is 1x4 = [b1 | b2] where b1 is 1x2, b2 is 1x2
    #   B = [[B1],[B2]] where B1 is 2x5, B2 is 2x5
    #   C_bottom = b1*B1 + b2*B2
    #
    # Now, A1*B1 is 2x2 * 2x5 = 2x5
    # A2*B2 is 2x2 * 2x5 = 2x5
    # b1*B1 is 1x2 * 2x5 = 1x5
    # b2*B2 is 1x2 * 2x5 = 1x5
    #
    # For 2x2 * 2x5: partition the 5 columns as 2+2+1
    #   Let's call the column blocks c1 (2 cols), c2 (2 cols), c3 (1 col)
    #   A1*B1 = A1 * [B1_c1 | B1_c2 | B1_c3]
    #   The first part: A1 * B1_c1 is 2x2 * 2x2 -> use Strassen (rank 7)
    #   The second part: A1 * B1_c2 is 2x2 * 2x2 -> use Strassen (rank 7)
    #   The third part: A1 * B1_c3 is 2x2 * 2x1 -> schoolbook (rank 4)
    #   Total for A1*B1: 7+7+4 = 18
    #
    # Similarly for A2*B2: 18
    # b1*B1: 1x2 * 2x5 = 1x5, partition cols as 2+2+1
    #   b1 * B1_c1: 1x2 * 2x2 = 4 terms (schoolbook, since 1x2 can't use Strassen)
    #   b1 * B1_c2: 4 terms
    #   b1 * B1_c3: 2 terms
    #   Total: 10
    #
    # Similarly b2*B2: 10
    #
    # Grand total: 18 + 18 + 10 + 10 = 56
    #
    # That gives rank 56, which beats 60. Let me verify this is valid.
    #
    # Actually wait - let me reconsider. The tensor M_{3,4,5} has been studied.
    # Known bounds: the rank of M_{3,4,5} is at most 56 via the construction above.
    # The schoolbook gives 60. So 56 is a genuine improvement.
    #
    # Let me carefully implement this.

    def strassen_2x2x2(A_indices, B_indices, C_indices, col_offset):
        """
        Return list of (u_dict, v_dict, w_dict) for Strassen 2x2x2.
        A_indices: dict mapping (i,j) -> column index in U (i in 0..1, j in 0..1)
        B_indices: dict mapping (j,k) -> column index in V
        C_indices: dict mapping (k,i) -> column index in W
        """
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = A*B = [[C00, C01], [C10, C11]]
        # C00 = A00*B00 + A01*B10... wait no.
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # Strassen for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        terms = []
        # M1 = (A00+A11)(B00+B11)
        u = {A_indices[(0,0)]: 1, A_indices[(1,1)]: 1}
        v = {B_indices[(0,0)]: 1, B_indices[(1,1)]: 1}
        w = {C_indices[(0,0)]: 1, C_indices[(1,0)]: -1, C_indices[(1,1)]: 1, C_indices[(1,1)]: 1}
        # C00 += M1, C10 -= M1... wait
        # C00 = M1 + M4 - M5 + M7 -> W contribution for C00 (index (0,0)) is +1
        # C10 = M2 + M4 -> no M1 in C10
        # C11 = M1 - M2 + M3 + M6 -> C11 (index (1,1)) gets +1
        # Wait, C_indices maps (k,i) -> column. 
        # C[0,0] corresponds to k=0,i=0 -> C_indices[(0,0)]
        # C[0,1] corresponds to k=1,i=0 -> C_indices[(1,0)]
        # C[1,0] corresponds to k=0,i=1 -> C_indices[(0,1)]
        # C[1,1] corresponds to k=1,i=1 -> C_indices[(1,1)]
        # 
        # C00 = C[0,0] = M1 + M4 - M5 + M7
        # C01 = C[0,1] = M3 + M5
        # C10 = C[1,0] = M2 + M4
        # C11 = C[1,1] = M1 - M2 + M3 + M6

        w1 = {C_indices[(0,0)]: 1}  # M1 contributes +1 to C00
        w1[C_indices[(1,1)]] = 1  # M1 contributes +1 to C11
        terms.append((u, v, w1))

        # M2 = (A10+A11)*B00
        u2 = {A_indices[(1,0)]: 1, A_indices[(1,1)]: 1}
        v2 = {B_indices[(0,0)]: 1}
        w2 = {C_indices[(0,1)]: 1}  # C10 = M2+M4, C[1,0] -> (k=0,i=1)
        w2[C_indices[(1,1)]] = -1  # C11 = M1-M2+M3+M6, C[1,1] -> (k=1,i=1)
        terms.append((u2, v2, w2))

        # M3 = A00*(B01-B11)
        u3 = {A_indices[(0,0)]: 1}
        v3 = {B_indices[(0,1)]: 1, B_indices[(1,1)]: -1}
        w3 = {C_indices[(1,0)]: 1}  # C01 = M3+M5, C[0,1] -> (k=1,i=0)
        w3[C_indices[(1,1)]] = 1  # C11 gets +M3
        terms.append((u3, v3, w3))

        # M4 = A11*(B10-B00)
        u4 = {A_indices[(1,1)]: 1}
        v4 = {B_indices[(1,0)]: 1, B_indices[(0,0)]: -1}
        w4 = {C_indices[(0,0)]: 1}  # C00 gets +M4
        w4[C_indices[(0,1)]] = 1  # C10 gets +M4, C[1,0] -> (k=0,i=1)
        terms.append((u4, v4, w4))

        # M5 = (A00+A01)*B11
        u5 = {A_indices[(0,0)]: 1, A_indices[(0,1)]: 1}
        v5 = {B_indices[(1,1)]: 1}
        w5 = {C_indices[(0,0)]: -1}  # C00 gets -M5
        w5[C_indices[(1,0)]] = 1  # C01 gets +M5, C[0,1] -> (k=1,i=0)
        terms.append((u5, v5, w5))

        # M6 = (A10-A00)*(B00+B01)
        u6 = {A_indices[(1,0)]: 1, A_indices[(0,0)]: -1}
        v6 = {B_indices[(0,0)]: 1, B_indices[(0,1)]: 1}
        w6 = {C_indices[(1,1)]: 1}  # C11 gets +M6, C[1,1] -> (k=1,i=1)
        terms.append((u6, v6, w6))

        # M7 = (A01-A11)*(B10+B11)
        u7 = {A_indices[(0,1)]: 1, A_indices[(1,1)]: -1}
        v7 = {B_indices[(1,0)]: 1, B_indices[(1,1)]: 1}
        w7 = {C_indices[(0,0)]: 1}  # C00 gets +M7
        terms.append((u7, v7, w7))

        return terms

    def schoolbook_2x2x2(A_indices, B_indices, C_indices):
        """Schoolbook rank-4 decomposition for 2x2x2."""
        terms = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = {A_indices[(i,j)]: 1}
                    v = {B_indices[(j,k)]: 1}
                    w = {C_indices[(k,i)]: 1}
                    terms.append((u, v, w))
        return terms

    def schoolbook_1x2x2(A_idx, B_indices, C_indices):
        """Schoolbook for 1x2 * 2x2 = 1x2, rank 4."""
        terms = []
        for j in range(2):
            for k in range(2):
                u = {A_idx[j]: 1}
                v = {B_indices[(j,k)]: 1}
                w = {C_indices[(k,0)]: 1}
                terms.append((u, v, w))
        return terms

    # Build the full decomposition
    # A is 3x4: rows i=0,1,2; cols j=0,1,2,3
    # B is 4x5: rows j=0,1,2,3; cols k=0,1,2,3,4
    # C is 3x5: rows i=0,1,2; cols k=0,1,2,3,4
    
    # U rows: i*m+j = i*4+j for i in 0..2, j in 0..3
    # V rows: j*p+k = j*5+k for j in 0..3, k in 0..4
    # W rows: k*n+i = k*3+i for k in 0..4, i in 0..2

    # Collect all terms as (u_dict, v_dict, w_dict) where keys are row indices
    all_terms = []

    # --- Top block: i in {0,1} (2 rows of C) ---
    # A_top: rows 0,1 of A, all 4 columns
    # Split intermediate index j into {0,1} and {2,3}
    # B: all 4 rows, split columns k into {0,1}, {2,3}, {4}
    
    # A1: A[0:2, 0:2] -> indices (0,0),(0,1),(1,0),(1,1) in A
    # A2: A[0:2, 2:4] -> indices (0,2),(0,3),(1,2),(1,3) in A
    # B1: B[0:2, :] -> rows 0,1 of B
    # B2: B[2:4, :] -> rows 2,3 of B

    # C_top[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    #             = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #             = (A1*B1)[i,k] + (A2*B2)[i,k]

    # For A1*B1: 2x2 * 2x5, split k into {0,1},{2,3},{4}
    # A1*B1[k=0:2] is 2x2*2x2 -> Strassen rank 7
    # A1*B1[k=2:4] is 2x2*2x2 -> Strassen rank 7
    # A1*B1[k=4:5] is 2x2*2x1 -> schoolbook rank 4

    # A1 indices in A: row i, col j
    # U row for A[i,j] = i*4+j
    # A1: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    A1_idx = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
    A2_idx = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    # B1: row j, col k -> V row = j*5+k
    # B1 rows 0,1: 
    # For k-block {0,1}: B1[k=0:2] -> rows 0,1 cols 0,1
    # V row for B[j,k] = j*5+k
    B1_k01 = {(0,0): 0*5+0, (0,1): 0*5+1, (1,0): 1*5+0, (1,1): 1*5+1}
    B1_k23 = {(0,0): 0*5+2, (0,1): 0*5+3, (1,0): 1*5+2, (1,1): 1*5+3}
    B1_k4  = {(0,0): 0*5+4, (0,1): 0*5+4, (1,0): 1*5+4, (1,1): 1*5+4}
    B2_k01 = {(0,0): 2*5+0, (0,1): 2*5+1, (1,0): 3*5+0, (1,1): 3*5+1}
    B2_k23 = {(0,0): 2*5+2, (0,1): 2*5+3, (1,0): 3*5+2, (1,1): 3*5+3}
    B2_k4  = {(0,0): 2*5+4, (0,1): 2*5+4, (1,0): 3*5+4, (1,1): 3*5+4}

    # C indices: C[i,k] -> W row = k*3+i
    # For top block i in {0,1}:
    # C_top[k=0:2, i]: W rows k*3+i for k=0,1; i=0,1
    C_k01_i01 = {(0,0): 0*3+0, (0,1): 0*3+1, (1,0): 1*3+0, (1,1): 1*3+1}
    C_k23_i01 = {(0,0): 2*3+0, (0,1): 2*3+1, (1,0): 3*3+0, (1,1): 3*3+1}
    C_k4_i01  = {(0,0): 4*3+0, (0,1): 4*3+1}

    # A1*B1, k=0:2 (2x2*2x2), C output cols 0,1 for rows 0,1
    all_terms.extend(strassen_2x2x2(A1_idx, B1_k01, C_k01_i01))
    # A1*B1, k=2:3 (2x2*2x2), C output cols 2,3 for rows 0,1
    all_terms.extend(strassen_2x2x2(A1_idx, B1_k23, C_k23_i01))
    # A1*B1, k=4 (2x2*2x1), C output col 4 for rows 0,1
    # Schoolbook: 4 terms
    for i in range(2):
        for j in range(2):
            u = {A1_idx[(i,j)]: 1}
            v = {B1_k4[(j,0)]: 1}  # B1_k4[j] maps to B[0+j,4]
            w = {C_k4_i01[(0,i)]: 1}  # C[i,4] -> (k=4, i)
            all_terms.append((u, v, w))

    # A2*B2, k=0:2
    all_terms.extend(strassen_2x2x2(A2_idx, B2_k01, C_k01_i01))
    # A2*B2, k=2:3
    all_terms.extend(strassen_2x2x2(A2_idx, B2_k23, C_k23_i01))
    # A2*B2, k=4
    for i in range(2):
        for j in range(2):
            u = {A2_idx[(i,j)]: 1}
            v = {B2_k4[(j,0)]: 1}
            w = {C_k4_i01[(0,i)]: 1}
            all_terms.append((u, v, w))

    # --- Bottom block: i=2 (1 row of C) ---
    # A_bottom: row 2 of A, cols 0,1,2,3
    # Split j into {0,1} and {2,3}
    # b1 = A[2,0:2] (1x2), b2 = A[2,2:4] (1x2)
    # C_bottom[k] = b1[k=0:1]*B[0:2,k] + b2[k=2:3]*B[2:4,k] ... 
    # Actually: C_bottom[k] = sum_{j=0}^{3} A[2,j]*B[j,k]
    #           = sum_{j=0}^{1} A[2,j]*B[j,k] + sum_{j=2}^{3} A[2,j]*B[j,k]

    # U row for A[2,j] = 2*4+j = 8+j
    b1_idx = {0: 8+0, 1: 8+1}  # A[2,0], A[2,1]
    b2_idx = {0: 8+2, 1: 8+3}  # A[2,2], A[2,3]

    # C_bottom: W row for C[2,k] = k*3+2
    C_k01_bot = {(0,0): 0*3+2}  # C[2,0] -> (k=0, i=2), but C_indices expects (k,i)
    # Wait, for 1x2*2x2 the C output is 1x2, so C[0,k] for k=0,1
    # W row = k*3 + i where i=2
    # For 1x2x2 with "i=0" being the single row:
    # C indices: (k, 0) -> actual W row k*3+2

    # b1*B1: 1x2 * 2x5, split k into {0,1},{2,3},{4}
    # k=0:1 (1x2*2x2), 4 terms
    # k=2:3 (1x2*2x2), 4 terms  
    # k=4   (1x2*2x1), 2 terms

    # For k-block {0,1}:
    B1_k01_for_1x2 = {(0,0): 0*5+0, (0,1): 0*5+1, (1,0): 1*5+0, (1,1): 1*5+1}
    C_k01_bot_idx = {(0,0): 0*3+2, (1,0): 1*3+2}  # (k,i=0 in local) -> actual W
    # Hmm, I need to be more careful with the mapping.
    
    # Let me redo this with a cleaner abstraction.

    all_terms = []

    # Helper: add a single rank-1 term
    def add_term(u_dict, v_dict, w_dict):
        all_terms.append((u_dict, v_dict, w_dict))

    # TOP BLOCK: i in {0,1}
    # A_top[2x4] * B[4x5] -> C_top[2x5]
    # Split j: {0,1} and {2,3}; split k: {0,1}, {2,3}, {4}
    
    # Sub-block: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, rank 7)
    # U rows: A[i,j] = i*4+j for i=0,1 j=0,1 -> 0,1,4,5
    # V rows: B[j,k] = j*5+k for j=0,1 k=0,1 -> 0,1,5,6
    # W rows: C[i,k] = k*3+i for i=0,1 k=0,1 -> 0,1,3,4
    
    A_topleft = {(0,0):0, (0,1):1, (1,0):4, (1,1):5}
    B_topleft = {(0,0):0, (0,1):1, (1,0):5, (1,1):6}
    C_topleft = {(0,0):0, (0,1):3, (1,0):1, (1,1):4}  # (j,k) for V, (k,i) for W
    
    all_terms.extend(strassen_2x2x2(A_topleft, B_topleft, C_topleft))

    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Strassen, rank 7)
    B_topmid = {(0,0):2, (0,1):3, (1,0):7, (1,1):8}  # B[j,2] and B[j,3] for j=0,1
    C_topmid = {(0,0):2, (0,1):5, (1,0):6, (1,1):9}  # C[i,k] for i=0,1 k=2,3 -> k*3+i

    all_terms.extend(strassen_2x2x2(A_topleft, B_topmid, C_topmid))

    # A[0:2, 0:2] * B[0:2, 4] -> C[0:2, 4] (schoolbook 2x2x1, rank 4)
    B_topright = {(0,0):4, (0,1):4, (1,0):9, (1,1):9}  # B[j,4] for j=0,1
    # Actually B[j,4]: j=0 -> 0*5+4=4, j=1 -> 1*5+4=9
    # For 2x2 * 2x1: A[i,j]*B[j,4], C[i,4]
    # Rank 4 terms:
    for i in range(2):
        for j in range(2):
            u = {A_topleft[(i,j)]: 1}
            v = {(j*5+4): 1}  # B[j,4]
            w = {(4*3+i): 1}  # C[i,4] -> (k=4, i)
            add_term(u, v, w)

    # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (Strassen, rank 7)
    A_topright_part = {(0,0):2, (0,1):3, (1,0):6, (1,1):7}  # A[i,2], A[i,3]
    B_botleft = {(0,0):10, (0,1):11, (1,0):15, (1,1):16}  # B[j,k] j=2,3 k=0,1
    C_topleft2 = {(0,0):0, (0,1):3, (1,0):1, (1,1):4}

    all_terms.extend(strassen_2x2x2(A_topright_part, B_botleft, C_topleft2))

    # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (Strassen, rank 7)
    B_botmid = {(0,0):12, (0,1):13, (1,0):17, (1,1):18}
    C_topmid2 = {(0,0):2, (0,1):5, (1,0):6, (1,1):9}

    all_terms.extend(strassen_2x2x2(A_topright_part, B_botmid, C_topmid2))

    # A[0:2, 2:4] * B[2:4, 4] -> C[0:2, 4] (schoolbook 2x2x1, rank 4)
    B_botright = {(0,0):14, (0,1):14, (1,0):19, (1,1):19}
    for i in range(2):
        for j in range(2):
            u = {A_topright_part[(i,j)]: 1}
            v = {( (2+j)*5+4 ): 1}  # B[2+j,4]
            w = {(4*3+i): 1}
            add_term(u, v, w)

    # BOTTOM BLOCK: i=2
    # A[2, :] (1x4) * B[4x5] -> C[2, :] (1x5)
    # Split j: {0,1} and {2,3}; split k: {0,1}, {2,3}, {4}

    # A[2,0:2] * B[0:2, 0:2] -> C[2, 0:2] (1x2 * 2x2, schoolbook rank 4)
    A_bottom_left = {0: 8, 1: 9}  # A[2,0], A[2,1]
    B_botleft_full = {(0,0):0, (0,1):1, (1,0):5, (1,1):6}
    for j in range(2):
        for k in range(2):
            u = {A_bottom_left[j]: 1}
            v = {B_botleft_full[(j,k)]: 1}
            w = {(k*3+2): 1}  # C[2,k]
            add_term(u, v, w)

    # A[2,0:2] * B[0:2, 2:4] -> C[2, 2:4] (rank 4)
    B_topmid_full = {(0,0):2, (0,1):3, (1,0):7, (1,1):8}
    for j in range(2):
        for k in range(2):
            u = {A_bottom_left[j]: 1}
            v = {B_topmid_full[(j,k)]: 1}
            w = {((2+k)*3+2): 1}
            add_term(u, v, w)

    # A[2,0:2] * B[0:2, 4] -> C[2, 4] (1x2 * 2x1, rank 2)
    for j in range(2):
        u = {A_bottom_left[j]: 1}
        v = {(j*5+4): 1}
        w = {(4*3+2): 1}
        add_term(u, v, w)

    # A[2,2:4] * B[2:4, 0:2] -> C[2, 0:2] (rank 4)
    A_bottom_right = {0: 10, 1: 11}  # A[2,2], A[2,3]
    for j in range(2):
        for k in range(2):
            u = {A_bottom_right[j]: 1}
            v = {B_botleft_full[(j,k)]: 1}
            w = {(k*3+2): 1}
            add_term(u, v, w)

    # A[2,2:4] * B[2:4, 2:4] -> C[2, 2:4] (rank 4)
    for j in range(2):
        for k in range(2):
            u = {A_bottom_right[j]: 1}
            v = {B_botmid[(j,k)]: 1}
            w = {((2+k)*3+2): 1}
            add_term(u, v, w)

    # A[2,2:4] * B[2:4, 4] -> C[2, 4] (rank 2)
    for j in range(2):
        u = {A_bottom_right[j]: 1}
        v = {((2+j)*5+4): 1}
        w = {(4*3+2): 1}
        add_term(u, v, w)

    rank = len(all_terms)
    # Build U, V, W matrices
    U = [[Fraction(0)] * rank for _ in range(n*m)]
    V = [[Fraction(0)] * rank for _ in range(m*p)]
    W = [[Fraction(0)] * rank for _ in range(p*n)]

    for c, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for row, val in u_dict.items():
            U[row][c] += val
        for row, val in v_dict.items():
            V[row][c] += val
        for row, val in w_dict.items():
            W[row][c] += val

    # Convert to lists of ints or half-ints
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for x in row:
                # x should be integer since all coefficients are +/-1
                r.append(int(x))
            result.append(r)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
