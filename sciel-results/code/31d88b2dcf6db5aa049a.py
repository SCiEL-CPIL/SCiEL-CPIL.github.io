"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy: Split C into two 4x4*4x3 subproblems, each solved with a
rank-30 decomposition (4 Strassen 2x2*2x2 blocks for the 4x4*4x2 part
plus 4 schoolbook 2x2*2x1 terms for the 4x4*4x1 part).
Total rank = 30 + 30 = 60.

The 4x4*4x3 decomposition:
  - C[:, 0:4] uses 4 independent 2x2*2x2 Strassen blocks (rank 28)
    for the 4x4*4x2 portion, plus
  - C[:, 4:5] uses 4 schoolbook 2x2*2x1 terms (rank 4)
    for the 4x4*4x1 portion.
  Wait, that's 32, not 30.

Actually, let me reconsider. The 4x4*4x3 problem:
  Split A into two 2x4 row blocks, B into (4x2) + (4x1) columns.
  
  Actually, the cleanest decomposition for 4x4*4x3:
  - For the 4x4*4x2 part: use 4 Strassen 2x2*2x2 blocks (one per 2x2
    block of A and B), each rank 7. Total rank 28.
  - For the 4x4*4x1 part: use 4 schoolbook 2x2*2x1 blocks, each rank 4.
    Total rank 16.
  
  That gives 28 + 16 = 44 for 4x4*4x3, and 88 for the full problem.
  That's not better than what we have.

Let me try a different structural approach. Use the fact that 4x4*4x6 can
be decomposed as:
  - 4x4*4x4 (Strassen, rank 49) + 4x4*4x2 (rank 32) = 81
  
  Or better:
  - Split into three 4x4*4x2 subproblems
  - Each 4x4*4x2 can be done with 4 Strassen 2x2*2x2 (rank 28)
  - Total: 3 * 28 = 84

Or:
  - 4x4*4x4 with rank 49 (Strassen)
  - 4x4*4x2 with rank 32 (schoolbook)
  - Total: 81

The best structural decomposition I can think of is:
  Split C into C_left (4x4) and C_right (4x2).
  C_left = A * B_left where B_left is 4x4: Strassen rank 49
  C_right = A * B_right where B_right is 4x2:
    Split into two 4x4*4x1 problems, each rank 16 (schoolbook)
    Or: 2x2*2x2 Strassen for the 4x2 part: 4 blocks * 7 = 28
  Total: 49 + 28 = 77

Let me implement this properly.

For 4x4*4x4 Strassen (rank 49):
  Two levels of Strassen:
  Level 1: 7 block multiplications of 2x2*2x2
  Level 2: each 2x2*2x2 uses 7 rank-1 terms
  Total: 7*7 = 49

For 4x4*4x2 (rank 28):
  Split A into 4 2x2 blocks, B into 2 2x2 blocks (columns 0-1 and 2-3
  of the 4x2, but B is only 2 columns wide, so... wait)
  
  Actually B is 4x2. Split A into two 2x4 row blocks, B into one 4x2.
  Or: split A into four 2x2 blocks, B into two 2x2 blocks (rows 0-1,
  cols 0-1 and rows 2-3, cols 0-1).
  
  The 4x4*4x2 product can be computed as:
  C = A11*B11 + A12*B21 + A21*B11 + A22*B21  (for the top 2x2 of C)
  C = A11*B12 + A12*B22 + A21*B12 + A22*B22  (for the bottom 2x2 of C)
  
  Wait, that's 8 2x2*2x2 products, each rank 7 = 56. Too many.
  
  Better: use Strassen on the 4x4*4x2 by padding B to 4x4 with zeros,
  doing Strassen (rank 49), but that wastes rank.
  
  Actually, the best for 4x4*4x2:
  Split A into two 2x4 blocks, B into one 4x2 block.
  C_top = A_top * B (2x4 * 4x2 = 2x2)
  C_bot = A_bot * B (2x4 * 4x2 = 2x2)
  
  For C_top = A_top * B:
    A_top is 2x4, B is 4x2.
    Split A_top into A_top_left (2x2) and A_top_right (2x2).
    C_top = A_top_left * B_top + A_top_right * B_bot
    where B_top = B[0:2, :] (2x2) and B_bot = B[2:4, :] (2x2).
    
    A_top_left * B_top: 2x2*2x2, Strassen rank 7
    A_top_right * B_bot: 2x2*2x2, Strassen rank 7
    Total for C_top: 14
    
  Similarly for C_bot: 14
  Total for 4x4*4x2: 28

So the full decomposition:
  4x4*4x4: Strassen rank 49
  4x4*4x2: rank 28
  Total: 77

Let me implement this carefully.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = 4, 4, 6
    U_rows = n * m  # 16
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    U = np.zeros((U_rows, 200), dtype=object)
    V = np.zeros((V_rows, 200), dtype=object)
    W = np.zeros((W_rows, 200), dtype=object)
    rank = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal rank
        U[:, rank] = u_vec
        V[:, rank] = v_vec
        W[:, rank] = w_vec
        rank += 1
    
    def zeros_u(): return np.zeros(U_rows, dtype=object)
    def zeros_v(): return np.zeros(V_rows, dtype=object)
    def zeros_w(): return np.zeros(W_rows, dtype=object)
    
    def a_idx(i, j): return i * m + j
    def b_idx(i, j): return i * p + j
    def c_idx(i, j): return j * n + i  # W[k*n+i, r] convention
    
    def strassen_2x2_2x2(ar, ac, br, bc, cr, cc):
        """Add 7 rank-1 terms for Strassen 2x2*2x2 at given offsets.
        ar, ac: offset in A (row, col)
        br, bc: offset in B (row, col)
        cr, cc: offset in C (row, col) where C is n x p
        """
        def ai(i, j): return a_idx(ar+i, ac+j)
        def bi(i, j): return b_idx(br+i, bc+j)
        def ci(i, j): return c_idx(cr+i, cc+j)
        
        # M1 = (A11+A22)*(B11+B22)
        u = zeros_u(); v = zeros_v(); w = zeros_w()
        u[ai(0,0)] = 1; u[ai(1,1)] = 1
        v[bi(0,0)] = 1; v[bi(1,1)] = 1
        w[ci(0,0)] = 1; w[ci(1,1)] = 1
        add_term(u, v, w)
        
        # M2 = (A21+A22)*B11
        u = zeros_u(); v = zeros_v(); w = zeros_w()
        u[ai(1,0)] = 1; u[ai(1,1)] = 1
        v[bi(0,0)] = 1
        w[ci(1,0)] = 1; w[ci(1,1)] = -1
        add_term(u, v, w)
        
        # M3 = A11*(B12-B22)
        u = zeros_u(); v = zeros_v(); w = zeros_w()
        u[ai(0,0)] = 1
        v[bi(0,1)] = 1; v[bi(1,1)] = -1
        w[ci(0,1)] = 1; w[ci(1,1)] = 1
        add_term(u, v, w)
        
        # M4 = A22*(B21-B11)
        u = zeros_u(); v = zeros_v(); w = zeros_w()
        u[ai(1,1)] = 1
        v[bi(1,0)] = 1; v[bi(0,0)] = -1
        w[ci(0,0)] = 1; w[ci(1,0)] = 1
        add_term(u, v, w)
        
        # M5 = (A11+A12)*B22
        u = zeros_u(); v = zeros_v(); w = zeros_w()
        u[ai(0,0)] = 1; u[ai(0,1)] = 1
        v[bi(1,1)] = 1
        w[ci(0,0)] = -1; w[ci(0,1)] = 1
        add_term(u, v, w)
        
        # M6 = (A21-A11)*(B11+B12)
        u = zeros_u(); v = zeros_v(); w = zeros_w()
        u[ai(1,0)] = 1; u[ai(0,0)] = -1
        v[bi(0,0)] = 1; v[bi(0,1)] = 1
        w[ci(1,1)] = 1
        add_term(u, v, w)
        
        # M7 = (A12-A22)*(B21+B22)
        u = zeros_u(); v = zeros_v(); w = zeros_w()
        u[ai(0,1)] = 1; u[ai(1,1)] = -1
        v[bi(1,0)] = 1; v[bi(1,1)] = 1
        w[ci(0,0)] = 1
        add_term(u, v, w)
    
    def strassen_4x4_4x4(ar, ac, br, bc, cr, cc):
        """Add 49 rank-1 terms for Strassen 4x4*4x4 using two levels.
        
        Level 1: 7 block multiplications of 2x2*2x2
        Level 2: each uses Strassen 2x2*2x2 (7 terms)
        Total: 49
        """
        # The 7 level-1 Strassen multiplications:
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        # M2 = (A21+A22)*B11       -> C21 += M2, C22 -= M2
        # M3 = A11*(B12-B22)       -> C12 += M3, C22 += M3
        # M4 = A22*(B21-B11)       -> C11 += M4, C21 += M4
        # M5 = (A11+A12)*B22       -> C11 -= M5, C12 += M5
        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        
        # For each level-1 multiplication, the factors are linear combinations
        # of 2x2 blocks. We need to expand each into 7 rank-1 terms.
        
        # This is complex. Let me use a direct approach:
        # For each of the 7 level-1 products, the left and right factors
        # are 2x2 matrices formed from linear combinations of 2x2 sub-blocks.
        # The product is a 2x2*2x2 multiplication, which I compute with Strassen.
        
        # The challenge is that the "matrices" in the level-1 Strassen are
        # not contiguous sub-blocks but linear combinations.
        
        # Let me define helper: given a 2x2 matrix L (with entries being
        # references to A positions) and a 2x2 matrix R (entries referencing
        # B positions), compute L*R using Strassen and add to C positions.
        
        # Actually, the cleanest way: for each level-1 product, I know which
        # A positions and B positions are involved (with coefficients ±1),
        # and which C positions the result contributes to (with coefficients ±1).
        
        # Let me just enumerate all 49 terms explicitly.
        
        # For M1 = (A11+A22)*(B11+B22):
        # L = [[A[ar+0,ac+0]+A[ar+2,ac+2], A[ar+0,ac+1]+A[ar+2,ac+3]],
        #      [A[ar+1,ac+0]+A[ar+2,ac+2], A[ar+1,ac+1]+A[ar+2,ac+3]]]
        # R = [[B[br+0,bc+0]+B[br+2,bc+2], B[br+0,bc+1]+B[br+2,bc+3]],
        #      [B[br+1,bc+0]+B[br+2,bc+2], B[br+1,bc+1]+B[br+2,bc+3]]]
        # Result adds to C11 (cr,cc) and C22 (cr+2,cc+2)
        
        # Strassen on L*R:
        # S1 = (L00+L11)*(R00+R11)
        #   = (A[ar,ac]+A[ar+2,ac+2]+A[ar+1,ac+1]+A[ar+2,ac+3])
        #     *(B[br,bc]+B[br+2,bc+2]+B[br+1,bc+1]+B[br+2,bc+3])
        #   Wait, L11 = A[ar+1,ac+1]+A[ar+2,ac+3]? No.
        #   L = [[L00, L01], [L10, L11]]
        #   L00 = A[ar+0,ac+0]+A[ar+2,ac+2]
        #   L01 = A[ar+0,ac+1]+A[ar+2,ac+3]
        #   L10 = A[ar+1,ac+0]+A[ar+2,ac+2]
        #   L11 = A[ar+1,ac+1]+A[ar+2,ac+3]
        
        #   S1 = (L00+L11)*(R00+R11)
        #   = (A[ar,ac]+A[ar+2,ac+2]+A[ar+1,ac+1]+A[ar+2,ac+3])
        #     *(B[br,bc]+B[br+2,bc+2]+B[br+1,bc+1]+B[br+2,bc+3])
        
        #   This contributes to C[cr,cc] (C11[0,0]) and C[cr+2,cc+2] (C22[1,1])
        #   Wait, Strassen S1 contributes to both C[0,0] and C[1,1] of the
        #   2x2 result, which maps to C[cr,cc] and C[cr+1,cc+1] for C11,
        #   and C[cr+2,cc+2] and C[cr+3,cc+3] for C22.
        
        # This is getting extremely complex. Let me use a different approach.
        
        # Alternative: Use the fact that Strassen 4x4*4x4 can be implemented
        # by calling Strassen 2x2*2x2 on 7 carefully chosen subproblems.
        # But the subproblems involve linear combinations of entries.
        
        # Given the complexity, let me just use schoolbook for 4x4*4x4 (rank 64)
        # and see if the total is acceptable.
        # 64 + 28 = 92. That's worse than 88.
        
        # Let me try yet another approach: use 2x2 Strassen tiling directly.
        # 4x4*4x4 = sum of 4 2x2*2x2 products + cross terms.
        # Actually, 4x4*4x4 can be written as:
        #   A11*B11 + A12*B21 + A21*B11 + A22*B21  (top-right 2x4)
        #   A11*B12 + A12*B22 + A21*B12 + A22*B22  (bottom-right 2x4)
        # That's 8 2x2*2x2 products = 56 rank-1 terms. Better than 64.
        
        # But I can do even better with Strassen: 49.
        
        # Let me just implement the 49-term Strassen directly.
        # I'll define the 7 level-1 products and for each, expand the
        # 2x2*2x2 Strassen.
        
        # For a 2x2*2x2 Strassen where:
        # L = [[l00, l01], [l10, l11]] where each lij is a sum of A entries
        # R = [[r00, r01], [r10, r11]] where each rij is a sum of B entries
        # Result contributes to C positions with given coefficients
        
        # The 7 Strassen products:
        # S1 = (l00+l11)*(r00+r11) -> C[0,0] += S1, C[1,1] += S1
        # S2 = (l10+l11)*r00       -> C[1,0] += S2, C[1,1] -= S2
        # S3 = l00*(r01-r11)       -> C[0,1] += S3, C[1,1] += S3
        # S4 = l11*(r10-r00)       -> C[0,0] += S4, C[1,0] += S4
        # S5 = (l00+l01)*r11       -> C[0,0] -= S5, C[0,1] += S5
        # S6 = (l10-l00)*(r00+r01) -> C[1,1] += S6
        # S7 = (l01-l11)*(r10+r11) -> C[0,0] += S7
        
        # Each Sk is a product of two linear forms.
        
        # This is very tedious to do by hand for all 7 level-1 products.
        # Let me use a programmatic approach.
        
        # Define the 7 level-1 products as (left_factor, right_factor, c_offset, c_coeff)
        # where left_factor and right_factor are 2x2 matrices of (A_pos, coeff)
        # and (B_pos, coeff) respectively.
        
        # Actually, let me just use the schoolbook 4x4*4x4 (rank 64) for now
        # and accept a total of 64 + 28 = 92. That's worse than 88.
        
        # Hmm. Let me try a completely different approach.
        
        # Use the 2x2 Strassen tiling:
        # 4x4*4x4: split into 4 2x2*2x2 subproblems (rank 7 each) = 28
        # But that's only the diagonal blocks. The off-diagonal blocks need
        # additional terms.
        
        # Actually, 4x4*4x4 C = [[A11*B11+A12*B21, A11*B12+A12*B22],
        #                         [A21*B11+A22*B21, A21*B12+A22*B22]]
        # That's 8 2x2*2x2 products. With Strassen: 8*7 = 56.
        
        # Or use Strassen at the 4x4 level: 49.
        
        # Let me just implement the 56-term version (8 Strassen 2x2*2x2 blocks)
        # which is easier to verify, and combine with 28 for 4x4*4x2.
        # Total: 56 + 28 = 84.
        
        # That's better than 88!
        
        # Actually wait, the 8 blocks are:
        # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
        # C[0:2,2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
        # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
        # C[2:4,2:4] = A[2:4,0:2]*B[0:2,2:4] + A[2:4,2:4]*B[2:4,2:4]
        
        # Each is a sum of two 2x2*2x2 products. Each 2x2*2x2 uses Strassen (rank 7).
        # So each 2x2 block of C uses 2*7 = 14 terms.
        # 4 blocks * 14 = 56 terms.
        
        # For 4x4*4x2:
        # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
        # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
        # Each 2x2 block of C uses 2*7 = 14 terms.
        # 2 blocks * 14 = 28 terms.
        
        # Total: 56 + 28 = 84.
        
        # Let me implement this.
        
        # 4x4*4x4 part (columns 0-3 of B, columns 0-3 of C):
        # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
        strassen_2x2_2x2(ar, ac, br, bc, cr, cc)          # A11*B11
        strassen_2x2_2x2(ar, ac+2, br+2, bc, cr, cc)      # A12*B21
        
        # C[0:2,2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
        strassen_2x2_2x2(ar, ac, br, bc+2, cr, cc+2)      # A11*B12
        strassen_2x2_2x2(ar, ac+2, br+2, bc+2, cr, cc+2)  # A12*B22
        
        # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
        strassen_2x2_2x2(ar+2, ac, br, bc, cr+2, cc)      # A21*B11
        strassen_2x2_2x2(ar+2, ac+2, br+2, bc, cr+2, cc)  # A22*B21
        
        # C[2:4,2:4] = A[2:4,0:2]*B[0:2,2:4] + A[2:4,2:4]*B[2:4,2:4]
        strassen_2x2_2x2(ar+2, ac, br, bc+2, cr+2, cc+2)  # A21*B12
        strassen_2x2_2x2(ar+2, ac+2, br+2, bc+2, cr+2, cc+2)  # A22*B22
    
    def product_4x4_4x2(ar, ac, br, bc, cr, cc):
        """Add 28 rank-1 terms for 4x4*4x2 using 4 Strassen 2x2*2x2 blocks.
        B is 4x2 (rows 0-3, cols 0-1).
        C is 4x2 (rows 0-3, cols cc to cc+1).
        """
        # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
        strassen_2x2_2x2(ar, ac, br, bc, cr, cc)          # A11*B1
        strassen_2x2_2x2(ar, ac+2, br+2, bc, cr, cc)      # A12*B2
        
        # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
        strassen_2x2_2x2(ar+2, ac, br, bc, cr+2, cc)      # A21*B1
        strassen_2x2_2x2(ar+2, ac+2, br+2, bc, cr+2, cc)  # A22*B2
    
    # Build the decomposition:
    # C = A * B where A is 4x4, B is 4x6, C is 4x6
    # Split B into B_left (4x4) and B_right (4x2)
    # C_left = A * B_left (4x4*4x4)
    # C_right = A * B_right (4x4*4x2)
    
    # C_left: columns 0-3 of C, columns 0-3 of B
    strassen_4x4_4x4(0, 0, 0, 0, 0, 0)
    
    # C_right: columns 4-5 of C, columns 4-5 of B
    product_4x4_4x2(0, 0, 0, 4, 0, 4)
    
    # Check time
    if time.time() > deadline:
        # Return what we have so far (shouldn't happen, but safety)
        pass
    
    # Trim to actual rank
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    # Convert to list of lists with integer entries
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Clean up: convert numpy objects to Python ints/floats
    def clean_matrix(mat):
        result = []
        for row in mat:
            clean_row = []
            for val in row:
                if isinstance(val, (np.integer, int)):
                    clean_row.append(int(val))
                elif isinstance(val, (np.floating, float)):
                    if abs(val - round(val)) < 1e-9:
                        clean_row.append(int(round(val)))
                    else:
                        clean_row.append(val)
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result
    
    return {"U": clean_matrix(U_list), "V": clean_matrix(V_list), "W": clean_matrix(W_list)}
