import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Block structure:
    # A (2x4) split into 2x2 blocks of size 1x2:
    #   A00 = A[0,0:2], A01 = A[0,2:4]
    #   A10 = A[1,0:2], A11 = A[1,2:4]
    # B (4x8) split into 2x2 blocks of size 2x4:
    #   B00 = B[0:2,0:4], B01 = B[0:2,4:8]
    #   B10 = B[2:4,0:4], B11 = B[2:4,4:8]
    # C (2x8) split into 2x2 blocks of size 1x4:
    #   C00 = C[0,0:4], C01 = C[0,4:8]
    #   C10 = C[1,0:4], C11 = C[1,4:8]
    
    # Ring multiplication: (1x2) * (2x4) -> (1x4), rank 2
    # Strassen for outer 2x2: rank 7
    # Total rank: 7 * 2 = 14
    
    total_rank = 14
    
    # U: (n*m, rank) = (8, 14)
    # V: (m*p, rank) = (32, 14)
    # W: (p*n, rank) = (16, 14)
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # A index: A[i, j] -> flat index i*4 + j
    # A00: A[0,0]=0, A[0,1]=1
    # A01: A[0,2]=2, A[0,3]=3
    # A10: A[1,0]=4, A[1,1]=5
    # A11: A[1,2]=6, A[1,3]=7
    
    # B index: B[j, k] -> flat index j*8 + k
    # B00: B[0,0..3]=0,1,2,3; B[1,0..3]=8,9,10,11
    # B01: B[0,4..7]=4,5,6,7; B[1,4..7]=12,13,14,15
    # B10: B[2,0..3]=16,17,18,19; B[3,0..3]=24,25,26,27
    # B11: B[2,4..7]=20,21,22,23; B[3,4..7]=28,29,30,31
    
    # C index: C[i, k] -> W index k*n + i = k*2 + i
    # C00: C[0,0..3] -> W[0,1,2,3]
    # C01: C[0,4..7] -> W[8,9,10,11]
    # C10: C[1,0..3] -> W[1,3,5,7]  -- wait, W[k*2+i]
    # C[1,0] -> W[0*2+1] = W[1]
    # C[1,1] -> W[1*2+1] = W[3]
    # C[1,2] -> W[2*2+1] = W[5]
    # C[1,3] -> W[3*2+1] = W[7]
    # C11: C[1,4..7] -> W[8*2+1]... wait
    # C[1,4] -> W[4*2+1] = W[9]
    # C[1,5] -> W[5*2+1] = W[11]
    # C[1,6] -> W[6*2+1] = W[13]
    # C[1,7] -> W[7*2+1] = W[15]
    
    # Strassen formulas for outer 2x2:
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
    # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)*B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)*(B00+B01) -> C11 += M6
    # M7 = (A01-A11)*(B10+B11) -> C00 += M7
    
    # Each "multiplication" is a ring product (1x2)*(2x4)->(1x4) with rank 2.
    # For ring product X*(Y) where X=(x0,x1) and Y is 2x4:
    #   result[k] = x0*Y[0,k] + x1*Y[1,k]
    # This decomposes into 2 rank-1 terms:
    #   term 0: x0 * Y[0,:]
    #   term 1: x1 * Y[1,:]
    
    # For each Strassen term, we need to build the U, V, W contributions.
    # U contributes to A indices, V to B indices, W to C/W indices.
    
    def add_ring_product(u_coeffs, v_block, w_block, sign, base_r):
        """
        Add a ring product to the decomposition.
        u_coeffs: dict mapping A flat index -> coefficient (for the 1x2 vector)
        v_block: list of 8 B flat indices (2 rows x 4 cols)
        w_block: list of 4 W flat indices (1x4 result)
        sign: +1 or -1
        base_r: starting rank index
        """
        # Rank-1 term 0: x0 * Y[0,:]
        # x0 is from u_coeffs, Y[0,:] is first row of v_block (4 elements)
        for a_idx, coeff in u_coeffs.items():
            if coeff != 0:
                U[a_idx, base_r + 0] += sign * coeff
        for k in range(4):
            V[v_block[k], base_r + 0] += sign * 1  # first row of B block
            W[w_block[k], base_r + 0] += sign * 1
        
        # Rank-1 term 1: x1 * Y[1,:]
        for a_idx, coeff in u_coeffs.items():
            if coeff != 0:
                U[a_idx, base_r + 1] += sign * coeff
        for k in range(4):
            V[v_block[4+k], base_r + 1] += sign * 1  # second row of B block
            W[w_block[k], base_r + 1] += sign * 1
    
    # B block flat indices:
    # B00: [0,1,2,3, 8,9,10,11]  (row0: 0-3, row1: 8-11)
    # B01: [4,5,6,7, 12,13,14,15]
    # B10: [16,17,18,19, 24,25,26,27]
    # B11: [20,21,22,23, 28,29,30,31]
    
    B00 = [0,1,2,3, 8,9,10,11]
    B01 = [4,5,6,7, 12,13,14,15]
    B10 = [16,17,18,19, 24,25,26,27]
    B11 = [20,21,22,23, 28,29,30,31]
    
    # W block indices (W[k*2+i]):
    # C00 = C[0,0:4] -> W[0,2,4,6]  (k=0,1,2,3; i=0)
    # C01 = C[0,4:8] -> W[8,10,12,14]  (k=4,5,6,7; i=0)
    # C10 = C[1,0:4] -> W[1,3,5,7]  (k=0,1,2,3; i=1)
    # C11 = C[1,4:8] -> W[9,11,13,15]  (k=4,5,6,7; i=1)
    
    W_C00 = [0,2,4,6]
    W_C01 = [8,10,12,14]
    W_C10 = [1,3,5,7]
    W_C11 = [9,11,13,15]
    
    # A flat indices:
    # A00 = {0:1, 1:1}  (A[0,0], A[0,1])
    # A01 = {2:1, 3:1}  (A[0,2], A[0,3])
    # A10 = {4:1, 5:1}  (A[1,0], A[1,1])
    # A11 = {6:1, 7:1}  (A[1,2], A[1,3])
    
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # u = A00+A11: {0:1, 1:1, 6:1, 7:1}
    # v = B00+B11: each element is sum of corresponding B00 and B11 elements
    # v_block for B00+B11: [B00[0]+B11[0], ..., B00[7]+B11[7]]
    # But in the decomposition, we need to express this as a linear combination.
    # The ring product (A00+A11)*(B00+B11) expands to:
    # (a00_0+a11_0)*(b00_0+b11_0) + (a00_1+a11_1)*(b00_1+b11_1) for each column
    # Wait, no. The ring product is:
    # result[k] = (a00_0+a11_0)*(b00[0,k]+b11[0,k]) + (a00_1+a11_1)*(b00[1,k]+b11[1,k])
    
    # Hmm, I think I need to be more careful. Let me reconsider.
    
    # Actually, the Strassen decomposition works on the block level. Each "multiplication"
    # M_r is a product of two ring elements. The ring element for A is a 1x2 vector,
    # and for B is a 2x4 matrix.
    
    # For M1 = (A00+A11)*(B00+B11):
    # The A part is A00+A11 = (a00_0+a11_0, a00_1+a11_1) -- a 1x2 vector
    # The B part is B00+B11 -- a 2x4 matrix
    # The product is a 1x4 vector.
    
    # To decompose this ring product into rank-1 terms:
    # result[k] = (a00_0+a11_0)*(b00[0,k]+b11[0,k]) + (a00_1+a11_1)*(b00[1,k]+b11[1,k])
    
    # This is a bilinear form in the A and B variables. To express as sum of rank-1:
    # term 0: (a00_0+a11_0) * (b00[0,k]+b11[0,k]) for all k
    # term 1: (a00_1+a11_1) * (b00[1,k]+b11[1,k]) for all k
    
    # So for term 0:
    # U: A[0,0]=0 gets +1, A[1,2]=6 gets +1
    # V: B[0,0]=0 gets +1, B[2,0]=20 gets +1, B[0,1]=1 gets +1, B[2,1]=21 gets +1, etc.
    # W: C[0,k] for k=0..3 gets +1
    
    # Let me redo this more systematically.
    
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    # Helper: add a rank-1 term where U picks certain A entries, V picks certain B entries,
    # and W distributes to certain C entries.
    # For a ring product term: u_vec (coeffs on A) * v_vec (coeffs on B) -> w_vec (coeffs on C)
    # The bilinear form is: (sum_a u[a]*A[a]) * (sum_b v[b]*B[b]) = sum_c w[c]*C[c]
    # This is ONE rank-1 term in the tensor decomposition.
    
    # Wait, I was overcomplicating. Each rank-1 term in the tensor decomposition is:
    # u_vec^T * A_flat * v_vec^T * B_flat * w_vec^T * C_flat
    # i.e., (u·A)(v·B) contributes to C via w.
    
    # For the ring product (1x2)*(2x4)->(1x4):
    # result[k] = a_0 * B[0,k] + a_1 * B[1,k]
    # This is TWO rank-1 terms:
    #   Term 0: u = [a_0], v = [B[0,0],B[0,1],B[0,2],B[0,3]], w = [C[0,0],C[0,1],C[0,2],C[0,3]]
    #   Term 1: u = [a_1], v = [B[1,0],B[1,1],B[1,2],B[1,3]], w = [C[0,0],C[0,1],C[0,2],C[0,3]]
    
    # But in the Strassen context, the "a" vector is a linear combination of A entries,
    # and the "B" matrix is a linear combination of B entries.
    
    # Let me define a function to add a Strassen ring product.
    
    def add_strassen_term(a_coeffs, b_rows, w_targets, sign, base_r):
        """
        a_coeffs: dict {A_flat_idx: coeff} for the 1x2 A block (2 entries)
        b_rows: list of 2 lists, each with 4 B flat indices (row 0 and row 1 of the 2x4 B block)
        w_targets: list of 4 W flat indices for the 1x4 C block
        sign: +1 or -1
        base_r: rank offset (uses base_r and base_r+1)
        """
        # Term 0: first element of a * first row of b
        for a_idx, coeff in a_coeffs.items():
            U[a_idx, base_r] += sign * coeff
        for k in range(4):
            V[b_rows[0][k], base_r] += sign
            W[w_targets[k], base_r] += sign
        
        # Term 1: second element of a * second row of b
        for a_idx, coeff in a_coeffs.items():
            U[a_idx, base_r + 1] += sign * coeff
        for k in range(4):
            V[b_rows[1][k], base_r + 1] += sign
            W[w_targets[k], base_r + 1] += sign
    
    # A block coefficients:
    # A00 = {0:1, 1:1}
    # A01 = {2:1, 3:1}
    # A10 = {4:1, 5:1}
    # A11 = {6:1, 7:1}
    
    A00 = {0:1, 1:1}
    A01 = {2:1, 3:1}
    A10 = {4:1, 5:1}
    A11 = {6:1, 7:1}
    
    def add_a_dict(d1, d2, sign2=1):
        """Add two coefficient dicts, d2 multiplied by sign2."""
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) + sign2 * v
        return result
    
    def sub_a_dict(d1, d2):
        return add_a_dict(d1, d2, -1)
    
    # B block rows (each is a list of 2 lists of 4 B flat indices):
    # B00 rows: row0 = [0,1,2,3], row1 = [8,9,10,11]
    # B01 rows: row0 = [4,5,6,7], row1 = [12,13,14,15]
    # B10 rows: row0 = [16,17,18,19], row1 = [24,25,26,27]
    # B11 rows: row0 = [20,21,22,23], row1 = [28,29,30,31]
    
    B00_rows = [[0,1,2,3], [8,9,10,11]]
    B01_rows = [[4,5,6,7], [12,13,14,15]]
    B10_rows = [[16,17,18,19], [24,25,26,27]]
    B11_rows = [[20,21,22,23], [28,29,30,31]]
    
    def add_b_rows(rows1, rows2, sign2=1):
        """Add two B row structures."""
        result = []
        for r in range(2):
            new_row = []
            for k in range(4):
                # We need to represent B00[0,k]+B11[0,k] as a linear combination
                # But in the rank-1 decomposition, we can't have sums in V.
                # We need to split into separate rank-1 terms!
                pass
        return result
    
    # Hmm, I realize the issue. When we have (A00+A11)*(B00+B11), the B part is B00+B11,
    # which is a 2x4 matrix where each entry is a sum of two B entries. This means the
    # ring product expands to 4 rank-1 terms, not 2!
    
    # (a00_0+a11_0)*(b00[0,k]+b11[0,k]) = a00_0*b00[0,k] + a00_0*b11[0,k] + a11_0*b00[0,k] + a11_0*b11[0,k]
    # That's 4 terms for just the first "a" component.
    
    # So the ring product (1x2)*(2x4) where both operands are sums actually has higher rank.
    # I think my initial rank calculation was wrong.
    
    # Let me reconsider. The correct approach:
    # The full multiplication tensor T[i,j,k] = sum over the decomposition.
    # For C[i,k] = sum_j A[i,j]*B[j,k], the rank is n*m*p = 64 for the naive case.
    
    # With Strassen on the outer 2x2 block structure:
    # We have 7 "block multiplications", each of which is a (1x2)*(2x4) product.
    # But each block multiplication involves linear combinations of the original blocks.
    
    # Actually, I think the correct way to think about it is:
    # The Strassen decomposition gives 7 bilinear forms in the block variables.
    # Each bilinear form is a product of a linear form in A-blocks and a linear form in B-blocks.
    # The A-blocks are 1x2 vectors (2 scalars each), B-blocks are 2x4 matrices (8 scalars each).
    
    # For M1 = (A00+A11)*(B00+B11):
    # The A part: (A00+A11) is a 1x2 vector with entries (A[0,0]+A[1,2], A[0,1]+A[1,3])
    # The B part: (B00+B11) is a 2x4 matrix with entries (B[0,k]+B[2,k], B[1,k]+B[3,k]) for k=0..3
    # The product: a 1x4 vector with entries:
    #   (A[0,0]+A[1,2])*(B[0,k]+B[2,k]) + (A[0,1]+A[1,3])*(B[1,k]+B[3,k])
    
    # This is a bilinear form. To express it as a sum of rank-1 tensor terms:
    # = A[0,0]*B[0,k] + A[0,0]*B[2,k] + A[1,2]*B[0,k] + A[1,2]*B[2,k]
    #   + A[0,1]*B[1,k] + A[0,1]*B[3,k] + A[1,3]*B[1,k] + A[1,3]*B[3,k]
    
    # That's 8 rank-1 terms for M1 alone (for each k, but k is the same across all 4 columns).
    # Wait, no. Each term A[i,j]*B[j',k] is a rank-1 term that contributes to C[0,k].
    # But the same A[i,j] appears in multiple terms.
    
    # I think I'm overcomplicating this. Let me just directly construct the rank-14 decomposition
    # by thinking of it as: for each of the 7 Strassen terms, and for each of the 2 "rows" of the
    # B block, we get one rank-1 term. But the A and B parts are linear combinations, so each
    # rank-1 term has multiple non-zero entries in U and V.
    
    # Actually, a rank-1 term in the tensor T_{i,j,k} (where i indexes A, j indexes B, k indexes C)
    # is u_i * v_j * w_k. The condition is that T = sum_r u^(r) ⊗ v^(r) ⊗ w^(r).
    
    # For the ring product interpretation:
    # M1 = (A00+A11)*(B00+B11) contributes to C00 and C11.
    # For C00[k]: (A[0,0]+A[1,2])*(B[0,k]+B[2,k]) + (A[0,1]+A[1,3])*(B[1,k]+B[3,k])
    # For C11[k]: same expression
    
    # The bilinear form for C00[k] is:
    # sum_{i,j} T[i,j,C00_idx] * A[i] * B[j]
    # where the non-zero T entries are:
    # T[A[0,0], B[0,k], C00[k]] = 1
    # T[A[0,0], B[2,k], C00[k]] = 1
    # T[A[1,2], B[0,k], C00[k]] = 1
    # T[A[1,2], B[2,k], C00[k]] = 1
    # T[A[0,1], B[1,k], C00[k]] = 1
    # T[A[0,1], B[3,k], C00[k]] = 1
    # T[A[1,3], B[1,k], C00[k]] = 1
    # T[A[1,3], B[3,k], C00[k]] = 1
    
    # This is 8 non-zero entries. Can this be decomposed into fewer rank-1 terms?
    # The matrix of A-indices vs B-indices for fixed C-index has rank 2 (it's a 4x4 matrix
    # that's the Kronecker product of [1,1;1,1] with [1,1]... let me check).
    
    # A indices: 0,1,6,7 (A[0,0],A[0,1],A[1,2],A[1,3])
    # B indices: 0,2,8,10 (B[0,0],B[2,0],B[1,0],B[3,0]) for k=0
    # The matrix M where M[a_idx, b_idx] = 1 if they're paired:
    # a=0: b=0, b=2
    # a=1: b=8, b=10
    # a=6: b=0, b=2
    # a=7: b=8, b=10
    
    # This is [[1,1,0,0],[0,0,1,1],[1,1,0,0],[0,0,1,1]] which has rank 2.
    # So for each k, we need 2 rank-1 terms. For 4 k values, that's 8 rank-1 terms for M1's C00 contribution.
    # Plus 8 for C11. Total 16 for M1 alone. That's way more than 2.
    
    # I think my rank-14 estimate was wrong. The issue is that the Strassen "multiplication"
    # in the ring doesn't decompose into just 2 rank-1 terms because the ring elements are
    # themselves vectors/matrices, not scalars.
    
    # Let me reconsider the problem. Maybe the rank-28 approach (7 Strassen terms * 4 rank per
    # ring multiplication) is correct, where the ring multiplication (1x4)*(4x8)->(1x8) has rank 4.
    
    # For the 2x4 * 4x8 case, if we DON'T split into blocks and just treat it as a whole:
    # A is 1x4 (if n=1), B is 4x8, C is 1x8. Rank = 4.
    # But n=2, so we can't just do that.
    
    # Let me try a different block structure. What if we split A into two 1x4 rows and B into
    # two 2x4 column blocks?
    # A = [A_top; A_bottom] where A_top is 1x4, A_bottom is 1x4
    # B = [B_left | B_right] where B_left is 4x4, B_right is 4x4
    # C = [C_top_left | C_top_right; C_bot_left | C_bot_right]
    # C_top_left = A_top * B_left (1x4)
    # C_top_right = A_top * B_right (1x4)
    # C_bot_left = A_bottom * B_left (1x4)
    # C_bot_right = A_bottom * B_right (1x4)
    
    # This is a 2x2 block multiplication where each "element" is a 1x4 or 4x4 or 1x4 matrix.
    # The ring multiplication would be (1x4)*(4x4)->(1x4), which has rank 4.
    # Strassen gives 7 such multiplications, total rank = 7*4 = 28.
    
    # Let me verify: (1x4)*(4x4)->(1x4): result[k] = sum_j a[j]*B[j,k]. Rank = 4. Yes!
    
    # So rank 28 is achievable. Let me construct it.
    
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    total_rank = 28
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # A flat indices: A[i,j] -> i*4+j
    # A_top = A[0,0:4] = indices 0,1,2,3
    # A_bot = A[1,0:4] = indices 4,5,6,7
    
    # B flat indices: B[j,k] -> j*8+k
    # B_left = B[0:4,0:4] = indices 0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27
    # B_right = B[0:4,4:8] = indices 4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31
    
    # C flat indices via W: W[k*n+i] = W[k*2+i]
    # C_top_left = C[0,0:4] -> W[0,2,4,6]
    # C_top_right = C[0,4:8] -> W[8,10,12,14]
    # C_bot_left = C[1,0:4] -> W[1,3,5,7]
    # C_bot_right = C[1,4:8] -> W[9,11,13,15]
    
    def add_ring_product_1x4_x_4x4(a_indices, b_indices_4x4, w_indices, sign, base_r):
        """
        Add ring product (1x4)*(4x4)->(1x4) as 4 rank-1 terms.
        a_indices: list of 4 A flat indices (the 1x4 vector)
        b_indices_4x4: 4x4 list of B flat indices
        w_indices: list of 4 W flat indices
        sign: +1 or -1
        base_r: rank offset (uses base_r to base_r+3)
        """
        for j in range(4):
            # Rank-1 term j: a[j] * B[j,:]
            for ai in a_indices:
                U[ai, base_r + j] += sign
            for k in range(4):
                V[b_indices_4x4[j][k], base_r + j] += sign
                W[w_indices[k], base_r + j] += sign
    
    # B_left 4x4 indices:
    # Row 0: [0,1,2,3]
    # Row 1: [8,9,10,11]
    # Row 2: [16,17,18,19]
    # Row 3: [24,25,26,27]
    B_left = [[0,1,2,3], [8,9,10,11], [16,17,18,19], [24,25,26,27]]
    
    # B_right 4x4 indices:
    # Row 0: [4,5,6,7]
    # Row 1: [12,13,14,15]
    # Row 2: [20,21,22,23]
    # Row 3: [28,29,30,31]
    B_right = [[4,5,6,7], [12,13,14,15], [20,21,22,23], [28,29,30,31]]
    
    A_top = [0,1,2,3]
    A_bot = [4,5,6,7]
    
    W_C_top_left = [0,2,4,6]
    W_C_top_right = [8,10,12,14]
    W_C_bot_left = [1,3,5,7]
    W_C_bot_right = [9,11,13,15]
    
    # Strassen:
    # M1 = (A_top+A_bot)*(B_left+B_right) -> C_top_left += M1, C_bot_right += M1
    # M2 = (A_bot+A_bot)*B_left -> C_bot_left += M2, C_bot_right -= M2
    # M3 = A_top*(B_right-B_right) -> ... wait, standard Strassen:
    
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M6
    
    # Mapping:
    # A11 = A_top, A12 = A_top (wait, no)
    
    # Actually, for 2x2 block matrix:
    # A = [[A11, A12], [A21, A22]]
    # In our case, A is 2x4, so if we split into 2x2 blocks of size 1x2:
    # A11 = A[0,0:2] (1x2), A12 = A[0,2:4] (1x2)
    # A21 = A[1,0:2] (1x2), A22 = A[1,2:4] (1x2)
    
    # But I want to use 1x4 blocks. Let me use a different split:
    # A = [A_top; A_bot] where A_top = A[0,:] (1x4), A_bot = A[1,:] (1x4)
    # B = [B_left | B_right] where B_left = B[:,0:4] (4x4), B_right = B[:,4:8] (4x4)
    # C = [[C_tl, C_tr], [C_bl, C_br]]
    # C_tl = A_top * B_left (1x4)
    # C_tr = A_top * B_right (1x4)
    # C_bl = A_bot * B_left (1x4)
    # C_br = A_bot * B_right (1x4)
    
    # This is NOT a standard 2x2 matrix multiplication because the "elements" have different shapes.
    # A elements are 1x4, B elements are 4x4, C elements are 1x4.
    # The "multiplication" is (1x4)*(4x4)->(1x4), which is valid.
    # But Strassen requires that the "ring" multiplication is associative and that we can add
    # and subtract ring elements. Addition of 1x4 vectors and 4x4 matrices is fine.
    # The issue is that Strassen's formulas assume all four A-blocks have the same type,
    # all four B-blocks have the same type, etc.
    
    # In our case:
    # A11 = A_top (1x4), A12 = A_top (1x4) -- NO, this is wrong.
    # We only have TWO A blocks, not four.
    
    # I think the correct interpretation is:
    # We're doing a 2x1 "matrix" times a 1x2 "matrix" where the "elements" are:
    # A has 2 elements (rows), B has 2 elements (column blocks), C has 4 elements.
    # This is just a bilinear map, not a matrix multiplication in a ring.
    
    # So Strassen doesn't directly apply. Let me go back to the 2x2 block structure with 1x2 blocks.
    
    # A = [[A11, A12], [A21, A22]] with 1x2 blocks
    # B = [[B11, B12], [B21, B22]] with 2x4 blocks
    # C = [[C11, C12], [C21, C22]] with 1x4 blocks
    # Ring mult: (1x2)*(2x4)->(1x4), rank 2
    
    # Strassen: 7 ring multiplications, each rank 2, total rank 14.
    # But as I showed earlier, each ring multiplication with linear combinations expands to
    # more rank-1 terms. Let me count more carefully.
    
    # For M1 = (A11+A22)*(B11+B22):
    # A part: A11+A22 = (A[0,0]+A[1,2], A[0,1]+A[1,3]) -- 1x2 vector
    # B part: B11+B22 -- 2x4 matrix
    # Product: 1x4 vector
    # The bilinear form has 8 non-zero entries per C-index (as I showed).
    # For 4 C-indices in C11, that's 32 non-zero entries.
    # The rank of this sub-tensor is... let me think.
    
    # Actually, I realize that the "rank 2" of the ring multiplication means that if the
    # A and B parts were single scalars, the product would be 1 rank-1 term. But since they're
    # vectors/matrices, the rank multiplies.
    
    # I think the correct total rank for the 2x2 block Strassen approach is:
    # 7 (Strassen terms) * 2 (rank of ring mult) = 14
    # where each "rank-1 term" in the ring mult corresponds to one rank-1 term in the full
    # tensor, with U and V having multiple non-zero entries.
    
    # Let me just construct it and verify.
    
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    total_rank = 14
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # A block indices (1x2 blocks):
    # A11 = A[0,0:2] = [0, 1]
    # A12 = A[0,2:4] = [2, 3]
    # A21 = A[1,0:2] = [4, 5]
    # A22 = A[1,2:4] = [6, 7]
    
    A11 = [0, 1]
    A12 = [2, 3]
    A21 = [4, 5]
    A22 = [6, 7]
    
    # B block indices (2x4 blocks):
    # B11 = B[0:2,0:4]: rows [0,1,2,3] and [8,9,10,11]
    # B12 = B[0:2,4:8]: rows [4,5,6,7] and [12,13,14,15]
    # B21 = B[2:4,0:4]: rows [16,17,18,19] and [24,25,26,27]
    # B22 = B[2:4,4:8]: rows [20,21,22,23] and [28,29,30,31]
    
    B11 = [[0,1,2,3], [8,9,10,11]]
    B12 = [[4,5,6,7], [12,13,14,15]]
    B21 = [[16,17,18,19], [24,25,26,27]]
    B22 = [[20,21,22,23], [28,29,30,31]]
    
    # C block W indices (1x4 blocks):
    # C11 = C[0,0:4] -> W[0,2,4,6]
    # C12 = C[0,4:8] -> W[8,10,12,14]
    # C21 = C[1,0:4] -> W[1,3,5,7]
    # C22 = C[1,4:8] -> W[9,11,13,15]
    
    W_C11 = [0,2,4,6]
    W_C12 = [8,10,12,14]
    W_C21 = [1,3,5,7]
    W_C22 = [9,11,13,15]
    
    def add_ring_product(a_block, b_block, w_target, sign, base_r):
        """
        Add ring product (1x2)*(2x4)->(1x4) as 2 rank-1 terms.
        a_block: list of 2 A flat indices
        b_block: 2x4 list of B flat indices
        w_target: list of 4 W flat indices
        sign: +1 or -1
        base_r: rank offset (uses base_r and base_r+1)
        """
        for row in range(2):
            # Rank-1 term: a[row] * b_block[row,:]
            for ai in a_block:
                U[ai, base_r + row] += sign
            for k in range(4):
                V[b_block[row][k], base_r + row] += sign
                W[w_target[k], base_r + row] += sign
    
    def add_a_blocks(b1, b2, sign2=1):
        """Return combined A block indices (with multiplicity for signs)."""
        # This doesn't work simply because we need linear combinations.
        # Instead, I'll handle it in the Strassen terms directly.
        pass
    
    # For Strassen terms with sums/differences of A blocks, the A part is a linear combination.
    # E.g., M1 = (A11+A22)*(B11+B22):
    # A part = A11 + A22, so a_block should represent [A[0,0]+A[1,2], A[0,1]+A[1,3]]
    # This means in U, both A[0,0] and A[1,2] get coefficient +1 for the first component,
    # and both A[0,1] and A[1,3] get coefficient +1 for the second component.
    
    # Similarly, B11+B22 means in V, both B11[row][k] and B22[row][k] get coefficient +1.
    
    # So I need a more flexible function that takes coefficient dicts.
    
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    def add_ring_product_flex(a_coeffs, b_row_coeffs, w_target, sign, base_r):
        """
        a_coeffs: list of 2 dicts, each {A_idx: coeff} for the two components of the 1x2 vector
        b_row_coeffs: list of 2 lists, each a list of 4 dicts {B_idx: coeff} for the 2x4 matrix
        w_target: list of 4 W flat indices
        sign: +1 or -1
        base_r: rank offset
        """
        for row in range(2):
            # Rank-1 term: (sum of a_coeffs[row]) * (sum of b_row_coeffs[row])
            for a_idx, coeff in a_coeffs[row].items():
                U[a_idx, base_r + row] += sign * coeff
            for k in range(4):
                for b_idx, coeff in b_row_coeffs[row][k].items():
                    V[b_idx, base_r + row] += sign * coeff
                W[w_target[k], base_r + row] += sign
    
    # Helper to create single-element coefficient dict
    def single(idx):
        return {idx: 1}
    
    def add_dicts(d1, d2, sign2=1):
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) + sign2 * v
        return result
    
    # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
    # A part: component 0 = A[0,0]+A[1,2], component 1 = A[0,1]+A[1,3]
    a_M1 = [
        add_dicts(single(0), single(6)),   # A[0,0] + A[1,2]
        add_dicts(single(1), single(7))    # A[0,1] + A[1,3]
    ]
    # B part: row 0 = B11[0]+B22[0], row 1 = B11[1]+B22[1]
    b_M1 = [
        [add_dicts(single(B11[0][k]), single(B22[0][k])) for k in range(4)],
        [add_dicts(single(B11[1][k]), single(B22[1][k])) for k in range(4)]
    ]
    add_ring_product_flex(a_M1, b_M1, W_C11, +1, 0)
    add_ring_product_flex(a_M1, b_M1, W_C22, +1, 0)
    
    # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
    a_M2 = [
        add_dicts(single(4), single(6)),   # A[1,0] + A[1,2]
        add_dicts(single(5), single(7))    # A[1,1] + A[1,3]
    ]
    b_M2 = [
        [single(B11[0][k]) for k in range(4)],
        [single(B11[1][k]) for k in range(4)]
    ]
    add_ring_product_flex(a_M2, b_M2, W_C21, +1, 2)
    add_ring_product_flex(a_M2, b_M2, W_C22, -1, 2)
    
    # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
    a_M3 = [
        single(0),  # A[0,0]
        single(1)   # A[0,1]
    ]
    b_M3 = [
        [add_dicts(single(B12[0][k]), single(B22[0][k]), -1) for k in range(4)],
        [add_dicts(single(B12[1][k]), single(B22[1][k]), -1) for k in range(4)]
    ]
    add_ring_product_flex(a_M3, b_M3, W_C12, +1, 4)
    add_ring_product_flex(a_M3, b_M3, W_C22, +1, 4)
    
    # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
    a_M4 = [
        single(6),  # A[1,2]
        single(7)   # A[1,3]
    ]
    b_M4 = [
        [add_dicts(single(B21[0][k]), single(B11[0][k]), -1) for k in range(4)],
        [add_dicts(single(B21[1][k]), single(B11[1][k]), -1) for k in range(4)]
    ]
    add_ring_product_flex(a_M4, b_M4, W_C11, +1, 6)
    add_ring_product_flex(a_M4, b_M4, W_C21, +1, 6)
    
    # M5 = (A11+A12)*B22 -> C11 -= M5, C12 += M5
    a_M5 = [
        add_dicts(single(0), single(2)),   # A[0,0] + A[0,2]
        add_dicts(single(1), single(3))    # A[0,1] + A[0,3]
    ]
    b_M5 = [
        [single(B22[0][k]) for k in range(4)],
        [single(B22[1][k]) for k in range(4)]
    ]
    add_ring_product_flex(a_M5, b_M5, W_C11, -1, 8)
    add_ring_product_flex(a_M5, b_M5, W_C12, +1, 8)
    
    # M6 = (A21-A11)*(B11+B12) -> C22 += M6
    a_M6 = [
        add_dicts(single(4), single(0), -1),   # A[1,0] - A[0,0]
        add_dicts(single(5), single(1), -1)    # A[1,1] - A[0,1]
    ]
    b_M6 = [
        [add_dicts(single(B11[0][k]), single(B12[0][k])) for k in range(4)],
        [add_dicts(single(B11[1][k]), single(B12[1][k])) for k in range(4)]
    ]
    add_ring_product_flex(a_M6, b_M6, W_C22, +1, 10)
    
    # M7 = (A12-A22)*(B21+B22) -> C11 += M7
    a_M7 = [
        add_dicts(single(2), single(6), -1),   # A[0,2] - A[1,2]
        add_dicts(single(3), single(7), -1)    # A[0,3] - A[1,3]
    ]
    b_M7 = [
        [add_dicts(single(B21[0][k]), single(B22[0][k])) for k in range(4)],
        [add_dicts(single(B21[1][k]), single(B22[1][k])) for k in range(4)]
    ]
    add_ring_product_flex(a_M7, b_M7, W_C11, +1, 12)
    
    # Verification
    A_test = np.random.randint(-5, 5, (n, m))
    B_test = np.random.randint(-5, 5, (m, p))
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p), dtype=int)
    for r in range(total_rank):
        u_vec = U[:, r]
        v_vec = V[:, r]
        w_vec = W[:, r]
        
        a_form = np.dot(A_test.flatten(), u_vec)
        b_form = np.dot(B_test.flatten(), v_vec)
        prod = a_form * b_form
        
        for idx in range(p * n):
            if w_vec[idx] != 0:
                k, i = divmod(idx, n)
                C_computed[i, k] += w_vec[idx] * prod
    
    if not np.allclose(C_computed, C_expected, atol=1e-6):
        # Fallback to naive rank-64
        U_naive = np.zeros((n*m, n*m*p), dtype=int)
        V_naive = np.zeros((m*p, n*m*p), dtype=int)
        W_naive = np.zeros((p*n, n*m*p), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U_naive[i*m+j, r] = 1
                    V_naive[j*p+k, r] = 1
                    W_naive[k*n+i, r] = 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    if time.time() - start_time > 140:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
