"""Matrix multiplication 6x6x6 using Strassen on 2x2 blocks of 3x3 with schoolbook 3x3."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 tensor decomposition for 6x6 matrix multiplication.
    Strategy: Split 6x6 into 2x2 blocks of 3x3 matrices.
    Use Strassen (7 terms) for the 2x2 block multiplication, where each "scalar"
    multiplication is actually a 3x3 matrix multiplication (rank 27 each).
    Total rank = 7 * 27 = 189.
    """
    n, m, p = DIMENSIONS  # all 6
    
    # Block size
    b = 3  # 3x3 blocks
    nb = 2  # 2x2 blocks of size b
    
    # We need to map indices carefully.
    # A is 6x6, B is 6x6, C is 6x6.
    # We view A as 2x2 blocks A_00, A_01, A_10, A_11, each 3x3.
    # Similarly for B.
    
    # Strassen formulas for C = A*B where A, B are 2x2 block matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M_k is a 3x3 matrix product.
    # We decompose each 3x3 product into 27 rank-1 terms (schoolbook).
    
    # Total rank = 7 * 27 = 189
    
    rank = 7 * 27
    
    # Initialize matrices
    # U has shape (n*m, rank) = (36, 189)
    # V has shape (m*p, rank) = (36, 189)
    # W has shape (p*n, rank) = (36, 189)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a block product term
    # A_block row index r1, col c1 (in 2x2 block space)
    # B_block row index r2, col c2
    # Coefficient for U, V, W
    # This is a bit complex because Strassen combines blocks with +/-
    
    # Let's define the Strassen terms more explicitly in terms of A and B indices.
    # A_{ij} refers to block at row i, col j of the 2x2 block matrix.
    # Within each block, indices are 0..2.
    # Global A index: (i*b + r, j*b + c) where r,c in 0..2
    # Global B index: (j*b + c, k*b + d)
    # Global C index: (i*b + r, k*b + d)
    
    # For a term in M_k = X * Y, where X is a combination of A blocks and Y is a combination of B blocks.
    # We can expand X and Y into linear combinations of individual A_{i,j} and B_{i,j} entries.
    # But it's easier to think of it as:
    # For each of the 7 M_k, we have 27 schoolbook terms.
    # Each schoolbook term l (0..26) corresponds to:
    #   U contribution: 1 at position corresponding to A element
    #   V contribution: 1 at position corresponding to B element
    #   W contribution: 1 at position corresponding to C element
    # But the "A element" is not a single global index; it's a sum of several global indices (the block).
    
    # Let's do it properly by enumerating.
    
    # Define Strassen terms as:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # And C contributions:
    # C00 += M1 + M4 - M5 + M7
    # C01 += M3 + M5
    # C10 += M2 + M4
    # C11 += M1 - M2 + M3 + M6
    
    # For each M_k, we decompose the 3x3 product into 27 rank-1 terms.
    # A 3x3 product X * Y where X, Y are 3x3 can be written as sum_{j=0}^{2} (X_col_j) * (Y_row_j)
    # where X_col_j is the j-th column of X (as a vector), Y_row_j is the j-th row of Y.
    # In tensor decomposition terms:
    # U[:, term] = X_col_j (padded to full A dimension)
    # V[:, term] = Y_row_j (padded to full B dimension)
    # W[:, term] = outer product contribution to C
    
    # Let's build this systematically.
    
    # First, define the block structure indices.
    # Block (bi, bj) of A covers global rows bi*b to (bi+1)*b-1, cols bj*b to (bj+1)*b-1.
    
    # For a given M_k, we need to specify:
    # - Which A blocks and B blocks are involved, and with what signs.
    # - Then for each of the 27 sub-terms (j=0,1,2 in the 3x3 product):
    #   - U contribution: sum over A blocks of sign * unit vector for A block column j
    #   - V contribution: sum over B blocks of sign * unit vector for B block row j
    #   - W contribution: for each C block that receives this M_k, add the outer product
    
    # Let's define the Strassen terms data structure.
    # Each term is (name, A_blocks, B_blocks, C_blocks_with_signs)
    # A_blocks: list of (block_row, block_col, sign)
    # B_blocks: list of (block_row, block_col, sign)
    # C_blocks_with_signs: list of (block_row, block_col, sign)
    
    strassen_terms = [
        # M1 = (A00 + A11) * (B00 + B11)
        {
            'A': [(0, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1), (1, 1, 1)]
        },
        # M2 = (A10 + A11) * B00
        {
            'A': [(1, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1)],
            'C': [(1, 0, 1), (1, 1, -1)]
        },
        # M3 = A00 * (B01 - B11)
        {
            'A': [(0, 0, 1)],
            'B': [(0, 1, 1), (1, 1, -1)],
            'C': [(0, 1, 1), (1, 1, 1)]
        },
        # M4 = A11 * (B10 - B00)
        {
            'A': [(1, 1, 1)],
            'B': [(1, 0, 1), (0, 0, -1)],
            'C': [(0, 0, 1), (1, 0, 1)]
        },
        # M5 = (A00 + A01) * B11
        {
            'A': [(0, 0, 1), (0, 1, 1)],
            'B': [(1, 1, 1)],
            'C': [(0, 0, -1), (0, 1, 1)]
        },
        # M6 = (A10 - A00) * (B00 + B01)
        {
            'A': [(1, 0, 1), (0, 0, -1)],
            'B': [(0, 0, 1), (0, 1, 1)],
            'C': [(1, 1, 1)]
        },
        # M7 = (A01 - A11) * (B10 + B11)
        {
            'A': [(0, 1, 1), (1, 1, -1)],
            'B': [(1, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1)]
        },
    ]
    
    term_idx = 0
    for st in strassen_terms:
        # For each of the 27 schoolbook sub-terms (middle index j in 0..2)
        for j in range(b):  # j is the column of A block / row of B block in the 3x3 product
            col_U = term_idx * b + j  # This is wrong; each M_k has 27 terms, not b terms
            
            # Actually, for a 3x3 product, there are 27 terms, one for each (r, j, d) where
            # r is row in A block, j is middle index, d is col in B block.
            # U picks out A[r, j] for each r
            # V picks out B[j, d] for each d
            # W picks out C[r, d]
            
            # Wait, the schoolbook decomposition of a 3x3 product X*Y:
            # (X*Y)_{rd} = sum_j X_{rj} Y_{jd}
            # So each term is indexed by (r, j, d) but we can fix j and iterate r, d?
            # No, in tensor decomposition, each rank-1 term is:
            # U vector: has 1 at position (r, j) in X, 0 elsewhere
            # V vector: has 1 at position (j, d) in Y, 0 elsewhere
            # W vector: has 1 at position (r, d) in Z, 0 elsewhere
            # Sum over r, j, d gives (X*Y)_{rd} = sum_j X_{rj} Y_{jd}
            
            # So for each j (0..2), and for each r (0..2) and d (0..2), we have a term.
            # That's 3*3*3 = 27 terms per M_k.
            
            pass
        
        # Let me redo this properly.
        for r in range(b):
            for j in range(b):
                for d in range(b):
                    # This is one rank-1 term for the 3x3 product
                    # U contribution: for each A block (ar, ac) with sign sa in st['A']:
                    #   U[(ar*b + r), (ac*b + j)] += sa
                    # V contribution: for each B block (br, bc) with sign sb in st['B']:
                    #   V[(br*b + j), (bc*b + d)] += sb
                    # W contribution: for each C block (cr, cc) with sign sc in st['C']:
                    #   W[(cr*b + r), (cc*b + d)] += sc * (sa * sb for each A,B pair? No!)
                    
                    # Wait, this is tricky. The Strassen formula is:
                    # M_k = (sum of A blocks with signs) * (sum of B blocks with signs)
                    # = sum_{A_block, B_block} sa * sb * (A_block * B_block)
                    
                    # So each M_k is actually a sum of products of individual blocks.
                    # For M1: (A00 + A11) * (B00 + B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
                    # Each of these 4 block products is a 3x3 product, each rank 27.
                    # So M1 actually has 4*27 = 108 rank-1 terms!
                    
                    # I think I need to expand the Strassen terms into individual block products first.
                    
                    break
            break
        break
    
    # Let me restructure. Expand each Strassen M_k into a sum of (A_block * B_block) products.
    
    expanded_terms = []
    for st in strassen_terms:
        for (ar, ac, sa) in st['A']:
            for (br, bc, sb) in st['B']:
                # This is one block product: A_{ar,ac} * B_{br,bc}
                # But wait, the block product A_{ar,ac} * B_{br,bc} only makes sense
                # if ac == br (column of A matches row of B).
                # In Strassen, the blocks are aligned so that this holds.
                # Let's check: In M1, A00 * B00: ac=0, br=0. OK.
                # A00 * B11: ac=0, br=1. NOT aligned!
                
                # Hmm, this is a problem. In standard Strassen for 2x2:
                # M1 = (A00+A11)(B00+B11)
                # A00 is top-left, A11 is bottom-right.
                # B00 is top-left, B11 is bottom-right.
                # A00 * B00: valid, contributes to C00
                # A00 * B11: A00 is row 0, B11 is col 1. This is A[0,0] * B[1,1]
                #   This contributes to C[0,1] if we're doing full matrix mult, but in block Strassen,
                #   M1 is used to compute both C00 and C11.
                
                # I think the issue is that in Strassen, the intermediate products M_k are
                # not directly block products; they are full 2x2 block matrix products where
                # each "entry" is a bxb block.
                
                # So M1 = (A00+A11) * (B00+B11) where A00+A11 is a 1x1 block matrix (conceptually)
                # No, that's not right either.
                
                # Let me think again. In Strassen for 2x2 block matrices:
                # A = [[A00, A01], [A10, A11]]
                # B = [[B00, B01], [B10, B11]]
                # M1 = (A00 + A11) * (B00 + B11)
                # Here A00 + A11 is a bxb matrix, B00 + B11 is a bxb matrix.
                # Their product is a bxb matrix.
                # This M1 is then used in:
                # C00 = M1 + M4 - M5 + M7
                # C11 = M1 - M2 + M3 + M6
                # So M1 contributes to both C00 and C11.
                
                # OK so my original approach was correct: each M_k is a single bxb matrix product.
                # The "A side" of M_k is a sum of A blocks, the "B side" is a sum of B blocks.
                # When we do the schoolbook decomposition of this bxb product, we need to
                # account for the fact that the A side is a sum.
                
                # For U: for each r, j, the U vector should have contributions from each A block
                # in the sum, at the corresponding global position.
                
                break
            break
        break
    
    # Let me just implement it directly.
    
    rank = 7 * 27
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for st in strassen_terms:
        # For this Strassen term, we have a bxb product of:
        # X = sum_{(ar,ac,sa) in st['A']} sa * A_{ar,ac}
        # Y = sum_{(br,bc,sb) in st['B']} sb * B_{br,bc}
        # The product X*Y is a bxb matrix.
        # Schoolbook: (X*Y)_{rd} = sum_j X_{rj} Y_{jd}
        # So for each j (0..b-1), we decompose X_{.,j} and Y_{j,.}
        # But in rank-1 terms, we fix r, j, d and create one term.
        
        for r in range(b):
            for j in range(b):
                for d in range(b):
                    # This is one rank-1 term for the bxb product.
                    # U contribution: X_{rj} = sum_{(ar,ac,sa) in st['A']} sa * A_{ar,ac}[r,j]
                    # So U[(ar*b + r), (ac*b + j)] += sa for each (ar,ac,sa) in st['A']
                    for (ar, ac, sa) in st['A']:
                        global_a_row = ar * b + r
                        global_a_col = ac * b + j
                        U[global_a_row * m + global_a_col, col] += sa
                    
                    # V contribution: Y_{jd} = sum_{(br,bc,sb) in st['B']} sb * B_{br,bc}[j,d]
                    # So V[(br*b + j), (bc*b + d)] += sb for each (br,bc,sb) in st['B']
                    for (br, bc, sb) in st['B']:
                        global_b_row = br * b + j
                        global_b_col = bc * b + d
                        V[global_b_row * p + global_b_col, col] += sb
                    
                    # W contribution: C_{rd} gets contribution from this term.
                    # But which C blocks receive this M_k?
                    # The coefficient is 1 (since we're decomposing X*Y directly).
                    # The C block (cr, cc) receives this with sign sc.
                    # So W[(cr*b + r), (cc*b + d)] += sc
                    for (cr, cc, sc) in st['C']:
                        global_c_row = cr * b + r
                        global_c_col = cc * b + d
                        W[global_c_row * n + global_c_col, col] += sc
                    
                    col += 1
    
    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
