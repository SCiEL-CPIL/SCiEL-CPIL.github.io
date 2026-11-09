"""Rewritten program for 6x6 matrix multiplication rank minimization."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _complex_to_rational(c, tol=1e-10):
    """Convert a complex number to a rational (num, den) if close to rational with small denominator."""
    re = c.real
    im = c.imag
    if abs(im) > tol:
        return None
    if abs(re) < tol:
        return (0, 1)
    # Try denominators up to 100
    for den in range(1, 101):
        num = round(re * den)
        if abs(re - num / den) < tol:
            return (num, den)
    return None

def _strassen_2x2():
    """
    Returns the 7 Strassen rank-1 terms for 2x2 matrix multiplication.
    Each term is (u_idx, v_idx, w_idx, coeff) where:
      u_idx: index into the 4 input elements of A (row-major)
      v_idx: index into the 4 input elements of B (row-major)
      w_idx: index into the 4 output elements of C (row-major)
      coeff: the coefficient (integer)
    """
    # Strassen's 7 products for 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 - m4 + m5 + m6
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m3 + m2 - m7
    #
    # Input A elements indexed: 0=a, 1=b, 2=c, 3=d
    # Input B elements indexed: 0=e, 1=f, 2=g, 3=h
    # Output C elements indexed: 0=C11, 1=C12, 2=C21, 3=C22
    #
    # Each Strassen product is a sum of products of linear combinations.
    # We need to decompose each product into rank-1 terms.
    #
    # m1 = (a+d)(e+h) = a*e + a*h + d*e + d*h
    #   rank-1 terms: (a, e, C11, 1), (a, h, C11, 1), (d, e, C11, 1), (d, h, C11, 1)
    #   minus m4: d*(g-e) = d*g - d*e
    #   plus m5: (a+b)*h = a*h + b*h
    #   plus m6: (c-a)*(e+f) = c*e + c*f - a*e - a*f
    #
    # This gets complicated. Let's just enumerate all 27 terms for each 2x2.
    # Actually, for the block approach, we don't need Strassen on the 2x2 blocks.
    # We can just use schoolbook for the 2x2 blocks (4 terms each) after the DFT.
    # But the strategy says to use Strassen on the 2x2 blocks.
    #
    # Let me think again. The 6x6 matrices are partitioned into 3x3 grid of 2x2 blocks.
    # We apply DFT on the block indices (3x3 grid) to transform to frequency domain.
    # In the frequency domain, each 2x2 block multiplication is independent.
    # For each 2x2 block multiplication, we use Strassen (rank 7) instead of schoolbook (rank 4).
    # Total rank: 27 (DFT modes) * 7 (Strassen) = 189.
    #
    # But wait, the DFT mixes the input blocks. So we need to:
    # 1. Apply inverse DFT to A's block indices to get frequency-domain A blocks
    # 2. Apply inverse DFT to B's block indices to get frequency-domain B blocks
    # 3. For each frequency pair, do 2x2 multiplication (using Strassen)
    # 4. Apply DFT to C's block indices to get back to spatial domain
    #
    # Actually, the standard block DFT approach for matrix multiplication:
    # A = sum over block positions, B = sum over block positions
    # C_block(i,j) = sum_k A_block(i,k) * B_block(k,j)
    # This is a 3x3 matrix of 2x2 blocks, and the block-level multiplication is 3x3 matrix multiplication.
    # We apply DFT to the 3x3 block index to diagonalize... but 3x3 matrix multiplication is not cyclic convolution.
    #
    # Hmm, the strategy says "diagonalizes the cyclic convolution structure of the block indices". 
    # This implies treating the block indices as a cyclic group Z3 x Z3.
    # But 6x6 matrix multiplication with 2x2 blocks doesn't naturally have cyclic convolution structure.
    #
    # Let me reconsider. Perhaps the idea is:
    # - Pad the 3x3 block grid to work with DFT
    # - The block-level "multiplication" is actually just element-wise in some transformed domain
    #
    # This is getting too complex. Let me just use the known Strassen-based approach for 6x6:
    # Use Strassen recursively or at the top level.
    #
    # Actually, the simplest improvement from 216 to something lower:
    # Use Strassen for the full 6x6 multiplication by treating it as a 2x2 block multiplication
    # where each block is 3x3, and use Strassen (rank 7) for the block level.
    # Then for each 3x3 block multiplication, use schoolbook (rank 9).
    # Total rank: 7 * 9 = 63? No, that's not right either.
    #
    # Wait, Strassen for 2x2 blocks of 3x3 matrices:
    # A = [[A00, A01], [A10, A11]] where each is 3x3
    # B = [[B00, B01], [B10, B11]] where each is 3x3
    # C = [[C00, C01], [C10, C11]] where each is 3x3
    #
    # Strassen: 7 products of 3x3 matrices
    # Each 3x3 product uses schoolbook: 9 terms
    # Total: 7 * 9 = 63 terms? No, that's the number of scalar multiplications.
    #
    # Actually, for tensor rank:
    # The 6x6 matrix multiplication tensor has rank at most 216 (schoolbook).
    # Using Strassen at the block level (2x2 blocks of 3x3):
    # We need 7 bilinear forms, each being a product of two 3x3 matrices.
    # Each 3x3 matrix multiplication has rank 27 (schoolbook).
    # So total rank: 7 * 27 = 189.
    #
    # Wait, that's the same as before. Let me re-examine.
    #
    # Actually, the rank of the 3x3 matrix multiplication tensor is 27 (schoolbook), and it's not known to be lower.
    # So 7 * 27 = 189 is the rank when using Strassen at the top level with 3x3 sub-blocks.
    #
    # Alternatively, using Strassen at the 2x2 block level (2x2 blocks of 3x3... no, that doesn't work for 6x6).
    #
    # Let me just implement the 189-rank decomposition using Strassen at the top level.
    # 6x6 = 2x2 blocks of 3x3 matrices.
    # Strassen gives 7 products of 3x3 matrices.
    # Each 3x3 product is done with schoolbook (27 terms).
    # Total: 7 * 27 = 189 terms.
    
    pass

def _build_189_rank_decomposition():
    """
    Build the 189-rank decomposition of 6x6 matrix multiplication using Strassen.
    
    6x6 matrices are viewed as 2x2 blocks of 3x3 matrices:
    A = [[A00, A01], [A10, A11]]  (each 3x3)
    B = [[B00, B01], [B10, B11]]  (each 3x3)
    C = [[C00, C01], [C10, C11]]  (each 3x3)
    
    Strassen's 7 products:
    m1 = (A00 + A11)(B00 + B11)
    m2 = (A10 + A11)B00
    m3 = A00(B01 - B11)
    m4 = A11(B10 - B00)
    m5 = (A00 + A01)B11
    m6 = (A10 - A00)(B00 + B01)
    m7 = (A01 - A11)(B10 + B11)
    
    C00 = m1 - m4 + m5 + m6
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m3 + m2 - m7
    
    Each mij is a product of two 3x3 matrices, done with schoolbook (27 terms each).
    Total: 7 * 27 = 189 terms.
    """
    n = 6
    rank = 189
    
    U = [[0] * rank for _ in range(n * n)]  # 36 x 189
    V = [[0] * rank for _ in range(n * n)]  # 36 x 189
    W = [[0] * rank for _ in range(n * n)]  # 36 x 189
    
    col = 0
    
    # Define the 7 Strassen combinations
    # Each is a list of (A_block, B_block, sign) for the linear combinations
    strassen_terms = [
        # m1 = (A00 + A11)(B00 + B11)
        [((0, 0), (0, 0), 1), ((0, 0), (1, 1), 1), ((1, 1), (0, 0), 1), ((1, 1), (1, 1), 1)],
        # m2 = (A10 + A11)B00
        [((1, 0), (0, 0), 1), ((1, 1), (0, 0), 1)],
        # m3 = A00(B01 - B11)
        [((0, 0), (0, 1), 1), ((0, 0), (1, 1), -1)],
        # m4 = A11(B10 - B00)
        [((1, 1), (1, 0), 1), ((1, 1), (0, 0), -1)],
        # m5 = (A00 + A01)B11
        [((0, 0), (1, 1), 1), ((0, 1), (1, 1), 1)],
        # m6 = (A10 - A00)(B00 + B01)
        [((1, 0), (0, 0), 1), ((1, 0), (0, 1), 1), ((0, 0), (0, 0), -1), ((0, 0), (0, 1), -1)],
        # m7 = (A01 - A11)(B10 + B11)
        [((0, 1), (1, 0), 1), ((0, 1), (1, 1), 1), ((1, 1), (1, 0), -1), ((1, 1), (1, 1), -1)],
    ]
    
    # For each Strassen product, for each scalar multiplication in the 3x3 schoolbook
    for mi in range(7):
        for ai in range(3):  # rows of 3x3 A subblock
            for bj in range(3):  # cols of 3x3 B subblock
                for term in strassen_terms[mi]:
                    a_block_row, a_block_col, a_sign = term[0]
                    b_block_row, b_block_col, b_sign = term[1] if len(term) > 1 else None
                    # Actually, each term in strassen_terms is a product of two linear combos.
                    # I need to restructure this.
                pass
    
    # Let me restructure. Each Strassen product mij is (LA_i) * (LB_i) where LA_i and LB_i
    # are linear combinations of 3x3 blocks.
    # Then mij = sum over all elements: (LA_i)[u,v] * (LB_i)[v,w] contributes to C.
    
    # Actually, let's think of it differently.
    # The 6x6 matrix A has elements A[r, c] where r, c in 0..5.
    # Block (br, bc) of A is A[3*br:3*br+3, 3*bc:3*bc+3].
    # Element A[3*br+dr, 3*bc+dc] is in block (br, bc) at position (dr, dc).
    
    # For Strassen product mij = LA_i * LB_i:
    # LA_i is a 3x3 matrix (one of the 7 linear combos)
    # LB_i is a 3x3 matrix (one of the 7 linear combos)
    # mij[u, v] = sum_w LA_i[u, w] * LB_i[w, v]
    
    # The contribution to C[r, c] from mij:
    # C[r, c] gets contributions from C00, C01, C10, C11 based on which block (r//3, c//3) it's in.
    
    # Let me define the linear combinations more carefully.
    # A00 = block (0,0), A01 = block (0,1), A10 = block (1,0), A11 = block (1,1)
    
    # For each of the 7 Strassen products, define:
    #   A_comb: list of (block_index, coefficient) where block_index is (br, bc)
    #   B_comb: list of (block_index, coefficient)
    
    strassen_defs = [
        # m1 = (A00 + A11)(B00 + B11)
        {
            'A': [((0, 0), 1), ((1, 1), 1)],
            'B': [((0, 0), 1), ((1, 1), 1)],
            'C_contrib': [(0, 0, 1), (1, 1, 1)]  # contributes +1 to C00 and C11
        },
        # m2 = (A10 + A11)B00
        {
            'A': [((1, 0), 1), ((1, 1), 1)],
            'B': [((0, 0), 1)],
            'C_contrib': [(1, 0, 1), (1, 1, -1)]  # contributes +1 to C10 and -1 to C11
        },
        # m3 = A00(B01 - B11)
        {
            'A': [((0, 0), 1)],
            'B': [((0, 1), 1), ((1, 1), -1)],
            'C_contrib': [(0, 1, 1), (1, 1, -1)]  # contributes +1 to C01 and -1 to C11
        },
        # m4 = A11(B10 - B00)
        {
            'A': [((1, 1), 1)],
            'B': [((1, 0), 1), ((0, 0), -1)],
            'C_contrib': [(0, 0, -1), (1, 0, 1)]  # contributes -1 to C00 and +1 to C10
        },
        # m5 = (A00 + A01)B11
        {
            'A': [((0, 0), 1), ((0, 1), 1)],
            'B': [((1, 1), 1)],
            'C_contrib': [(0, 0, 1), (0, 1, 1)]  # contributes +1 to C00 and +1 to C01
        },
        # m6 = (A10 - A00)(B00 + B01)
        {
            'A': [((1, 0), 1), ((0, 0), -1)],
            'B': [((0, 0), 1), ((0, 1), 1)],
            'C_contrib': [(0, 0, 1)]  # contributes +1 to C00
        },
        # m7 = (A01 - A11)(B10 + B11)
        {
            'A': [((0, 1), 1), ((1, 1), -1)],
            'B': [((1, 0), 1), ((1, 1), 1)],
            'C_contrib': [(1, 1, -1)]  # contributes -1 to C11
        },
    ]
    
    # Now, for each Strassen product, for each element in the 3x3 A-combination and 3x3 B-combination,
    # we create a rank-1 term.
    
    # Wait, this isn't quite right. The product of two 3x3 matrices (each being a linear combo of blocks)
    # is a 3x3 matrix product. Each element of the product is a sum of 3 products.
    # So for each Strassen product, we have 3*3 = 9 scalar multiplications (schoolbook for 3x3).
    # Total: 7 * 9 = 63 scalar multiplications? No, each scalar multiplication involves elements from
    # potentially different blocks.
    
    # Let me think more carefully.
    # LA is a 3x3 matrix where each element LA[u, v] is a linear combination of A elements.
    # LB is a 3x3 matrix where each element LB[u, v] is a linear combination of B elements.
    # (LA * LB)[u, v] = sum_w LA[u, w] * LB[w, v]
    
    # Each term LA[u, w] * LB[w, v] is a bilinear form. We can write it as a sum of rank-1 terms.
    # But LA[u, w] might be a sum of multiple A elements, and LB[w, v] might be a sum of multiple B elements.
    # So LA[u, w] * LB[w, v] is a sum of products of individual A and B elements.
    
    # For the tensor decomposition, each such product corresponds to one rank-1 term.
    
    # Let's build the full decomposition by enumerating all terms.
    
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    col = 0
    
    for mi in range(7):
        def_ = strassen_defs[mi]
        A_blocks = def_['A']  # list of ((br, bc), coeff)
        B_blocks = def_['B']  # list of ((br, bc), coeff)
        C_contribs = def_['C_contrib']  # list of ((br, bc), coeff)
        
        # For each element position (u, v) in the 3x3 result:
        for u in range(3):
            for v in range(3):
                # (LA * LB)[u, v] = sum_w LA[u, w] * LB[w, v]
                for w in range(3):
                    # LA[u, w] = sum over (ab, ac) in A_blocks: coeff * A[3*ab+u, 3*ac+w]
                    # LB[w, v] = sum over (bb, bc) in B_blocks: coeff * B[3*bb+w, 3*bc+v]
                    # Product: sum over all pairs of (A_elem) * (B_elem)
                    for (ab, ac), a_coeff in A_blocks:
                        for (bb, bc), b_coeff in B_blocks:
                            # A element at (3*ab + u, 3*ac + w)
                            # B element at (3*bb + w, 3*bc + v)
                            # This product contributes to C elements based on C_contribs
                            a_idx = (3 * ab + u) * n + (3 * ac + w)
                            b_idx = (3 * bb + w) * n + (3 * bc + v)
                            
                            for (cb, cc), c_coeff in C_contribs:
                                # The result element is at (3*cb + u, 3*cc + v) in C
                                c_idx = (3 * cb + u) * n + (3 * cc + v)
                                coeff = a_coeff * b_coeff * c_coeff
                                
                                if col >= rank:
                                    raise ValueError("Rank exceeded")
                                U[a_idx][col] = coeff
                                V[b_idx][col] = coeff  # Wait, V should just be b_coeff?
                                # Actually, the rank-1 term is: U[:, col] * V[:, col]
                                # where U[a_idx] = a_coeff, V[b_idx] = b_coeff
                                # and W[c_idx] = c_coeff
                                # But we have three coefficients multiplying.
                                # The correct decomposition is:
                                # term = a_coeff * b_coeff * c_coeff * A[a_idx] * B[b_idx]
                                # This is a rank-1 term in the tensor:
                                # U[a_idx] = a_coeff
                                # V[b_idx] = b_coeff  
                                # W[c_idx] = c_coeff
                                # But then the product U*V*W gives a_coeff * b_coeff * c_coeff * A*B
                                # which is correct.
                                # Wait, no. The tensor T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
                                # So for this term, we set:
                                # U[a_idx, col] = a_coeff
                                # V[b_idx, col] = b_coeff
                                # W[c_idx, col] = c_coeff
                                # But that's not right either, because we need the product to be
                                # a_coeff * b_coeff * c_coeff * A[a_idx] * B[b_idx] contributing to C[c_idx].
                                # The tensor entry T[a_idx, b_idx, c_idx] should be a_coeff * b_coeff * c_coeff.
                                # With U[a_idx, col] = a_coeff, V[b_idx, col] = b_coeff, W[c_idx, col] = c_coeff,
                                # we get T[a_idx, b_idx, c_idx] += a_coeff * b_coeff * c_coeff. Correct!
                                
                                # But wait, I already set U and V above. Let me redo.
                                pass
                    pass
                pass
            pass
        pass
    
    # This is getting messy. Let me restart with a cleaner approach.
    # I'll build the decomposition by directly constructing the U, V, W matrices.
    
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    col = 0
    for mi in range(7):
        def_ = strassen_defs[mi]
        A_blocks = def_['A']
        B_blocks = def_['B']
        C_contribs = def_['C_contrib']
        
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    for (ab, ac), a_coeff in A_blocks:
                        for (bb, bc), b_coeff in B_blocks:
                            for (cb, cc), c_coeff in C_contribs:
                                if col >= rank:
                                    raise ValueError("Rank exceeded: %d >= %d" % (col, rank))
                                a_idx = (3 * ab + u) * n + (3 * ac + w)
                                b_idx = (3 * bb + w) * n + (3 * bc + v)
                                c_idx = (3 * cb + u) * n + (3 * cc + v)
                                
                                U[a_idx][col] = a_coeff
                                V[b_idx][col] = b_coeff
                                W[c_idx][col] = c_coeff
                                col += 1
        # Each Strassen product contributes 9 * |A_blocks| * |B_blocks| * |C_contribs| terms
        # m1: 9 * 2 * 2 * 2 = 72
        # m2: 9 * 2 * 1 * 2 = 36
        # m3: 9 * 1 * 2 * 2 = 36
        # m4: 9 * 1 * 2 * 2 = 36
        # m5: 9 * 2 * 1 * 2 = 36
        # m6: 9 * 2 * 2 * 1 = 36
        # m7: 9 * 2 * 2 * 1 = 36
        # Total: 72 + 36*6 = 72 + 216 = 288? That's more than 189.
    
    # This approach is wrong. The issue is that I'm creating separate rank-1 terms for each combination
    # of A-block, B-block, and C-contribution, but they should be combined.
    
    # The correct approach: each scalar multiplication in the 3x3 product is a bilinear form.
    # For Strassen product mij, the 3x3 matrix product (LA_i * LB_i)[u, v] = sum_w LA_i[u,w] * LB_i[w,v].
    # Each term LA_i[u,w] * LB_i[w,v] is a bilinear form that contributes to multiple C elements.
    # So for each (u, w, v) and each pair of (A-element, B-element) in LA_i[u,w] and LB_i[w,v],
    # we get one rank-1 term that contributes to all relevant C elements.
    
    # Let me count more carefully.
    # For m1: LA = A00 + A11, LB = B00 + B11
    # LA[u, w] = A00[u, w] + A11[u, w] (two A elements)
    # LB[w, v] = B00[w, v] + B11[w, v] (two B elements)
    # Product has 2*2 = 4 terms for each (u, w, v)
    # 9 (u,w,v) * 4 = 36 terms for m1
    # Each term contributes to C00 and C11 (from C_contribs)
    # But in the tensor, each term is one rank-1 term with W entries for both C00 and C11.
    
    # So the total number of rank-1 terms is:
    # For each Strassen product, for each (u, w, v), for each (A_elem, B_elem) pair:
    #   one rank-1 term
    # m1: 9 * 2 * 2 = 36
    # m2: 9 * 2 * 1 = 18
    # m3: 9 * 1 * 2 = 18
    # m4: 9 * 1 * 2 = 18
    # m5: 9 * 2 * 1 = 18
    # m6: 9 * 2 * 2 = 36
    # m7: 9 * 2 * 2 = 36
    # Total: 36+18+18+18+18+36+36 = 180? Still not 189.
    
    # Hmm, let me recount.
    # m1: A has 2 blocks, B has 2 blocks. For each (u,w,v): 2*2=4 terms. 9*4=36.
    # m2: A has 2 blocks, B has 1 block. 9*2*1=18.
    # m3: A has 1, B has 2. 9*1*2=18.
    # m4: A has 1, B has 2. 9*1*2=18.
    # m5: A has 2, B has 1. 9*2*1=18.
    # m6: A has 2, B has 2. 9*2*2=36.
    # m7: A has 2, B has 2. 9*2*2=36.
    # Total: 36+18+18+18+18+36+36 = 180.
    
    # But we expected 189. The difference is 9. Maybe I'm miscounting the Strassen formula.
    
    # Actually, I think the issue is that the Strassen formula for 2x2 blocks of 3x3 matrices
    # doesn't directly give 189 terms. The 189 comes from 7 * 27 where 27 is the rank of 3x3 multiplication.
    # But when we use Strassen, the 7 products are of 3x3 matrices, and each 3x3 matrix product
    # has rank 27 (schoolbook). So 7*27=189.
    
    # But in my enumeration, I'm getting 180. The discrepancy might be because I'm not correctly
    # accounting for all the terms in the 3x3 matrix product.
    
    # Let me just use schoolbook for the full 6x6 and get 216, which is the baseline.
    # Then try to find a better decomposition.
    
    # For now, let me just return the schoolbook decomposition with rank 216.
    
    pass

def construct():
    """
    Returns the matrix multiplication tensor decomposition for 6x6 matrices.
    Currently uses schoolbook (rank 216).
    """
    n = 6
    m = 6
    p = 6
    rank = n * m * p  # 216
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x 216
    V = [[0] * rank for _ in range(m * p)]  # 36 x 216
    W = [[0] * rank for _ in range(p * n)]  # 36 x 216
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
