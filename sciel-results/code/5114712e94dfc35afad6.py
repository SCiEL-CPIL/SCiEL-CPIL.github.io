"""Matrix multiplication 6x6x6: Strassen-based rank-189 decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-189 exact tensor decomposition for 6x6 matrix multiplication
    using Strassen's algorithm applied to 3x3 blocks.
    
    The 6x6 matrices are partitioned into four 3x3 blocks:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    C = A*B = [[C11, C12], [C21, C22]]
    
    Strassen computes the 4 blocks of C using 7 multiplications of 3x3 matrices.
    Each 3x3 matrix multiplication has rank 27 (schoolbook), so total rank = 7*27 = 189.
    
    Index conventions:
    - U has shape (n*m, rank) = (36, 189): U[i*m+j, c]
    - V has shape (m*p, rank) = (36, 189): V[j*p+k, c]  
    - W has shape (p*n, rank) = (36, 189): W[k*n+i, c]
    
    For 6x6: n=m=p=6
    Block indices: block row 0 = rows 0-2, block row 1 = rows 3-5
    Within a 3x3 block, local indices 0,1,2 map to global offsets 0,1,2 or 3,4,5.
    """
    n, m, p = DIMENSIONS  # all 6
    
    # Strassen's 7 products for 2x2 block matrices with 3x3 blocks:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mx is a 3x3 matrix product. For a 3x3 product P = X*Y,
    # we decompose using schoolbook: 9 terms, each a rank-1 tensor.
    # For term (a,b,c) in the 3x3 product: X[local_a, local_b] * Y[local_b, local_c]
    # contributes to P[local_a, local_c].
    
    # We'll build U, V, W as lists of lists (or numpy arrays).
    rank = 189
    U = [[0.0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    
    def add_term(u_row, v_row, w_row, coeff):
        """Add a rank-1 term with given coefficients."""
        nonlocal col
        U[u_row][col] = coeff
        V[v_row][col] = coeff
        W[w_row][col] = coeff
        col += 1
    
    # For each of the 7 Strassen products, add 9 schoolbook terms for the 3x3 multiply.
    # 
    # Strassen product structure:
    # M1 = (A11+A22)*(B11+B22): sum over a,b,c in {0,1,2} of (A11[a,b]+A22[a,b])*(B11[b,c]+B22[b,c])
    #   This expands to 4 sub-products, each 3x3. But we can't just do 4*9=36 terms per Strassen product.
    #   Instead, we note that (X+Y)*(Z+W) = X*Z + X*W + Y*Z + Y*W, each being a 3x3 product.
    #   So each Strassen product that's a sum of two matrices times a sum of two matrices
    #   expands to 4 schoolbook 3x3 products = 36 rank-1 terms.
    #
    # Let me reconsider. The 7 Strassen products are:
    # M1 = (A11+A22)*(B11+B22) -> 4 sub-multiplications
    # M2 = (A21+A22)*B11 -> 2 sub-multiplications  
    # M3 = A11*(B12-B22) -> 2 sub-multiplications
    # M4 = A22*(B21-B11) -> 2 sub-multiplications
    # M5 = (A11+A12)*B22 -> 2 sub-multiplications
    # M6 = (A21-A11)*(B11+B12) -> 4 sub-multiplications
    # M7 = (A12-A22)*(B21+B22) -> 4 sub-multiplications
    #
    # Total sub-multiplications: 4+2+2+2+2+4+4 = 20
    # Each sub-multiplication is a 3x3 product = 9 rank-1 terms
    # Total rank = 20 * 9 = 180? That's not 189.
    
    # Wait, I need to be more careful. Let me recount.
    # Actually, the standard Strassen for 2x2 block matrices with scalar blocks gives rank 7.
    # When blocks are 3x3 matrices, each "multiplication" of two 3x3 matrices has rank 27.
    # So 7 * 27 = 189.
    #
    # The key insight: each Strassen product Mx is itself a 3x3 matrix (the result of multiplying
    # two 3x3 matrices). So Mx = X * Y where X and Y are 3x3 matrices. The decomposition of
    # this 3x3 matrix product has rank 27 (9 terms for the outer structure... no).
    #
    # Actually, a 3x3 matrix product C = A*B where A, B are 3x3 has:
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # This is a sum of 9 rank-1 terms (one for each k): A[i,k]*B[k,j]
    # So rank of 3x3 matmul tensor = 9, not 27.
    #
    # Wait no. The tensor for multiplying two 3x3 matrices:
    # T_{i,j,k} where we compute C[i,j] = sum_k A[i,k] * B[k,j]
    # The standard decomposition has 9 terms (one per k value).
    # So rank(3x3 matmul) = 9.
    #
    # Then Strassen: 7 products, each of rank 9, total rank = 63?
    # That doesn't seem right either. Let me think again.
    
    # OK let me be very precise. The matrix multiplication tensor for n x n matrices
    # T_{i,j,k} = 1 if we need A[i,k]*B[k,j] to contribute to C[i,j].
    # The standard (schoolbook) decomposition uses n terms: for each k, the term
    # (A[i,k]) * (B[k,j]) * (C[i,j]) with coefficient 1.
    # So rank(n x n matmul) <= n for the "simple" case? No, that's wrong.
    
    # Let me reconsider. For 2x2 matmul: C[i,j] = sum_k A[i,k]*B[k,j], k=0,1
    # Term k=0: A[i,0]*B[0,j] contributes to C[i,j] for all i,j
    # Term k=1: A[i,1]*B[1,j] contributes to C[i,j] for all i,j
    # So we have 2 terms, each being a rank-1 tensor in the 3-way tensor space.
    # Rank(2x2 matmul) = 2? But Strassen says rank 7...
    
    # I'm confusing myself. Let me use the standard formulation.
    # The matrix multiplication tensor T has entries T_{i,j,k} where:
    # - i indexes rows of A (and C)
    # - j indexes columns of B (and C)  
    # - k indexes columns of A (and rows of B)
    # T_{i,j,k} = 1 (it's a 0-1 tensor)
    # 
    # A decomposition of rank r means we can write:
    # T_{i,j,k} = sum_{c=1}^{r} U_{i,k}^{(c)} * V_{k,j}^{(c)} * W_{i,j}^{(c)}
    # (or some equivalent factorization)
    #
    # For schoolbook: for each k, define U^{(k)}_{i,k'} = delta_{k,k'}, V^{(k)}_{k',j} = delta_{k,k'}, W^{(k)}_{i,j} = 1
    # Then sum_k U^{(k)}_{i,k'} V^{(k)}_{k',j} W^{(k)}_{i,j} = sum_k delta_{k,k'} delta_{k,k'} = delta_{k,k'}
    # Hmm, that gives T_{i,j,k} = 1 for all i,j,k which is wrong.
    
    # Let me look at this differently. The standard formulation:
    # C = A * B, C[i,j] = sum_k A[i,k] * B[k,j]
    # As a bilinear map: (A, B) -> C
    # The tensor T in U*V*W* (where U* is dual of A-space, etc.)
    # T = sum_{i,j,k} e_i^* ⊗ e_j^* ⊗ e_k ⊗ (e_i ⊗ e_j)  ... this is getting complicated.
    
    # Let me just use the practical formulation from the seed program.
    # The seed uses: for each (i,j,k), a term where:
    # U[i*m+j, c] = 1  (this indexes A[i,j] where i is row, j is col of A)
    # V[j*p+k, c] = 1  (this indexes B[j,k] where j is row, k is col of B)
    # W[k*n+i, c] = 1  (this indexes C[k,i]... wait that's odd)
    
    # Looking at the seed: W[k*n+i, c]. With n=6, this is W[k*6+i, c].
    # The output C should be C[row, col]. The convention says W[k*n+i, r].
    # So the output index is (k, i) interpreted as... let me re-read.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has p*n rows, indexed by k*n+i where k is from B's column (=C's column?) 
    # and i is from A's row (=C's row?).
    # Actually for C = A*B: C[i,k] = sum_j A[i,j]*B[j,k]
    # So C is indexed by (i, k) where i is row (from A) and k is column (from B).
    # W[k*n+i] means the output is stored with k as the "row" index and i as the "col" index?
    # Or it's W[(k,i)] = W[k*n + i].
    
    # From the seed: U[i*m+j] indexes A[i,j], V[j*p+k] indexes B[j,k], W[k*n+i] indexes C[?,?].
    # The product term is A[i,j]*B[j,k] which contributes to C[i,k].
    # So W[k*n+i] must correspond to C[i,k], meaning the storage is C[col, row] or the
    # index is (k, i) representing column k, row i of C.
    
    # OK so the convention is: output C is stored as C_transposed or with (col, row) ordering.
    # W[k*n+i, c] corresponds to C[i, k] (row i, col k of the result).
    
    # Now for Strassen with 3x3 blocks:
    # A is 6x6, partitioned into 3x3 blocks:
    # A[0:3, 0:3] = A11, A[0:3, 3:6] = A12
    # A[3:6, 0:3] = A21, A[3:6, 3:6] = A22
    # Similarly for B.
    # C[0:3, 0:3] = C11, C[0:3, 3:6] = C12
    # C[3:6, 0:3] = C21, C[3:6, 3:6] = C22
    
    # Strassen products (each is a 3x3 matrix product):
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B21-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mx is a 3x3 matrix = product of two 3x3 matrices.
    # For a 3x3 product P = X*Y where X,Y are 3x3:
    # P[a,c] = sum_b X[a,b]*Y[b,c]
    # Schoolbook: 9 terms (one per b).
    
    # So each Mx contributes 9 rank-1 terms to the overall decomposition.
    # 7 * 9 = 63 total rank-1 terms? But we need rank 189...
    
    # I think I'm still confused. Let me think about what "rank" means here.
    # 
    # The full 6x6 matmul tensor has 6*6*6 = 216 entries (i,j,k triples).
    # A rank-1 term in the tensor decomposition corresponds to one "multiplication"
    # in the algorithm, i.e., one scalar product A[i,j]*B[j,k] contributing to C[i,k].
    #
    # Schoolbook: 6*6*6 = 216 such terms (one for each (i,j,k)).
    # 
    # Strassen reduces the number of "multiplications" (scalar products) from 216 to 189.
    # Each scalar product is a rank-1 term.
    #
    # For the 3x3 block version: each 3x3 matmul has 3*3*3 = 27 scalar products.
    # 7 Strassen products * 27 = 189 scalar products. Yes! That's the rank.
    
    # So for each Strassen product Mx = X*Y (3x3 matrices), we add 27 rank-1 terms
    # corresponding to all (a,b,c) in {0,1,2}^3: X[a,b]*Y[b,c] contributes to P[a,c].
    
    # Now I need to map these to the global indices.
    
    # Helper: given a Strassen product definition, generate the 27 terms.
    
    # Strassen product definitions:
    # Each product is of the form (sum of A-blocks with signs) * (sum of B-blocks with signs)
    # where each block is a 3x3 submatrix.
    
    # Let me define the 7 products as lists of (A_block, sign_A) and (B_block, sign_B).
    # A_block is one of 'A11','A12','A21','A22'
    # B_block is one of 'B11','B12','B21','B22'
    
    # Product definitions:
    # M1: (A11+A22) * (B11+B22)
    # M2: (A21+A22) * B11
    # M3: A11 * (B12-B22)
    # M4: A22 * (B21-B11)
    # M5: (A11+A12) * B22
    # M6: (A21-A11) * (B11+B12)
    # M7: (A12-A22) * (B21+B22)
    
    # For each product, expand into individual block multiplications:
    # M1: A11*B11 + A11*B22 + A22*B11 + A22*B22
    # M2: A21*B11 + A22*B11
    # M3: A11*B12 - A11*B22
    # M4: A22*B21 - A22*B11
    # M5: A11*B22 + A12*B22
    # M6: A21*B11 + A21*B12 - A11*B11 - A11*B12
    # M7: A12*B21 + A12*B22 - A22*B21 - A22*B22
    
    # Total block multiplications: 4+2+2+2+2+4+4 = 20
    # Each block multiplication is a 3x3 matmul = 27 scalar products
    # Total: 20 * 27 = 540? That's way more than 189.
    
    # I see the issue. Strassen's algorithm works at the level of matrix multiplications,
    # not scalar multiplications. The "rank" in tensor decomposition refers to the number
    # of scalar multiplications.
    
    # When we say Strassen has rank 7 for 2x2 block matrices, it means 7 matrix multiplications.
    # If each block is 3x3, each matrix multiplication is a 3x3 matmul with rank 27.
    # So total rank = 7 * 27 = 189.
    
    # But the expansion into individual block multiplications gives 20 block multiplications,
    # not 7. The key is that in Strassen, the 7 products are computed as single matrix
    # multiplications, not as sums of smaller ones.
    
    # So for the tensor decomposition, each Strassen product Mx = X*Y where X and Y are
    # 3x3 matrices (which are linear combinations of the original blocks) is decomposed
    # as a 3x3 matmul with rank 27.
    
    # The challenge: X and Y are not single blocks but sums/differences of blocks.
    # For example, M1 = (A11+A22)*(B11+B22).
    # X = A11+A22 (a 3x3 matrix that is the sum of two 3x3 blocks)
    # Y = B11+B22
    # 
    # The 27 terms for this product are:
    # For a,b,c in {0,1,2}: X[a,b]*Y[b,c] contributes to P[a,c]
    # X[a,b] = A11[a,b] + A22[a,b]
    # Y[b,c] = B11[b,c] + B22[b,c]
    # 
    # So the term is (A11[a,b]+A22[a,b])*(B11[b,c]+B22[b,c])
    # This is a single rank-1 term in the tensor where:
    # U part: A11[a,b] + A22[a,b] (coefficient 1 for each)
    # V part: B11[b,c] + B22[b,c] (coefficient 1 for each)
    # W part: contributes to P[a,c] with coefficient 1
    
    # So for each (a,b,c), we have ONE rank-1 term where:
    # U has coefficient 1 at positions corresponding to A11[a,b] and A22[a,b]
    # V has coefficient 1 at positions corresponding to B11[b,c] and B22[b,c]
    # W has coefficient 1 at position corresponding to P[a,c]
    
    # This gives 9 terms per Strassen product, 7*9 = 63 total. Still not 189.
    
    # I think the issue is that when X = A11+A22, the term X[a,b] appears in U,
    # but U[a11_row, col] and U[a22_row, col] both get coefficient 1.
    # This is still one rank-1 term in the tensor (one column in U, V, W).
    
    # So rank should be 63? But the known result is 189.
    
    # Let me look up: the rank of 6x6 matrix multiplication.
    # Known: rank(6x6 matmul) <= 189 (via Strassen on 3x3 blocks)
    # Actually, I recall that the best known upper bound for 6x6 matmul is around 189-200.
    
    # I think my confusion is about what "rank" means in this context.
    # Let me just follow the pattern: 7 Strassen products, each a 3x3 matmul.
    # A 3x3 matmul tensor has rank 27 (not 9). Why?
    
    # Ah, I see. For a 3x3 matmul C = A*B:
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # The tensor T_{i,j,k} where T_{i,j,k} = 1 for the (i,j,k) triple.
    # Schoolbook decomposition: for each k (0,1,2), for each i (0,1,2), for each j (0,1,2):
    # Term: A[i,k]*B[k,j] contributes to C[i,j]
    # That's 3*3*3 = 27 terms.
    
    # So rank(3x3 matmul) = 27 with schoolbook.
    # And Strassen for 2x2 blocks of 3x3 matrices: 7 * 27 = 189.
    
    # Now, for M1 = (A11+A22)*(B11+B22):
    # This is a 3x3 matmul where the "A" matrix is X = A11+A22 and "B" matrix is Y = B11+B22.
    # The 27 terms are for (i,k,j) in {0,1,2}^3: X[i,k]*Y[k,j] -> P[i,j]
    # 
    # But X[i,k] = A11[i,k] + A22[i,k], so in U, we need coefficients at both
    # A11[i,k] and A22[i,k] positions.
    # Similarly Y[k,j] = B11[k,j] + B22[k,j].
    
    # Each such term is one rank-1 term in the overall 6x6 matmul tensor.
    # So M1 contributes 27 rank-1 terms.
    # 7 products * 27 = 189. 
    
    # Now let me implement this properly.
    
    # Block index mapping:
    # A11: rows 0-2, cols 0-2 of A. Global index in U: i*m+j where i in [0,2], j in [0,2] -> i*6+j
    # A12: rows 0-2, cols 3-5 of A. Global index: i*6+(j+3) where i in [0,2], j in [0,2]
    # A21: rows 3-5, cols 0-2 of A. Global index: (i+3)*6+j
    # A22: rows 3-5, cols 3-5 of A. Global index: (i+3)*6+(j+3)
    
    # Similarly for B:
    # B11: rows 0-2, cols 0-2. V index: j*p+k = j*6+k, j in [0,2], k in [0,2]
    # B12: rows 0-2, cols 3-5. V index: j*6+(k+3)
    # B21: rows 3-5, cols 0-2. V index: (j+3)*6+k
    # B22: rows 3-5, cols 3-5. V index: (j+3)*6+(k+3)
    
    # For output C (stored as W[k*n+i] = W[k*6+i]):
    # C11: C[0:3, 0:3]. C[i,k] with i in [0,2], k in [0,2]. W index: k*6+i
    # C12: C[0:3, 3:6]. C[i,k] with i in [0,2], k in [3,5]. W index: k*6+i
    # C21: C[3:6, 0:3]. C[i,k] with i in [3,5], k in [0,2]. W index: k*6+i
    # C22: C[3:6, 3:6]. C[i,k] with i in [3,5], k in [3,5]. W index: k*6+i
    
    # Strassen product definitions:
    # Each product is a list of (A_block, sign) for the left matrix and (B_block, sign) for right.
    # The product is (sum of A terms) * (sum of B terms).
    
    # For each product, for each (i,k,j) in {0,1,2}^3:
    #   Left matrix element: sum of A_block[i,k] with signs
    #   Right matrix element: sum of B_block[k,j] with signs
    #   Output: contributes to C_block[i,j] with coefficient 1
    
    # The rank-1 term has:
    #   U: coefficient 1 at each A_block[i,k] position (with the sign)
    #   V: coefficient 1 at each B_block[k,j] position (with the sign)
    #   W: coefficient 1 at C_block[i,j] position (with the sign from C formula)
    
    # Wait, I need to be careful with signs. Let me define each Strassen product
    # as a list of (A_block, sign_A) pairs and (B_block, sign_B) pairs.
    
    # M1: left = [(A11, +1), (A22, +1)], right = [(B11, +1), (B22, +1)]
    # M2: left = [(A21, +1), (A22, +1)], right = [(B11, +1)]
    # M3: left = [(A11, +1)], right = [(B12, +1), (B22, -1)]
    # M4: left = [(A22, +1)], right = [(B21, +1), (B11, -1)]
    # M5: left = [(A11, +1), (A12, +1)], right = [(B22, +1)]
    # M6: left = [(A21, +1), (A11, -1)], right = [(B11, +1), (B12, +1)]
    # M7: left = [(A12, +1), (A22, -1)], right = [(B21, +1), (B22, +1)]
    
    # C formulas:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen product Mx and each (i,k,j), the term contributes to
    # multiple C blocks with different signs.
    
    # Actually, the contribution of Mx to C is:
    # Mx[i,j] appears in C blocks with certain coefficients.
    # For M1: C11 += M1, C22 += M1
    # For M2: C21 += M2, C22 -= M2
    # For M3: C12 += M3, C22 += M3
    # For M4: C11 += M4, C21 += M4
    # For M5: C11 -= M5, C12 += M5
    # For M6: C22 += M6
    # For M7: C11 += M7
    
    # So for each (i,k,j) in Mx, the output P[i,j] contributes to:
    # Various C blocks with various signs.
    
    # Let me define the contribution of each Mx to C blocks:
    # M1: {(C11, +1), (C22, +1)}
    # M2: {(C21, +1), (C22, -1)}
    # M3: {(C12, +1), (C22, +1)}
    # M4: {(C11, +1), (C21, +1)}
    # M5: {(C11, -1), (C12, +1)}
    # M6: {(C22, +1)}
    # M7: {(C11, +1)}
    
    # Now I can build the decomposition.
    
    # Block to global index mapping:
    def a_block_idx(block, i, k):
        """Map A block (i,k) to global U index i*m+j."""
        if block == 'A11':
            return i * m + k
        elif block == 'A12':
            return i * m + (k + 3)
        elif block == 'A21':
            return (i + 3) * m + k
        elif block == 'A22':
            return (i + 3) * m + (k + 3)
    
    def b_block_idx(block, k, j):
        """Map B block (k,j) to global V index j*p+k... wait."""
        # V index is j*p+k where j is row of B, k is col of B.
        # In our notation, B[k,j] where k is row, j is col.
        # So V index = k*p + j = k*6 + j.
        if block == 'B11':
            return k * p + j
        elif block == 'B12':
            return k * p + (j + 3)
        elif block == 'B21':
            return (k + 3) * p + j
        elif block == 'B22':
            return (k + 3) * p + (j + 3)
    
    def c_block_idx(block, i, j):
        """Map C block (i,j) to global W index k*n+i where i is row, j is col of C."""
        # W index = k*n + i where k is col of C, i is row of C.
        # C[i,j] -> W[j*n + i] = W[j*6 + i]
        if block == 'C11':
            return j * n + i
        elif block == 'C12':
            return (j + 3) * n + i
        elif block == 'C21':
            return j * n + (i + 3)
        elif block == 'C22':
            return (j + 3) * n + (i + 3)
    
    # Strassen products definition
    strassen_products = [
        # (name, left_blocks, right_blocks, c_contributions)
        ('M1', [('A11', 1), ('A22', 1)], [('B11', 1), ('B22', 1)],
         [('C11', 1), ('C22', 1)]),
        ('M2', [('A21', 1), ('A22', 1)], [('B11', 1)],
         [('C21', 1), ('C22', -1)]),
        ('M3', [('A11', 1)], [('B12', 1), ('B22', -1)],
         [('C12', 1), ('C22', 1)]),
        ('M4', [('A22', 1)], [('B21', 1), ('B11', -1)],
         [('C11', 1), ('C21', 1)]),
        ('M5', [('A11', 1), ('A12', 1)], [('B22', 1)],
         [('C11', -1), ('C12', 1)]),
        ('M6', [('A21', 1), ('A11', -1)], [('B11', 1), ('B12', 1)],
         [('C22', 1)]),
        ('M7', [('A12', 1), ('A22', -1)], [('B21', 1), ('B22', 1)],
         [('C11', 1)]),
    ]
    
    col = 0
    for name, left_blocks, right_blocks, c_contribs in strassen_products:
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    # This is one rank-1 term
                    # U: for each (block, sign) in left_blocks, add sign at a_block_idx(block, i, k)
                    # V: for each (block, sign) in right_blocks, add sign at b_block_idx(block, k, j)
                    # W: for each (cblock, sign) in c_contribs, add sign at c_block_idx(cblock, i, j)
                    
                    for (ablock, asgn) in left_blocks:
                        u_idx = a_block_idx(ablock, i, k)
                        U[u_idx][col] += asgn
                    
                    for (bblock, bsgn) in right_blocks:
                        v_idx = b_block_idx(bblock, k, j)
                        V[v_idx][col] += bsgn
                    
                    for (cblock, csgn) in c_contribs:
                        w_idx = c_block_idx(cblock, i, j)
                        W[w_idx][col] += csgn
                    
                    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
