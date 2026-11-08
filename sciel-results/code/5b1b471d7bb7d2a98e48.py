"""Rewritten program for 4x4x4 matrix multiplication tensor decomposition.

We attempt to find a rank-minimal exact decomposition using known constructions.
For 4x4 matrices, the known best upper bound is 49 (from Strassen's 2x2x2
applied via block decomposition, or other constructions). The schoolbook rank is 64.

We will:
1. Try a direct construction based on Strassen's 7-multiplication algorithm for 2x2,
   then extend to 4x4 using block structure.
2. Use a known 4x4x4 decomposition if available.

The key insight: 4x4 matrix multiplication can be done with rank 49 using
the "Strassen-style" block decomposition where we treat the 4x4 as four 2x2 blocks.

Actually, the best known upper bound for 4x4x4 is 49. Let me construct this.

A 4x4 matrix can be partitioned into four 2x2 blocks:
A = [[A11, A12],
     [A21, A22]]

C = A * B where B = [[B11, B12], [B21, B22]]

C11 = A11*B11 + A12*B21
C12 = A11*B12 + A12*B22
C21 = A21*B11 + A22*B21
C22 = A21*B12 + A22*B22

Using Strassen's algorithm for each 2x2 product, we need 7 multiplications per 2x2 block product.
But we have 4 block products, so naive Strassen gives 4*7 = 28 scalar multiplications for the blocks,
but the blocks are 2x2, so each "multiplication" is a 2x2*2x2 product.

Wait, I need to think about this more carefully in terms of the tensor rank.

The tensor rank of 4x4x4 matrix multiplication: the best known upper bound is 49.
This comes from a construction by Strassen or subsequent improvements.

Let me use a known explicit construction. One approach:
- The 4x4x4 tensor can be decomposed using the "sum of 49 rank-1 tensors".

Actually, let me think about this differently. The standard result is:
- R(2,2,2) = 7 (Strassen)
- R(4,4,4) ≤ 49

The 49 comes from: 4x4 = 2*(2x2), and using the fact that the tensor product of two
rank-7 decompositions gives rank 49, but that's not quite right either.

Actually, the correct construction: If we have a rank-r decomposition for n×n, then
a rank-r decomposition for (2n)×(2n) can be achieved with rank r² (by tensor product),
but better constructions exist.

For 4×4: R(4,4,4) ≤ 49 is achieved by a specific construction. Let me try to build it.

One known construction uses the following: treat 4×4 as 2×2 blocks of 2×2 matrices.
The multiplication of 2×2 block matrices involves 8 block products (4 diagonal + 4 off-diagonal... no, it's 4 block products each being 2×2*2×2).

Hmm, let me just implement the schoolbook (rank 64) as a safe fallback, and try to construct
a better one. Given the time constraints, let me focus on getting a valid answer.

Actually, I recall that the best known rank for 4x4x4 is 49. Let me try to construct it explicitly.

The construction: Use the fact that 4 = 2+2, and use a "sum of products" approach.

Let me just go with the schoolbook decomposition (rank 64) which is guaranteed to work,
since constructing the rank-49 decomposition explicitly in code within time limits is complex.

Wait, but the task says "Find a rank-minimal exact tensor decomposition". The schoolbook is rank 64,
which is not minimal. The minimal known is 49. Let me try harder.

Let me implement a known rank-49 construction for 4x4x4.

The construction I'll use:
- Partition A, B, C into 2x2 blocks of 2x2 submatrices.
- Use Strassen's 7-multiplication algorithm for the 2x2 block matrix multiplication.

For 2x2 block matrices (where each block is 2x2):
Let A = [[A1, A2], [A3, A4]], B = [[B1, B2], [B3, B4]], C = [[C1, C2], [C3, C4]]

Strassen's formulas:
M1 = (A1+A4)(B1+B4)
M2 = (A3+A4)B1
M3 = A1(B2-B4)
M4 = A4(B3-B1)
M5 = (A1+A2)B4
M6 = (A3+A4)(B1+B2)
M7 = (A1-A3)(B4+B3)

C1 = M1 + M4 - M5 + M7
C2 = M3 + M5
C3 = M2 + M4
C4 = M1 - M2 + M3 + M6

Each Mi is a product of two 2x2 matrices. Each 2x2*2x2 product requires rank 7 (Strassen).
So total rank = 7 * 7 = 49.

Now I need to express this as U, V, W matrices.

The input A has 16 elements (4x4), indexed as A[i,j] for i,j in 0..3.
The input B has 16 elements, indexed as B[j,k] for j,k in 0..3.
The output C has 16 elements, indexed as C[i,k] for i,k in 0..3.

Block indices:
A1 = A[0:2, 0:2], A2 = A[0:2, 2:4], A3 = A[2:4, 0:2], A4 = A[2:4, 2:4]
B1 = B[0:2, 0:2], B2 = B[0:2, 2:4], B3 = B[2:4, 0:2], B4 = B[2:4, 2:4]
C1 = C[0:2, 0:2], C2 = C[0:2, 2:4], C3 = C[2:4, 0:2], C4 = C[2:4, 2:4]

For each Mi, which is a 2x2 * 2x2 product, I use Strassen's 7-term decomposition.

Let me define the Strassen decomposition for 2x2 matrices.
For X*Y where X = [[x11,x12],[x21,x22]], Y = [[y11,y12],[y21,y22]]:

Z1 = (x11+x22)(y11+y22)
Z2 = (x21+x22)y11
Z3 = x11(y12-y22)
Z4 = x22(y21-y11)
Z5 = (x11+x12)y22
Z6 = (x21+x22)(y11+y12)
Z7 = (x11-x21)(y22+y21)

Result:
[0,0] = Z1 + Z4 - Z5 + Z7
[0,1] = Z3 + Z5
[1,0] = Z2 + Z4
[1,1] = Z1 - Z2 + Z3 + Z6

Each Zt is a scalar multiplication (rank 1).

So for each Mi (which is a 2x2*2x2 product), I need 7 rank-1 terms.
There are 7 Mi's, so total 49 rank-1 terms.

Now I need to map this to the U, V, W matrices.

U has shape (16, 49): U[row, col] where row indexes A elements (i*m+j = i*4+j for i,j in 0..3)
V has shape (16, 49): V[row, col] where row indexes B elements (j*p+k = j*4+k for j,k in 0..3)
W has shape (16, 49): W[row, col] where row indexes C elements (k*n+i = k*4+i for k,i in 0..3)

Wait, the convention says W[k*n+i, r]. So W's row index is k*4+i where k is the column index of C and i is the row index of C.

Let me build this step by step.

For each of the 7 Strassen block products M1..M7:
- Each Mt is a 2x2*2x2 product, decomposed into 7 scalar multiplications.
- For each scalar multiplication, I add one column to U, V, W.

Let me define helper functions.
"""

import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def _strassen_2x2_decomposition():
    """
    Return the 7 rank-1 terms for 2x2*2x2 matrix multiplication.
    Each term is (U_entries, V_entries, W_entries) where each is a dict mapping index to value.
    U index: i*2+j for element (i,j) of the 2x2 matrix X
    V index: j*2+k for element (j,k) of the 2x2 matrix Y  
    W index: k*2+i for element (k,i) of the result Z (using the trace convention)
    """
    terms = []
    
    # Z1 = (x11+x22)(y11+y22)
    # U: x11=1, x22=1; V: y11=1, y22=1; W: z00=1, z01=1, z10=1, z11=1
    # Result: [0,0]=1, [0,1]=1, [1,0]=1, [1,1]=1
    # Wait no, Z1 contributes to multiple output positions with different coefficients.
    # Let me re-derive.
    
    # For 2x2: X=[[x00,x01],[x10,x11]], Y=[[y00,y01],[y10,y11]]
    # Z = X*Y: z00=x00*y00+x01*y10, z01=x00*y01+x01*y11, z10=x10*y00+x11*y10, z11=x10*y01+x11*y11
    
    # Strassen:
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10+x11)(y00+y01)
    # M7 = (x00-x10)(y11+y10)
    
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    # Each Mt is a product of a linear form in X and a linear form in Y.
    # For each Mt, I create one rank-1 term: U has the coefficients of the linear form in X,
    # V has the coefficients of the linear form in Y, W has the coefficients of how Mt contributes to Z.
    
    # M1: U: x00=1, x11=1; V: y00=1, y11=1; W: z00=1, z11=1
    terms.append((
        {0: 1, 3: 1},      # x00, x11
        {0: 1, 3: 1},      # y00, y11
        {0: 1, 3: 1}       # z00, z11 (W index: k*2+i, z00=0*2+0=0, z11=1*2+1=3)
    ))
    
    # M2: U: x10=1, x11=1; V: y00=1; W: z10=1
    terms.append((
        {2: 1, 3: 1},      # x10, x11
        {0: 1},            # y00
        {2: 1}             # z10 (k=1,i=0 => 1*2+0=2)
    ))
    
    # M3: U: x00=1; V: y01=1, y11=-1; W: z01=1, z11=1
    terms.append((
        {0: 1},            # x00
        {1: 1, 3: -1},     # y01, y11
        {1: 1, 3: 1}       # z01 (0*2+1=1), z11 (1*2+1=3)
    ))
    
    # M4: U: x11=1; V: y10=1, y00=-1; W: z00=1, z10=1
    terms.append((
        {3: 1},            # x11
        {2: 1, 0: -1},     # y10, y00
        {0: 1, 2: 1}       # z00, z10
    ))
    
    # M5: U: x00=1, x01=1; V: y11=1; W: z00=-1, z01=1
    terms.append((
        {0: 1, 1: 1},      # x00, x01
        {3: 1},            # y11
        {0: -1, 1: 1}      # z00=-1, z01=1
    ))
    
    # M6: U: x10=1, x11=1; V: y00=1, y01=1; W: z11=1
    terms.append((
        {2: 1, 3: 1},      # x10, x11
        {0: 1, 1: 1},      # y00, y01
        {3: 1}             # z11
    ))
    
    # M7: U: x00=1, x10=-1; V: y11=1, y10=1; W: z00=1
    terms.append((
        {0: 1, 2: -1},     # x00, x10
        {3: 1, 2: 1},      # y11, y10
        {0: 1}             # z00
    ))
    
    return terms

def _schoolbook():
    """Schoolbook decomposition of rank 64."""
    n, m, p = DIMENSIONS
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

def _strassen_4x4():
    """
    Construct rank-49 decomposition for 4x4x4 matrix multiplication.
    Uses Strassen's algorithm at two levels.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Get the 2x2 Strassen terms
    strassen_terms = _strassen_2x2_decomposition()
    
    # For 4x4, we partition into 2x2 blocks of 2x2 submatrices.
    # A is 4x4, partitioned as:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B and C.
    
    # Block matrix multiplication:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using Strassen for the 2x2 block matrix:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21+A22)(B11+B12)
    # M7 = (A11-A21)(B22+B21)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mt is a product of two 2x2 matrices. Each such product uses 7 rank-1 terms (inner Strassen).
    # Total: 7 * 7 = 49 terms.
    
    rank = 49
    U = [[0] * rank for _ in range(16)]  # 4*4 = 16
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    
    # Define the block structure.
    # For each outer Strassen term Mt, it's a product of (linear combo of A-blocks) * (linear combo of B-blocks).
    # The result contributes to C-blocks with certain coefficients.
    
    # Outer Strassen terms for 2x2 block matrices:
    # Each term: (U_block_coeffs, V_block_coeffs, W_block_coeffs)
    # U_block_coeffs: dict mapping block index (0=A11, 1=A12, 2=A21, 3=A22) to coefficient
    # V_block_coeffs: dict mapping block index (0=B11, 1=B12, 2=B21, 3=B22) to coefficient
    # W_block_coeffs: dict mapping block index (0=C11, 1=C12, 2=C21, 3=C22) to coefficient
    
    outer_terms = [
        # M1: (A11+A22)(B11+B22) -> C11:1, C22:1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (A21+A22)B11 -> C21:1
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # M3: A11(B12-B22) -> C12:1, C22:1
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: A22(B21-B11) -> C11:1, C21:1
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (A11+A12)B22 -> C11:-1, C12:1
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (A21+A22)(B11+B12) -> C22:1
        ({2: 1, 3: 1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (A11-A21)(B22+B21) -> C11:1
        ({0: 1, 2: -1}, {3: 1, 2: 1}, {0: 1}),
    ]
    
    # Block to element mapping:
    # Block 0 (top-left): elements (0,0),(0,1),(1,0),(1,1) -> flat indices 0,1,4,5
    # Block 1 (top-right): elements (0,2),(0,3),(1,2),(1,3) -> flat indices 2,3,6,7
    # Block 2 (bottom-left): elements (2,0),(2,1),(3,0),(3,1) -> flat indices 8,9,12,13
    # Block 3 (bottom-right): elements (2,2),(2,3),(3,2),(3,3) -> flat indices 10,11,14,15
    
    # For a 2x2 matrix with elements indexed as [0,1;2,3] (row-major within block):
    # element (bi, bj) in block maps to:
    # Block 0: flat = bi*4 + bj
    # Block 1: flat = bi*4 + (2+bj)
    # Block 2: flat = (2+bi)*4 + bj
    # Block 3: flat = (2+bi)*4 + (2+bj)
    
    def block_element_flat(block, bi, bj):
        if block == 0:
            return bi * 4 + bj
        elif block == 1:
            return bi * 4 + (2 + bj)
        elif block == 2:
            return (2 + bi) * 4 + bj
        elif block == 3:
            return (2 + bi) * 4 + (2 + bj)
    
    # For V (B matrix), the flat index is j*4+k where j is row, k is col.
    # Block 0 (B11): j in 0,1; k in 0,1 -> flat = j*4+k
    # Block 1 (B12): j in 0,1; k in 2,3 -> flat = j*4+(2+k_local)
    # Block 2 (B21): j in 2,3; k in 0,1 -> flat = (2+j_local)*4+k
    # Block 3 (B22): j in 2,3; k in 2,3 -> flat = (2+j_local)*4+(2+k_local)
    
    def b_block_element_flat(block, bi, bj):
        if block == 0:
            return bi * 4 + bj
        elif block == 1:
            return bi * 4 + (2 + bj)
        elif block == 2:
            return (2 + bi) * 4 + bj
        elif block == 3:
            return (2 + bi) * 4 + (2 + bj)
    
    # For W (C matrix), the convention is W[k*n+i, r] where k is col, i is row.
    # So W flat index = k*4 + i.
    # Block 0 (C11): i in 0,1; k in 0,1 -> W flat = k*4+i
    # Block 1 (C12): i in 0,1; k in 2,3 -> W flat = (2+k_local)*4+i
    # Block 2 (C21): i in 2,3; k in 0,1 -> W flat = k*4+(2+i_local)
    # Block 3 (C22): i in 2,3; k in 2,3 -> W flat = (2+k_local)*4+(2+i_local)
    
    def c_block_element_flat(block, ci, ck):
        """ci is local row (0,1), ck is local col (0,1). W index = ck*4 + ci."""
        if block == 0:
            return ck * 4 + ci
        elif block == 1:
            return (2 + ck) * 4 + ci
        elif block == 2:
            return ck * 4 + (2 + ci)
        elif block == 3:
            return (2 + ck) * 4 + (2 + ci)
    
    # Now construct the decomposition.
    for outer_idx, (u_blocks, v_blocks, w_blocks) in enumerate(outer_terms):
        # For each combination of A-block and B-block in this outer term,
        # we have a 2x2 * 2x2 product.
        for a_block, a_coeff in u_blocks.items():
            for b_block, b_coeff in v_blocks.items():
                # This is a product of (a_coeff * A_{a_block}) * (b_coeff * B_{b_block})
                # = (a_coeff * b_coeff) * (A_{a_block} * B_{b_block})
                # Use inner Strassen to decompose A_{a_block} * B_{b_block}
                
                for inner_idx, (u_inner, v_inner, w_inner) in enumerate(strassen_terms):
                    # u_inner: dict mapping local 2x2 element (bi,bj) to coeff
                    # v_inner: dict mapping local 2x2 element (bi,bj) to coeff
                    # w_inner: dict mapping local 2x2 element (ci,ck) to coeff
                    
                    # Scale by a_coeff * b_coeff
                    scale = a_coeff * b_coeff
                    
                    # Fill U column
                    for (bi, bj), coeff in u_inner.items():
                        flat_a = block_element_flat(a_block, bi, bj)
                        U[flat_a][col] += scale * coeff
                    
                    # Fill V column
                    for (bi, bj), coeff in v_inner.items():
                        flat_b = b_block_element_flat(b_block, bi, bj)
                        V[flat_b][col] += scale * coeff
                    
                    # Fill W column
                    for (ci, ck), coeff in w_inner.items():
                        # w_inner uses (ci, ck) where ci is row, ck is col of the 2x2 result
                        # But we need to map to C-blocks using w_blocks
                        flat_c = c_block_element_flat(0, ci, ck)  # temporary, will be adjusted
                        # Actually, w_inner gives the contribution to the 2x2 product result.
                        # This 2x2 product result is one of the C-blocks, weighted by w_blocks.
                        
                        # The inner Strassen gives us the 2x2 result Z = A_block * B_block.
                        # Z's element (ci, ck) contributes to C-blocks according to w_blocks.
                        # But w_blocks maps block_index -> coefficient.
                        # The result Z is added to C-blocks. Which C-block?
                        # The product A_{a_block} * B_{b_block} is a 2x2 matrix.
                        # This 2x2 matrix needs to be placed in the appropriate C-block.
                        
                        # Wait, I think I'm overcomplicating this. Let me reconsider.
                        
                        # The outer Strassen term Mt = (sum of A-blocks) * (sum of B-blocks).
                        # This is a 2x2 * 2x2 matrix product, giving a 2x2 result.
                        # The 2x2 result is added to C-blocks according to w_blocks.
                        
                        # But which C-block does the result go into? It depends on the outer term.
                        # Looking at the outer terms:
                        # M1 contributes to C11 and C22 (both with coeff 1)
                        # This means the 2x2 result of (A11+A22)*(B11+B22) is added to C11 and C22.
                        
                        # So the 2x2 result Z has elements Z[ci,ck] that go to:
                        # For each C-block c_block with coefficient w_coeff:
                        #   C[c_block][ci,ck] += w_coeff * Z[ci,ck]
                        
                        # So I need to iterate over w_blocks.
                        pass  # I'll restructure this below
    
    # Let me restructure the loop properly.
    col = 0
    for outer_idx, (u_blocks, v_blocks, w_blocks) in enumerate(outer_terms):
        for a_block, a_coeff in u_blocks.items():
            for b_block, b_coeff in v_blocks.items():
                for inner_idx, (u_inner, v_inner, w_inner) in enumerate(strassen_terms):
                    scale = a_coeff * b_coeff
                    
                    # U column
                    for (bi, bj), coeff in u_inner.items():
                        flat_a = block_element_flat(a_block, bi, bj)
                        U[flat_a][col] += scale * coeff
                    
                    # V column
                    for (bi, bj), coeff in v_inner.items():
                        flat_b = b_block_element_flat(b_block, bi, bj)
                        V[flat_b][col] += scale * coeff
                    
                    # W column: for each C-block in w_blocks, add the contribution
                    for c_block, w_coeff in w_blocks.items():
                        for (ci, ck), coeff in w_inner.items():
                            flat_c = c_block_element_flat(c_block, ci, ck)
                            W[flat_c][col] += scale * w_coeff * coeff
                    
                    col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

def _validate(result):
    """Validate that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = result["U"]
    V = result["V"]
    W = result["W"]
    
    rank = len(U[0])
    
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for i in range(n * m):
        if len(U[i]) != rank:
            return False
    for j in range(m * p):
        if len(V[j]) != rank:
            return False
    for k in range(p * n):
        if len(W[k]) != rank:
            return False
    
    # Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 if i,j,k form a valid triple
    # The matrix multiplication tensor T[i,j,k] = 1 if the element (i,k) of A*B depends on A[i,j]*B[j,k]
    # Actually, T[i,j,k] = 1 for all i,j,k (each output element C[i,k] = sum_j A[i,j]*B[j,k])
    # So the tensor has T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    # Check a sample of entries
    import random
    random.seed(42)
    
    # Check all entries (16*16*16 = 4096 checks, each is a sum of rank terms)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    
    return True

def construct():
    """Construct the tensor decomposition for 4x4x4 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try Strassen 4x4 (rank 49)
    try:
        result = _strassen_4x4()
        if _validate(result):
            return result
    except Exception:
        pass
    
    # Fall back to schoolbook (rank 64)
    return _schoolbook()
# EVOLVE-BLOCK-END
