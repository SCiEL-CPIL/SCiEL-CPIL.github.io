"""Rewritten program for 4x4 matrix multiplication tensor decomposition.
Uses the explicit rank-49 decomposition derived from Strassen's algorithm applied
to 2x2 blocks, then subdivided into 1x1 scalars. This is the known optimal rank
for 4x4 matrix multiplication over a field of characteristic 0 (or at least the
best known construction)."""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-49 decomposition for 4x4 matrix multiplication.
    
    The idea: A 4x4 matrix multiplication can be viewed as multiplying 2x2 blocks
    where each block entry is a 2x2 matrix. Using Strassen's algorithm on the 2x2
    block structure gives 7 multiplications of 2x2 matrices. Each 2x2 multiplication
    can be done with 7 rank-1 tensors (again Strassen). So total rank = 7 * 7 = 49.
    
    However, we need to be careful about the exact indexing. Let's think more carefully.
    
    Actually, the standard approach: 
    - A 4x4 matrix A has entries A[i,j] for i,j in 0..3
    - B is 4x4 with entries B[j,k] for j,k in 0..3
    - C = A*B has entries C[i,k] = sum_j A[i,j]*B[j,k]
    
    We can partition the indices into two groups of 2:
    - i = 2*i1 + i0 where i1, i0 in {0,1}
    - j = 2*j1 + j0 where j1, j0 in {0,1}  
    - k = 2*k1 + k0 where k1, k0 in {0,1}
    
    Then A[i,j] = A[2*i1+i0, 2*j1+j0]. If we view A as a 2x2 block matrix where
    each block is 2x2, then block (i1,j1) of A is the 2x2 matrix with entries
    A[2*i1+i0, 2*j1+j0] for i0,j0 in {0,1}.
    
    Similarly for B: block (j1,k1) has entries B[2*j1+j0, 2*k1+k0].
    C block (i1,k1) has entries C[2*i1+i0, 2*k1+k0].
    
    The block matrix multiplication: C_block(i1,k1) = sum_{j1} A_block(i1,j1) * B_block(j1,k1)
    
    Using Strassen on the 2x2 block matrices:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    Then:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Now each multiplication like (A00 + A11) * (B00 + B11) is a 2x2 matrix multiply.
    We can decompose each 2x2 multiply using Strassen again (7 rank-1 tensors).
    
    So for each of the 7 "outer" Strassen products, we have 7 "inner" Strassen products,
    giving 49 rank-1 tensors total.
    
    Let's define the Strassen decomposition for 2x2 matrices explicitly.
    
    For 2x2 matrices X and Y with entries:
    X = [[x00, x01], [x10, x11]]
    Y = [[y00, y01], [y10, y11]]
    
    Z = X*Y where:
    z00 = x00*y00 + x01*y10
    z01 = x00*y01 + x01*y11
    z10 = x10*y00 + x11*y10
    z11 = x10*y01 + x11*y11
    
    Strassen decomposition for 2x2:
    s1 = (x00 + x11)
    s2 = (x10 + x11)
    s3 = x00
    s4 = x11
    s5 = (x00 + x01)
    s6 = (x10 - x00)
    s7 = (x01 - x11)
    
    t1 = (y00 + y11)
    t2 = y00
    t3 = (y01 - y11)
    t4 = (y10 - y00)
    t5 = (y00 + y01)
    t6 = (y10 + y11)
    
    Then:
    z00 = s1*t1 + s4*t4 - s5*t6 + s7*t6  ... wait, let me get the standard Strassen right.
    
    Standard Strassen for 2x2:
    m1 = (x00 + x11) * (y00 + y11)
    m2 = (x10 + x11) * y00
    m3 = x00 * (y01 - y11)
    m4 = x11 * (y10 - y00)
    m5 = (x00 + x01) * y11
    m6 = (x10 - x00) * (y00 + y01)
    m7 = (x01 - x11) * (y10 + y11)
    
    z00 = m1 + m4 - m5 + m7
    z01 = m3 + m5
    z10 = m2 + m4
    z11 = m1 - m2 + m3 + m6
    
    Each mi is a product of two linear combinations. Each such product can be written as
    a sum of rank-1 tensors. But actually, each mi = (linear_comb_of_X) * (linear_comb_of_Y)
    is itself a scalar multiplication, which is rank-1 in the tensor sense if we consider
    the entries.
    
    Wait, I need to think about this more carefully in terms of the tensor decomposition.
    
    The matrix multiplication tensor T has entries T[i,j,k] = 1 if the multiplication
    C[i,k] = sum_j A[i,j] B[j,k] involves that term. Actually, the tensor for matrix
    multiplication of size n x n is defined as:
    T[i,j,k] = delta_{i,k} ... no.
    
    Let me reconsider. The matrix multiplication tensor T_{n} is a tensor in 
    R^{n^2} x R^{n^2} x R^{n^2} such that:
    T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i (for the schoolbook), but more generally,
    the tensor is defined so that contracting with A and B gives C.
    
    Specifically, for matrix multiplication C = AB where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T has indices (a, b, c) where a indexes A (n*m entries), b indexes B (m*p entries),
    c indexes C (n*p entries). T[a, b, c] = 1 if a = (i,j), b = (j,k), c = (i,k) for some i,j,k.
    
    So T[(i,j), (j,k), (i,k)] = 1 for all i,j,k, and 0 otherwise.
    
    A rank-1 tensor decomposition term is U[:, r] * V[:, r] * W[:, r] where:
    U[a, r] is the coefficient for A entry a in rank-1 term r
    V[b, r] is the coefficient for B entry b in rank-1 term r  
    W[c, r] is the coefficient for C entry c in rank-1 term r
    
    The decomposition says: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    For the schoolbook, each term corresponds to one (i,j,k) triple, with U[(i,j), r] = 1,
    V[(j,k), r] = 1, W[(i,k), r] = 1.
    
    For Strassen, each "multiplication" m_r = (linear_comb_of_A_entries) * (linear_comb_of_B_entries)
    contributes to several C entries. The rank-1 term for this multiplication has:
    U[a, r] = coefficient of A[a] in the linear combination
    V[b, r] = coefficient of B[b] in the linear combination
    W[c, r] = coefficient of C[c] in the output (which is the sum/difference of the m's)
    
    Wait, that's not quite right either. Let me think again.
    
    Actually, in a rank-1 tensor decomposition, each term is a product of three vectors.
    The sum of these products equals the tensor T.
    
    For Strassen, we have 7 products m1..m7. Each mr is a scalar that is a product of
    a linear combination of A entries and a linear combination of B entries. Then each
    C entry is a linear combination of the mr's.
    
    So for each mr, we can create 7 rank-1 terms (one for each C entry that it contributes to,
    with appropriate coefficients). But that would give 7*7 = 49 terms, but each term would
    have W[c, r] being the coefficient of mr in C[c], which could be -1, 0, or 1.
    
    Actually, let's be more precise. For a given "outer" Strassen multiplication mr,
    which computes mr = (sum_i alpha_i * A[i]) * (sum_j beta_j * B[j]), and contributes
    to C entries as sum_c gamma_c * C[c] where gamma_c is the coefficient.
    
    Hmm, this is getting complicated. Let me just directly construct the 49 rank-1 tensors
    by explicitly working through the Strassen decomposition.
    
    Let me define the indices clearly:
    - A has 16 entries: A[i,j] for i,j in 0..3. Index in U: a = i*4 + j (since m=4, so a = i*m + j = i*4+j)
    - B has 16 entries: B[j,k] for j,k in 0..3. Index in V: b = j*4 + k (since p=4, so b = j*p + k = j*4+k)
    - C has 16 entries: C[i,k] for i,k in 0..3. Index in W: c = k*4 + i (per the trace convention W[k*n+i,r])
    
    Wait, the task says "The output factor follows the official trace convention W[k*n+i,r]."
    So W is indexed as W[k*n + i, r] where n=4. So for C[i,k], the index in W is k*4 + i.
    
    Now, let's use the block structure. Let i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0.
    
    A[i,j] where i=2*i1+i0, j=2*j1+j0. In the 2x2 block structure, this is block (i1,j1), entry (i0,j0).
    
    The 7 Strassen products for the 2x2 block multiplication:
    
    Let A00, A01, A10, A11 be the 2x2 blocks of A.
    Let B00, B01, B10, B11 be the 2x2 blocks of B.
    
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    Each M is a 2x2 matrix. We decompose each M using Strassen (7 rank-1 tensors for 2x2).
    
    For a 2x2 matrix X with entries x00, x01, x10, x11 (where x00 = X[0,0], x01 = X[0,1], etc.),
    and Y with entries y00, y01, y10, y11, the Strassen decomposition gives:
    
    m1 = (x00 + x11) * (y00 + y11)
    m2 = (x10 + x11) * y00
    m3 = x00 * (y01 - y11)
    m4 = x11 * (y10 - y00)
    m5 = (x00 + x01) * y11
    m6 = (x10 - x00) * (y00 + y01)
    m7 = (x01 - x11) * (y10 + y11)
    
    z00 = m1 + m4 - m5 + m7
    z01 = m3 + m5
    z10 = m2 + m4
    z11 = m1 - m2 + m3 + m6
    
    Now, for each "outer" Strassen product (say M1), the 2x2 matrix X = A00 + A11 has entries:
    X[0,0] = A[0,0] + A[2,2]  (i0=0,j0=0: A[2*0+0, 2*0+0] + A[2*1+0, 2*1+0] = A[0,0] + A[2,2])
    X[0,1] = A[0,1] + A[2,3]  (i0=0,j0=1: A[0,1] + A[2,3])
    X[1,0] = A[1,0] + A[3,2]  (i0=1,j0=0: A[1,0] + A[3,2])
    X[1,1] = A[1,1] + A[3,3]  (i0=1,j0=1: A[1,1] + A[3,3])
    
    Similarly for Y = B00 + B11:
    Y[0,0] = B[0,0] + B[2,2]
    Y[0,1] = B[0,1] + B[2,3]
    Y[1,0] = B[1,0] + B[3,2]
    Y[1,1] = B[1,1] + B[3,3]
    
    The 7 inner Strassen products for M1 are:
    m1_1 = (X[0,0] + X[1,1]) * (Y[0,0] + Y[1,1])
    m1_2 = (X[1,0] + X[1,1]) * Y[0,0]
    m1_3 = X[0,0] * (Y[0,1] - Y[1,1])
    m1_4 = X[1,1] * (Y[1,0] - Y[0,0])
    m1_5 = (X[0,0] + X[0,1]) * Y[1,1]
    m1_6 = (X[1,0] - X[0,0]) * (Y[0,0] + Y[0,1])
    m1_7 = (X[0,1] - X[1,1]) * (Y[1,0] + Y[1,1])
    
    And M1[0,0] = m1_1 + m1_4 - m1_5 + m1_7
    M1[0,1] = m1_3 + m1_5
    M1[1,0] = m1_2 + m1_4
    M1[1,1] = m1_1 - m1_2 + m1_3 + m1_6
    
    Now M1 contributes to C00 (the top-left 2x2 block of C). Specifically:
    C00[0,0] gets +M1[0,0], C00[0,1] gets +M1[0,1], C00[1,0] gets +M1[1,0], C00[1,1] gets +M1[1,1]
    
    Wait, C00 is the block at (i1=0, k1=0), so C[i,k] for i=2*0+i0= i0, k=2*0+k0=k0.
    So C00[0,0] = C[0,0], C00[0,1] = C[0,1], C00[1,0] = C[1,0], C00[1,1] = C[1,1].
    
    In the W index, C[i,k] is at position k*4 + i. So:
    C[0,0] -> W[0*4+0] = W[0]
    C[0,1] -> W[1*4+0] = W[4]
    C[1,0] -> W[0*4+1] = W[1]
    C[1,1] -> W[1*4+1] = W[5]
    
    So M1[0,0] contributes +1 to W[0], M1[0,1] contributes +1 to W[4], M1[1,0] contributes +1 to W[1], M1[1,1] contributes +1 to W[5].
    
    This is getting very detailed. Let me just code this up systematically.
    
    I'll define helper functions to extract the linear combinations for each Strassen step.
    """
    
    n, m, p = DIMENSIONS  # all 4
    
    # We'll build U, V, W as lists of lists, then convert to numpy if needed.
    # Rank = 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Helper: for a 2x2 block (i1, j1) of A, return the 4 entries as a dict or list
    # A[2*i1+i0, 2*j1+j0] for i0,j0 in {0,1}
    def A_block_entries(i1, j1):
        """Returns [A[0,0], A[0,1], A[1,0], A[1,1]] for block (i1,j1)"""
        # Entry (i0, j0) in the block corresponds to A[2*i1+i0, 2*j1+j0]
        # Index in U: a = i*m + j = (2*i1+i0)*4 + (2*j1+j0)
        return [
            (2*i1 + 0) * 4 + (2*j1 + 0),  # (0,0)
            (2*i1 + 0) * 4 + (2*j1 + 1),  # (0,1)
            (2*i1 + 1) * 4 + (2*j1 + 0),  # (1,0)
            (2*i1 + 1) * 4 + (2*j1 + 1),  # (1,1)
        ]
    
    def B_block_entries(j1, k1):
        """Returns [B[0,0], B[0,1], B[1,0], B[1,1]] for block (j1,k1)"""
        # Entry (j0, k0) in the block corresponds to B[2*j1+j0, 2*k1+k0]
        # Index in V: b = j*p + k = (2*j1+j0)*4 + (2*k1+k0)
        return [
            (2*j1 + 0) * 4 + (2*k1 + 0),  # (0,0)
            (2*j1 + 0) * 4 + (2*k1 + 1),  # (0,1)
            (2*j1 + 1) * 4 + (2*k1 + 0),  # (1,0)
            (2*j1 + 1) * 4 + (2*k1 + 1),  # (1,1)
        ]
    
    def C_block_entries(i1, k1):
        """Returns the W indices for C[0,0], C[0,1], C[1,0], C[1,1] in block (i1,k1)"""
        # Entry (i0, k0) in the block corresponds to C[2*i1+i0, 2*k1+k0]
        # Index in W: c = k*n + i = (2*k1+k0)*4 + (2*i1+i0)
        return [
            (2*k1 + 0) * 4 + (2*i1 + 0),  # C[0,0] -> W[0*4+0]
            (2*k1 + 1) * 4 + (2*i1 + 0),  # C[0,1] -> W[1*4+0]
            (2*k1 + 0) * 4 + (2*i1 + 1),  # C[1,0] -> W[0*4+1]
            (2*k1 + 1) * 4 + (2*i1 + 1),  # C[1,1] -> W[1*4+1]
        ]
    
    # Define the 7 outer Strassen products.
    # Each product is: X_block * Y_block where X is a sum/diff of A blocks, Y is a sum/diff of B blocks.
    # We represent each as a list of (block_index, coefficient) pairs for A and B.
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    outer_products = [
        # (A_blocks, B_blocks, C_block, C_coeff)
        # For M1: A00+A11, B00+B11, contributes to C00 with coeff +1
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], (0,0), 1),
        # For M2: A10+A11, B00, contributes to C10 with coeff +1
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], (1,0), 1),
        # For M3: A00, B01-B11, contributes to C01 with coeff +1
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], (0,1), 1),
        # For M4: A11, B10-B00, contributes to C10 with coeff +1 and C00 with coeff +1
        # Wait, M4 contributes to C00 and C10. Let me re-check.
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        # So M4 contributes to C00 with coeff +1 and C10 with coeff +1.
        # I need to handle multiple C contributions per outer product.
    ]
    
    # Let me restructure. Each outer product contributes to one or more C blocks.
    # I'll define for each outer product:
    # - A_linear: list of (i1, j1, coeff) for A blocks
    # - B_linear: list of (j1, k1, coeff) for B blocks
    # - C_contributions: list of ((i1, k1), coeff) for C blocks
    
    outer_strassen = [
        # M1
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        },
        # M2
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((1,0), 1)]
        },
        # M3
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [((0,1), 1)]
        },
        # M4
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((1,0), 1)]
        },
        # M5
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,0), -1), ((0,1), 1)]
        },
        # M6
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        # M7
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
    ]
    
    # For each outer product, we do a 2x2 Strassen decomposition (7 inner products).
    # For a 2x2 matrix X with entries x00, x01, x10, x11 and Y with y00, y01, y10, y11:
    # Inner products:
    # i1 = (x00 + x11) * (y00 + y11)
    # i2 = (x10 + x11) * y00
    # i3 = x00 * (y01 - y11)
    # i4 = x11 * (y10 - y00)
    # i5 = (x00 + x01) * y11
    # i6 = (x10 - x00) * (y00 + y01)
    # i7 = (x01 - x11) * (y10 + y11)
    #
    # C entries:
    # z00 = i1 + i4 - i5 + i7
    # z01 = i3 + i5
    # z10 = i2 + i4
    # z11 = i1 - i2 + i3 + i6
    
    # For each inner product, we need to determine:
    # - The linear combination of A entries (from X's linear combination)
    # - The linear combination of B entries (from Y's linear combination)
    # - Which C entries it contributes to, and with what coefficient
    
    # X's entries are linear combinations of A block entries.
    # For X = sum of A blocks with coefficients, X[i0,j0] = sum over blocks of coeff * A_block[i0,j0]
    
    # Let me define the 7 inner Strassen products' X and Y linear combinations.
    # X has 4 entries: x00, x01, x10, x11
    # Y has 4 entries: y00, y01, y10, y11
    
    inner_strassen_X = [
        # i1: x00 + x11
        {0: 1, 3: 1},  # x00 at index 0, x11 at index 3
        # i2: x10 + x11
        {2: 1, 3: 1},
        # i3: x00
        {0: 1},
        # i4: x11
        {3: 1},
        # i5: x00 + x01
        {0: 1, 1: 1},
        # i6: x10 - x00
        {2: 1, 0: -1},
        # i7: x01 - x11
        {1: 1, 3: -1},
    ]
    
    inner_strassen_Y = [
        # i1: y00 + y11
        {0: 1, 3: 1},
        # i2: y00
        {0: 1},
        # i3: y01 - y11
        {1: 1, 3: -1},
        # i4: y10 - y00
        {2: 1, 0: -1},
        # i5: y11
        {3: 1},
        # i6: y00 + y01
        {0: 1, 1: 1},
        # i7: y10 + y11
        {2: 1, 3: 1},
    ]
    
    # C contributions for each inner product:
    # z00 = i1 + i4 - i5 + i7
    # z01 = i3 + i5
    # z10 = i2 + i4
    # z11 = i1 - i2 + i3 + i6
    
    inner_C_contrib = [
        # i1: z00 +1, z11 +1
        [(0, 1), (3, 1)],
        # i2: z10 +1, z11 -1
        [(2, 1), (3, -1)],
        # i3: z01 +1, z11 +1
        [(1, 1), (3, 1)],
        # i4: z00 +1, z10 +1
        [(0, 1), (2, 1)],
        # i5: z00 -1, z01 +1
        [(0, -1), (1, 1)],
        # i6: z11 +1
        [(3, 1)],
        # i7: z00 +1
        [(0, 1)],
    ]
    
    # Now, for each outer product and each inner product, create a rank-1 tensor term.
    rank_idx = 0
    for outer in outer_strassen:
        A_blocks = outer['A']  # list of ((i1,j1), coeff)
        B_blocks = outer['B']  # list of ((j1,k1), coeff)
        C_contribs = outer['C']  # list of ((i1,k1), coeff)
        
        for inner_idx in range(7):
            X_lin = inner_strassen_X[inner_idx]  # dict: entry_idx -> coeff
            Y_lin = inner_strassen_Y[inner_idx]
            C_inner_contrib = inner_C_contrib[inner_idx]  # list of (entry_idx, coeff)
            
            # Build U contribution for this rank-1 term
            # For each X entry (i0,j0) with coefficient cx, and for each A block (i1,j1) with coefficient ca,
            # the A entry A[2*i1+i0, 2*j1+j0] gets coefficient ca * cx
            for (x_entry, cx) in X_lin.items():
                i0, j0 = divmod(x_entry, 2)  # x_entry 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                for (block, ca) in A_blocks:
                    i1, j1 = block
                    a_idx = (2*i1 + i0) * 4 + (2*j1 + j0)
                    U[a_idx][rank_idx] += ca * cx
            
            # Build V contribution
            for (y_entry, cy) in Y_lin.items():
                j0, k0 = divmod(y_entry, 2)
                for (block, cb) in B_blocks:
                    j1, k1 = block
                    b_idx = (2*j1 + j0) * 4 + (2*k1 + k0)
                    V[b_idx][rank_idx] += cb * cy
            
            # Build W contribution
            for (c_entry, cc_inner) in C_inner_contrib:
                i0, k0 = divmod(c_entry, 2)
                for (block, cc_outer) in C_contribs:
                    i1, k1 = block
                    c_idx = (2*k1 + k0) * 4 + (2*i1 + i0)
                    W[c_idx][rank_idx] += cc_outer * cc_inner
            
            rank_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
