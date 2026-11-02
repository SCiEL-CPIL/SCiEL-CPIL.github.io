"""
Rank-65 exact decomposition for 4x4 by 4x5 matrix multiplication.

Approach:
  - Split the 4x5 right factor B into B1 (4x4) and B2 (4x1).
  - For A*B1 (4x4x4), use a 2x2-block Strassen decomposition with inner 2x2x2
    also done via Strassen (7 multiplications each), giving 7*7 = 49 rank-1 terms.
  - For A*B2 (4x4x1), use schoolbook: 16 rank-1 terms.
  - Total rank = 49 + 16 = 65.

Strassen for 2x2x2:
  Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
  P1 = (a+d)(e+h)
  P2 = (c+d)e
  P3 = a(f-h)
  P4 = d(g-e)
  P5 = (a+b)h
  P6 = (c-a)(e+f)
  P7 = (b-d)(g+h)
  C11 = P1 - P3 + P5 - P7
  C12 = P3 + P4
  C21 = P2 + P4
  C22 = P1 - P2 + P6 + P7

For 4x4x4 via 2x2 blocks:
  A = [[A11, A12],
       [A21, A22]]  (each 2x2)
  B = [[B11, B12],
       [B21, B22]]  (each 2x2)
  
  C11 = A11*B11 + A12*B21
  C12 = A11*B12 + A12*B22
  C21 = A21*B11 + A22*B21
  C22 = A21*B12 + A22*B22

Each 2x2x2 product uses 7 rank-1 terms (Strassen).
So 4 block products * 7 = 28 rank-1 terms? No wait.

Actually, each C-block is a 2x2 matrix computed from two 2x2x2 products.
Each 2x2x2 product requires 7 scalar multiplications (rank-1 terms in the tensor sense).
So each C-block needs 7+7 = 14 rank-1 terms.
4 C-blocks * 14 = 56? No, that's not right either.

Let me reconsider. The tensor for 4x4x4 matrix multiplication has rank at most 64.
Using 2x2 block Strassen:
- We have 4 block positions in C, each a 2x2.
- Each 2x2 C-block = sum of two 2x2x2 products.
- Each 2x2x2 product = 7 rank-1 terms (Strassen).
- So each C-block = 14 rank-1 terms.
- Total = 4 * 14 = 56 rank-1 terms.

But wait, that's not how Strassen works for 4x4. Let me think again.

Strassen for 4x4: view as 2x2 blocks of 2x2. The outer Strassen gives 7 "products" of 2x2 matrices.
Each 2x2 product is then computed with Strassen (7 scalar mults) or schoolbook (4 scalar mults).

If inner is Strassen: 7 * 7 = 49.
If inner is schoolbook: 7 * 4 = 28... no, 7 * 4 = 28? No, each 2x2x2 with schoolbook is 4 rank-1 terms.
So 7 * 4 = 28.

Wait, I think I was confusing myself. Let me be precise.

The 4x4x4 tensor M_{4,4,4} can be decomposed as follows:
- Use Strassen at the top level: 7 multiplications of 2x2 matrices.
- Each 2x2x2 multiplication can be done with 7 rank-1 terms (Strassen) or 4 (schoolbook).

So rank <= 7*7 = 49 (both Strassen) or 7*4 = 28 (outer Strassen, inner schoolbook).

Wait, 28 seems too good. Let me verify: the rank of M_{2,2,2} is 7 (Strassen is optimal).
The rank of M_{4,4,4} is at most 7*7=49 if we use Strassen recursively.

Actually I recall that the known upper bound for 4x4x4 is 49 (from recursive Strassen).
And it's been shown that the rank is at most 49.

So for 4x4x4: rank 49.
For 4x4x1: rank 16 (schoolbook, and this is actually minimal since the tensor is full rank).

Total: 49 + 16 = 65.

Let me implement this carefully.

Strassen 2x2x2 tensor (7 terms):
For A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B:
  C11 = P1 - P3 + P5 - P7
  C12 = P3 + P4
  C21 = P2 + P4
  C22 = P1 - P2 + P6 + P7

where:
  P1 = (a+d)(e+h)
  P2 = (c+d)e
  P3 = a(f-h)
  P4 = d(g-e)
  P5 = (a+b)h
  P6 = (c-a)(e+f)
  P7 = (b-d)(g+h)

Each Pi is a rank-1 term: (linear combo of A entries) * (linear combo of B entries) contributes to C entries.

For the tensor decomposition, each rank-1 term corresponds to:
  u (vector over A indices) * v (vector over B indices) * w (vector over C indices)

For P1 = (a+d)(e+h):
  u: A[0,0]=1, A[1,1]=1
  v: B[0,0]=1, B[1,1]=1
  w: C[0,0]=1, C[1,1]=1 (since P1 appears in both C11 and C22 with +1)

For P2 = (c+d)e:
  u: A[1,0]=1, A[1,1]=1
  v: B[0,0]=1
  w: C[1,0]=1, C[1,1]=-1

For P3 = a(f-h):
  u: A[0,0]=1
  v: B[0,1]=1, B[1,1]=-1
  w: C[0,0]=-1, C[0,1]=1

For P4 = d(g-e):
  u: A[1,1]=1
  v: B[1,0]=1, B[0,0]=-1
  w: C[0,1]=1, C[1,0]=1

For P5 = (a+b)h:
  u: A[0,0]=1, A[0,1]=1
  v: B[1,1]=1
  w: C[0,0]=1

For P6 = (c-a)(e+f):
  u: A[1,0]=1, A[0,0]=-1
  v: B[0,0]=1, B[0,1]=1
  w: C[1,1]=1

For P7 = (b-d)(g+h):
  u: A[0,1]=1, A[1,1]=-1
  v: B[1,0]=1, B[1,1]=1
  w: C[0,0]=-1, C[1,1]=1

Now for 4x4x4 via outer Strassen:
A = [[A00, A01],
     [A10, A11]] where each is 2x2
B = [[B00, B01],
     [B10, B11]] where each is 2x2

Outer Strassen products:
  Q1 = (A00+A11)(B00+B11)
  Q2 = (A10+A11)B00
  Q3 = A00(B01-B11)
  Q4 = A11(B10-B00)
  Q5 = (A00+A01)B11
  Q6 = (A10-A00)(B00+B01)
  Q7 = (A01-A11)(B10+B11)

Then:
  C00 = Q1 - Q3 + Q5 - Q7
  C01 = Q3 + Q4
  C10 = Q2 + Q4
  C11 = Q1 - Q2 + Q6 + Q7

Each Qi is a 2x2x2 product, decomposed with inner Strassen (7 terms each).
Total: 7 * 7 = 49 terms.

For 4x4x1 (A is 4x4, B is 4x1, C is 4x1):
Schoolbook: C[i] = sum_j A[i,j]*B[j]
16 terms.

Total rank: 49 + 16 = 65.
"""

def construct():
    n, m, p = 4, 4, 5
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    # Helper: index for A (n x m = 4x4), row i, col j
    def a_idx(i, j):
        return i * m + j

    # Helper: index for B (m x p = 4x5), row j, col k
    def b_idx(j, k):
        return j * p + k

    # Helper: index for C (p x n = 5x4), row k, col i  (trace convention W[k*n+i])
    def c_idx(k, i):
        return k * n + i

    # Strassen 2x2x2: returns list of (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict from (i,j) in 2x2 A to coefficient
    # v_coeffs: dict from (j,k) in 2x2 B to coefficient
    # w_coeffs: dict from (k,i) in 2x2 C to coefficient
    def strassen_2x2x2():
        # Each term: (u, v, w) where u is over 4 A-entries, v over 4 B-entries, w over 4 C-entries
        terms = []
        # P1 = (a+d)(e+h) -> C11 += 1, C22 += 1
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ))
        # P2 = (c+d)e -> C21 += 1, C22 += -1
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ))
        # P3 = a(f-h) -> C11 += -1, C12 += 1
        terms.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,0): -1, (0,1): 1}
        ))
        # P4 = d(g-e) -> C12 += 1, C21 += 1
        terms.append((
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,1): 1, (1,0): 1}
        ))
        # P5 = (a+b)h -> C11 += 1
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): 1}
        ))
        # P6 = (c-a)(e+f) -> C22 += 1
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ))
        # P7 = (b-d)(g+h) -> C11 += -1, C22 += 1
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): -1, (1,1): 1}
        ))
        return terms

    strassen_terms = strassen_2x2x2()

    # Outer Strassen for 4x4x4
    # A blocks: A00=(0,0)-(1,1), A01=(0,2)-(1,3), A10=(2,0)-(3,1), A11=(2,2)-(3,3)
    # B blocks: B00=(0,0)-(1,1), B01=(0,2)-(1,3), B10=(2,0)-(3,1), B11=(2,2)-(3,3)
    # C blocks: C00=(0,0)-(1,1), C01=(0,2)-(1,3), C10=(2,0)-(3,1), C11=(2,2)-(3,3)
    # Note: C is p x n = 5x4, so C block (ci, cj) where ci in {0,1}, cj in {0,1}
    #   C block (0,0): rows 0-1 of C (k=0,1), cols 0-1 (i=0,1)
    #   C block (0,1): rows 0-1 of C (k=0,1), cols 2-3 (i=2,3)
    #   C block (1,0): rows 2-3 of C (k=2,3), cols 0-1 (i=0,1)
    #   C block (1,1): rows 2-3 of C (k=2,3), cols 2-3 (i=2,3)
    #   C block (2,0): row 4 of C (k=4), cols 0-1 (i=0,1) -- this is for the 4x4x1 part
    #   C block (2,1): row 4 of C (k=4), cols 2-3 (i=2,3) -- this is for the 4x4x1 part

    # Outer Strassen coefficients for each Q:
    # Q1 = (A00+A11)(B00+B11)
    # Q2 = (A10+A11)B00
    # Q3 = A00(B01-B11)
    # Q4 = A11(B10-B00)
    # Q5 = (A00+A01)B11
    # Q6 = (A10-A00)(B00+B01)
    # Q7 = (A01-A11)(B10+B11)

    # For each outer Q, we need to map the 2x2 A-block indices to global 4x4 A indices,
    # the 2x2 B-block indices to global 4x4 B indices (cols 0-3),
    # and the 2x2 C-block indices to global 4x4 C indices (rows 0-3).

    # A block offsets:
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    a_block_offsets = {
        'A00': (0, 0),
        'A01': (0, 2),
        'A10': (2, 0),
        'A11': (2, 2)
    }

    # B block offsets (for 4x4 part, cols 0-3):
    # B00: (0,0), B01: (0,2), B10: (2,0), B11: (2,2)
    b_block_offsets = {
        'B00': (0, 0),
        'B01': (0, 2),
        'B10': (2, 0),
        'B11': (2, 2)
    }

    # C block offsets (for 4x4 part, rows 0-3 of C which is 5x4):
    # C00: (0,0), C01: (0,2), C10: (2,0), C11: (2,2)
    c_block_offsets = {
        'C00': (0, 0),
        'C01': (0, 2),
        'C10': (2, 0),
        'C11': (2, 2)
    }

    # Outer Strassen: define each Q as (a_blocks_with_coeffs, b_blocks_with_coeffs)
    # a_blocks: dict from block name to coefficient
    # b_blocks: dict from block name to coefficient
    outer_qs = [
        # Q1 = (A00+A11)(B00+B11)
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}),
        # Q2 = (A10+A11)B00
        ({'A10': 1, 'A11': 1}, {'B00': 1}),
        # Q3 = A00(B01-B11)
        ({'A00': 1}, {'B01': 1, 'B11': -1}),
        # Q4 = A11(B10-B00)
        ({'A11': 1}, {'B10': 1, 'B00': -1}),
        # Q5 = (A00+A01)B11
        ({'A00': 1, 'A01': 1}, {'B11': 1}),
        # Q6 = (A10-A00)(B00+B01)
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B01': 1}),
        # Q7 = (A01-A11)(B10+B11)
        ({'A01': 1, 'A11': -1}, {'B10': 1, 'B11': 1}),
    ]

    # C block contributions from each Q:
    # C00 = Q1 - Q3 + Q5 - Q7
    # C01 = Q3 + Q4
    # C10 = Q2 + Q4
    # C11 = Q1 - Q2 + Q6 + Q7
    c_contrib = [
        # For Q1: C00 += 1, C11 += 1
        {'C00': 1, 'C11': 1},
        # For Q2: C10 += 1, C11 += -1
        {'C10': 1, 'C11': -1},
        # For Q3: C00 += -1, C01 += 1
        {'C00': -1, 'C01': 1},
        # For Q4: C01 += 1, C10 += 1
        {'C01': 1, 'C10': 1},
        # For Q5: C00 += 1
        {'C00': 1},
        # For Q6: C11 += 1
        {'C11': 1},
        # For Q7: C00 += -1, C11 += 1
        {'C00': -1, 'C11': 1},
    ]

    col = 0
    for q_idx in range(7):
        a_blocks, b_blocks = outer_qs[q_idx]
        contrib = c_contrib[q_idx]

        for inner_idx, (u_inner, v_inner, w_inner) in enumerate(strassen_terms):
            # For this rank-1 term, build global U, V, W vectors
            # U: over 16 A entries (4x4)
            # V: over 20 B entries (4x5), but for 4x4 part only cols 0-3
            # W: over 20 C entries (5x4), but for 4x4 part only rows 0-3

            u_global = {}
            v_global = {}
            w_global = {}

            # Expand A blocks
            for block_name, coeff in a_blocks.items():
                row_off, col_off = a_block_offsets[block_name]
                for (bi, bj), u_coeff in u_inner.items():
                    gi = row_off + bi
                    gj = col_off + bj
                    idx = a_idx(gi, gj)
                    u_global[idx] = u_global.get(idx, 0) + coeff * u_coeff

            # Expand B blocks
            for block_name, coeff in b_blocks.items():
                row_off, col_off = b_block_offsets[block_name]
                for (bi, bj), v_coeff in v_inner.items():
                    gj = row_off + bi
                    gk = col_off + bj
                    idx = b_idx(gj, gk)
                    v_global[idx] = v_global.get(idx, 0) + coeff * v_coeff

            # Expand C blocks
            for block_name, coeff in contrib.items():
                row_off, col_off = c_block_offsets[block_name]
                for (ci, cj), w_coeff in w_inner.items():
                    # ci is row in C (k index), cj is col in C (i index)
                    gk = row_off + ci
                    gi = col_off + cj
                    idx = c_idx(gk, gi)
                    w_global[idx] = w_global.get(idx, 0) + coeff * w_coeff

            # Fill in the column
            for idx, val in u_global.items():
                U[idx][col] = val
            for idx, val in v_global.items():
                V[idx][col] = val
            for idx, val in w_global.items():
                W[idx][col] = val

            col += 1

    # Now add the 4x4x1 part: A (4x4) * B2 (4x1) = C2 (4x1)
    # B2 is columns 4 of B (index 4 in B, which is row j, col 4)
    # C2 is row 4 of C (k=4), cols 0-3 (i=0-3)
    # Schoolbook: C2[i] = sum_j A[i,j] * B2[j]
    for i in range(4):
        for j in range(4):
            U[a_idx(i, j)][col] = 1
            V[b_idx(j, 4)][col] = 1
            W[c_idx(4, i)][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
