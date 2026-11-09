"""Matrix multiplication 2x3x6 rank-minimal decomposition."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # safety margin before 120s limit

    n, m, p = 2, 3, 6
    # We'll build the Strassen-based decomposition of rank 33.
    #
    # Strategy: Split A (2x3) into A1 (2x2) and A2 (2x1).
    # Split B (3x6) into B1 (2x6) and B2 (1x6).
    # C = A1*B1 + A2*B2
    #
    # For A1*B1 (2x2 times 2x6): split B1 into three 2x2 blocks B1a, B1b, B1c.
    # Use Strassen's algorithm (rank 7) for each 2x2x2 block multiplication.
    # That gives 3*7 = 21 terms for A1*B1.
    #
    # For A2*B2 (2x1 times 1x6): schoolbook rank = 2*1*6 = 12 terms.
    #
    # Total rank = 21 + 12 = 33.
    #
    # Indexing conventions:
    # U has shape (n*m, rank) = (6, 33): rows indexed by i*m + j where i in [0,2), j in [0,3)
    # V has shape (m*p, rank) = (18, 33): rows indexed by j*p + k where j in [0,3), k in [0,6)
    # W has shape (p*n, rank) = (12, 33): rows indexed by k*n + i where k in [0,6), i in [0,2)
    #
    # The tensor identity is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = A[i,j]*B[j,k]
    #
    # Strassen for 2x2 * 2x2:
    #   C = A*B where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    #   m1 = (a+d)(e+h)
    #   m2 = (c+d)(e)
    #   m3 = (a)(f-h)
    #   m4 = (d)(g-e)
    #   m5 = (a+b)(h)
    #   m6 = (c-a)(e+g)
    #   m7 = (b-d)(f+h)
    #   C[0,0] = m1 + m4 - m5 - m7
    #   C[0,1] = m3 + m5
    #   C[1,0] = m2 + m4
    #   C[1,1] = m1 - m2 + m3 - m6
    #
    # We need to express each Strassen term as a rank-1 contribution to the full tensor.
    # For a 2x2x2 block at position (block_col_offset = 2*block_idx):
    #   A elements: A1[i1, j1] where i1 in {0,1}, j1 in {0,1} (these are rows i1 of A, cols j1, j1+1 of A)
    #   B elements: B1[j1, 2*block_idx + k1] where j1 in {0,1}, k1 in {0,1}
    #
    # For each Strassen product like (a+d)(e+h):
    #   a = A[0,0], d = A[1,1] in the 2x2 sub-block
    #   e = B[0,0], h = B[1,1] in the corresponding 2x2 sub-block of B
    #
    # The rank-1 tensor for this product:
    #   U[i*m + j] gets the A-coefficient
    #   V[j*p + k] gets the B-coefficient
    #   W[k*n + i] gets +1 (the C coefficient)
    #
    # For (a+d)(e+h) = a*e + a*h + d*e + d*h, we need 4 rank-1 terms.
    # But Strassen groups them into one "multiply" of sums, which in tensor decomposition
    # means we introduce an auxiliary factor. Actually no - in tensor decomposition,
    # each rank-1 term is a single product of three scalars.
    #
    # Wait. Let me reconsider. Strassen reduces the number of scalar multiplications
    # from 4 to 7 for 2x2. But in tensor decomposition, each "multiplication" corresponds
    # to one rank-1 term. So Strassen gives a rank-7 decomposition of the 2x2x2 tensor.
    #
    # Strassen rank-7 decomposition of 2x2x2:
    # The 7 rank-1 tensors are:
    #   r1: A[[0,0],[0,1],[1,0],[1,1]] dot B[[0,0],[0,1],[1,0],[1,1]] -> contributes to C
    #       Specifically: (a+d)(e+h) contributes to C[0,0] and C[1,1]
    #       U: A[0,0]=1, A[1,1]=1 (for the a+d part)
    #       V: B[0,0]=1, B[1,1]=1 (for the e+h part)
    #       W: C[0,0]=1, C[1,1]=1
    #
    #   r2: (c+d)(e)
    #       U: A[0,1]=1, A[1,1]=1
    #       V: B[0,0]=1
    #       W: C[1,0]=1
    #
    #   r3: (a)(f-h)
    #       U: A[0,0]=1
    #       V: B[0,1]=1, B[1,1]=-1
    #       W: C[0,1]=1, C[1,1]=1
    #
    #   r4: (d)(g-e)
    #       U: A[1,1]=1
    #       V: B[1,0]=1, B[0,0]=-1
    #       W: C[0,0]=1, C[1,0]=1
    #
    #   r5: (a+b)(h)
    #       U: A[0,0]=1, A[0,1]=1
    #       V: B[1,1]=1
    #       W: C[0,1]=1, C[1,1]=-1
    #
    #   r6: (c-a)(e+g)
    #       U: A[0,1]=1, A[0,0]=-1
    #       V: B[0,0]=1, B[1,0]=1
    #       W: C[1,1]=-1
    #
    #   r7: (b-d)(f+h)
    #       U: A[0,1]=1, A[1,1]=-1
    #       V: B[0,1]=1, B[1,1]=1
    #       W: C[0,0]=-1, C[1,1]=1

    rank = 33
    U = [[0] * rank for _ in range(6)]   # n*m = 6 rows
    V = [[0] * rank for _ in range(18)]  # m*p = 18 rows
    W = [[0] * rank for _ in range(12)]  # p*n = 12 rows

    # A1 is 2x2 (rows 0-1, cols 0-1 of A), A2 is 2x1 (rows 0-1, col 2 of A)
    # B1 is 2x6 (rows 0-1 of B, all cols), B2 is 1x6 (row 2 of B, all cols)

    # Strassen 2x2x2 rank-7 terms: (u_coeffs, v_coeffs, w_coeffs)
    # Each term: list of (U_row_offset, coeff), list of (V_row_offset, coeff), list of (W_row_offset, coeff)
    # U_row_offset is within the 2x2 A block: 0->A[0,0], 1->A[0,1], 2->A[1,0], 3->A[1,1]
    # In global U indexing: row = i*m + j where i is the A row (0 or 1), j is the A col
    #   For A1 (cols 0,1): j=0 -> col 0, j=1 -> col 1
    #   So U row for A[i, j] in A1 is i*3 + j
    # V_row_offset within the 2x2 B block: 0->B[0,0], 1->B[0,1], 2->B[1,0], 3->B[1,1]
    # In global V indexing: row = j*p + k where j is B row, k is B col
    #   For B1 block at column offset 2*bi: B[j, 2*bi + k1]
    #   So V row = j*6 + (2*bi + k1)
    # W_row_offset within 2x2 C block: 0->C[0,0], 1->C[0,1], 2->C[1,0], 3->C[1,1]
    # In global W indexing: row = k*n + i where k is C col, i is C row
    #   So W row = k*2 + i

    # Strassen terms defined as (u_list, v_list, w_list)
    # u_list: [(a_row, a_col_local, coeff), ...] for A
    # v_list: [(b_row, b_col_local, coeff), ...] for B
    # w_list: [(c_row, c_col_local, coeff), ...] for C
    strassen_terms = [
        # r1: (a+d)(e+h)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # r2: (c+d)(e)
        ([(0,1,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        # r3: (a)(f-h)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # r4: (d)(g-e)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # r5: (a+b)(h)
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(1,1,-1)]),
        # r6: (c-a)(e+g)
        ([(0,1,1),(0,0,-1)], [(0,0,1),(1,0,1)], [(1,1,-1)]),
        # r7: (b-d)(f+h)
        ([(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)], [(0,0,-1),(1,1,1)]),
    ]

    col = 0
    # Process the three 2x2 blocks of B1
    for bi in range(3):
        for (u_list, v_list, w_list) in strassen_terms:
            # Verify this is a valid rank-1 tensor: the product of U-part, V-part, W-part
            # should equal sum over all combinations of (a_i + ...)*(b_j + ...)*(c_k + ...)
            # This is the outer product structure.
            for (ai, aj, ucoeff) in u_list:
                # ai = A row (0 or 1 in A1), aj = A col local (0 or 1)
                u_row = ai * 3 + aj  # global U row index
                for (brow, bcol, vcoeff) in v_list:
                    # brow = B row (0 or 1 in B1), bcol = B col local (0 or 1)
                    v_row = brow * 6 + (2 * bi + bcol)  # global V row index
                    for (crow, ccol, wcoeff) in w_list:
                        # crow = C row (0 or 1), ccol = C col local (0 or 1)
                        c_col_global = 2 * bi + ccol
                        w_row = c_col_global * 2 + crow  # global W row index
                        U[u_row][col] += ucoeff
                        V[v_row][col] += vcoeff
                        W[w_row][col] += wcoeff
            col += 1

    # Now A2*B2: A2 is 2x1 (col 2 of A), B2 is 1x6 (row 2 of B)
    # Schoolbook: 2*1*6 = 12 terms
    # C[i, k] = A[i, 2] * B[2, k]
    for i in range(2):
        for k in range(6):
            U[i * 3 + 2][col] = 1   # A[i, 2]
            V[2 * 6 + k][col] = 1   # B[2, k]
            W[k * 2 + i][col] = 1   # C[i, k]
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
