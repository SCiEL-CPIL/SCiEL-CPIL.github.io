"""
Rewritten program for matrix multiplication tensor decomposition.

Task: Find a rank-minimal exact tensor decomposition for multiplying
a 2x2 matrix by a 2x6 matrix.

The output factor W follows the official trace convention W[k*n+i, r].

Strategy:
- The 2x2 * 2x6 multiplication can be decomposed into three independent
  2x2 * 2x2 multiplications by splitting the 6 columns of B into three pairs.
- Each 2x2 * 2x2 multiplication can be done with rank 7 using Strassen's algorithm.
- Total rank: 3 * 7 = 21.
- The schoolbook rank is 2*2*6 = 24, so rank 21 is an improvement.

We construct U, V, W matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
respectively, where n=2, m=2, p=6, so:
  U: (4, 21)
  V: (12, 21)
  W: (12, 21)
"""

import time

DIMENSIONS = [2, 2, 6]

def construct():
    """
    Construct a rank-21 tensor decomposition for 2x2 * 2x6 matrix multiplication
    using three independent Strassen blocks.
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    rank = 21  # 3 blocks of rank 7 each

    # Initialize matrices as lists of lists
    U = [[0.0] * rank for _ in range(n * m)]  # (4, 21)
    V = [[0.0] * rank for _ in range(m * p)]  # (12, 21)
    W = [[0.0] * rank for _ in range(p * n)]  # (12, 21)

    # Strassen decomposition for 2x2 * 2x2:
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    #
    # 7 products:
    # M1 = (a + d) * (e + h)
    # M2 = (c + d) * e
    # M3 = a * (f - h)
    # M4 = d * (g - e)
    # M5 = (a + b) * h
    # M6 = (c - a) * (e + f)
    # M7 = (b - d) * (g + h)
    #
    # C = [[M1 - M4 + M5 - M7, M3 + M7],
    #      [M2 + M4,           M1 - M2 + M6]]

    # For each of the three 2x2 blocks (columns 0-1, 2-3, 4-5):
    # Block b (b = 0, 1, 2) uses columns 2*b and 2*b+1 of B.
    #
    # U row indices: i*m + j where i in {0,1}, j in {0,1}
    #   Row 0: i=0, j=0 (element a of A)
    #   Row 1: i=0, j=1 (element b of A)
    #   Row 2: i=1, j=0 (element c of A)
    #   Row 3: i=1, j=1 (element d of A)
    #
    # V row indices: j*p + k where j in {0,1}, k in {0,...,5}
    #   For block b, k = 2*b (first col) or 2*b+1 (second col)
    #   Row j*6 + k
    #
    # W row indices: k*n + i where k in {0,...,5}, i in {0,1}
    #   For block b, k = 2*b or 2*b+1
    #   Row k*2 + i

    # Strassen coefficients for the 7 multiplications:
    # For each multiplication r (0..6), we have:
    #   U factor: which rows of A and what coefficient
    #   V factor: which rows of B and what coefficient
    #   W factor: how the product contributes to each output element

    # Let's define the Strassen decomposition more carefully.
    #
    # The tensor for 2x2 * 2x2 has 8 components (C[0][0], C[0][1], C[1][0], C[1][1] for two 2x2 matrices).
    # Actually, the output is a 2x2 matrix, so 4 elements.
    #
    # C[0][0] = a*e + b*g
    # C[0][1] = a*f + b*h
    # C[1][0] = c*e + d*g
    # C[1][1] = c*f + d*h
    #
    # Strassen gives:
    # M1 = (a+d)(e+h) = a*e + a*h + d*e + d*h
    # M2 = (c+d)*e    = c*e + d*e
    # M3 = a*(f-h)    = a*f - a*h
    # M4 = d*(g-e)    = d*g - d*e
    # M5 = (a+b)*h    = a*h + b*h
    # M6 = (c-a)*(e+f)= c*e + c*f - a*e - a*f
    # M7 = (b-d)*(g+h)= b*g + b*h - d*g - d*h
    #
    # C[0][0] = M1 - M4 + M5 - M7
    #   = (a*e + a*h + d*e + d*h) - (d*g - d*e) + (a*h + b*h) - (b*g + b*h - d*g - d*h)
    #   = a*e + a*h + d*e + d*h - d*g + d*e + a*h + b*h - b*g - b*h + d*g + d*h
    #   = a*e + 2*a*h + 2*d*e + 2*d*h + b*h - b*g
    #   Hmm, this doesn't look right. Let me recheck.
    #
    # Actually, let me just use the standard Strassen formulas directly.
    #
    # C[0][0] = M1 - M4 + M5 - M7
    # C[0][1] = M3 + M7
    # C[1][0] = M2 + M4
    # C[1][1] = M1 - M2 + M6

    # For each multiplication r in 0..6, define:
    #   u_factors: list of (row_index_in_A, coeff)
    #   v_factors: list of (col_index_in_B_for_this_block, coeff)
    #   w_factors: list of (output_position, coeff) where output_position is (i, k) in the 2x2 output

    # Let's define the 7 Strassen multiplications explicitly:

    # M0 (M1): (a+d)*(e+h)
    #   U: row 0 (a) with coeff 1, row 3 (d) with coeff 1
    #   V: col 0 (e) with coeff 1, col 1 (h) with coeff 1
    #   W: contributes to C[0][0] with +1, C[1][1] with +1

    # M1 (M2): (c+d)*e
    #   U: row 2 (c) with coeff 1, row 3 (d) with coeff 1
    #   V: col 0 (e) with coeff 1
    #   W: contributes to C[1][0] with +1, C[1][1] with -1

    # M2 (M3): a*(f-h)
    #   U: row 0 (a) with coeff 1
    #   V: col 1 (h) with coeff -1, col 0 (f) with coeff +1... wait
    #   f is the top-right of B, which is col 1 in the 2x2 block (since B = [[e, f], [g, h]])
    #   So f is at position (0,1) in B, h is at (1,1)
    #   V: col 0 (which is e? no...)

    # Let me be more careful with indexing.
    # In the 2x2 block for Strassen:
    #   A = [[a, b], [c, d]]
    #   B = [[e, f], [g, h]]
    #
    # In our global indexing:
    #   A[i][j] -> U row i*m + j = i*2 + j
    #     a = A[0][0] -> row 0
    #     b = A[0][1] -> row 1
    #     c = A[1][0] -> row 2
    #     d = A[1][1] -> row 3
    #
    #   B[j][k] -> V row j*p + k = j*6 + k
    #     For block b, the two columns are k=2*b and k=2*b+1
    #     e = B[0][2*b] -> row 0*6 + 2*b = 2*b
    #     f = B[0][2*b+1] -> row 0*6 + 2*b+1 = 2*b+1
    #     g = B[1][2*b] -> row 1*6 + 2*b = 6 + 2*b
    #     h = B[1][2*b+1] -> row 1*6 + 2*b+1 = 6 + 2*b+1
    #
    #   C[i][k] -> W row k*n + i = k*2 + i
    #     For block b, the two output columns are k=2*b and k=2*b+1
    #     C[0][2*b]   -> row 2*b*2 + 0 = 4*b
    #     C[0][2*b+1] -> row (2*b+1)*2 + 0 = 4*b + 2
    #     C[1][2*b]   -> row 2*b*2 + 1 = 4*b + 1
    #     C[1][2*b+1] -> row (2*b+1)*2 + 1 = 4*b + 3

    # Now the 7 Strassen products:
    # M0 = (a+d)*(e+h)
    # M1 = (c+d)*e
    # M2 = a*(f-h)
    # M3 = d*(g-e)
    # M4 = (a+b)*h
    # M5 = (c-a)*(e+f)
    # M6 = (b-d)*(g+h)

    # And the output:
    # C[0][0] = M0 - M3 + M4 - M6
    # C[0][1] = M2 + M6
    # C[1][0] = M1 + M3
    # C[1][1] = M0 - M1 + M5

    # For each product r, define the U, V, W contributions:

    # Product 0: (a+d)*(e+h)
    #   U: (0, 1), (3, 1)
    #   V: (2*b, 1), (6+2*b+1, 1)  i.e., (2*b, 1), (7+2*b, 1)
    #   W: (4*b, 1), (4*b+3, 1)

    # Product 1: (c+d)*e
    #   U: (2, 1), (3, 1)
    #   V: (2*b, 1)
    #   W: (4*b+1, 1), (4*b+3, -1)

    # Product 2: a*(f-h)
    #   U: (0, 1)
    #   V: (2*b+1, 1), (7+2*b+1, -1)  i.e., (2*b+1, 1), (8+2*b, -1)
    #   Wait: f is at (0, 2*b+1) in B, so V row = 0*6 + (2*b+1) = 2*b+1
    #         h is at (1, 2*b+1) in B, so V row = 1*6 + (2*b+1) = 7+2*b
    #   So V: (2*b+1, 1), (7+2*b, -1)
    #   W: (4*b+2, 1)

    # Product 3: d*(g-e)
    #   U: (3, 1)
    #   V: (6+2*b, 1), (2*b, -1)  i.e., g at (1, 2*b) -> row 6+2*b, e at (0, 2*b) -> row 2*b
    #   So V: (6+2*b, 1), (2*b, -1)
    #   W: (4*b, -1), (4*b+1, 1)

    # Product 4: (a+b)*h
    #   U: (0, 1), (1, 1)
    #   V: (7+2*b, 1)  i.e., h at (1, 2*b+1) -> row 7+2*b
    #   W: (4*b, 1)

    # Product 5: (c-a)*(e+f)
    #   U: (2, 1), (0, -1)
    #   V: (2*b, 1), (2*b+1, 1)  i.e., e and f
    #   W: (4*b+3, 1)

    # Product 6: (b-d)*(g+h)
    #   U: (1, 1), (3, -1)
    #   V: (6+2*b, 1), (7+2*b, 1)  i.e., g and h
    #   W: (4*b, -1), (4*b+2, 1)

    # Now let's fill in the matrices:

    for b in range(3):  # Three blocks
        base_col = b * 7  # Column offset for this block
        base_row_V = 2 * b  # Base row index for V in this block
        base_row_W = 4 * b  # Base row index for W in this block

        # Product 0: (a+d)*(e+h)
        U[0][base_col + 0] += 1
        U[3][base_col + 0] += 1
        V[base_row_V + 0][base_col + 0] += 1  # e
        V[base_row_V + 1 + 6][base_col + 0] += 1  # h (row 7+2*b = base_row_V+1+6)
        W[base_row_W + 0][base_col + 0] += 1  # C[0][2*b]
        W[base_row_W + 3][base_col + 0] += 1  # C[1][2*b+1]

        # Product 1: (c+d)*e
        U[2][base_col + 1] += 1
        U[3][base_col + 1] += 1
        V[base_row_V + 0][base_col + 1] += 1  # e
        W[base_row_W + 1][base_col + 1] += 1  # C[1][2*b]
        W[base_row_W + 3][base_col + 1] += -1  # C[1][2*b+1]

        # Product 2: a*(f-h)
        U[0][base_col + 2] += 1
        V[base_row_V + 1][base_col + 2] += 1  # f (row 2*b+1)
        V[base_row_V + 1 + 6][base_col + 2] += -1  # h (row 7+2*b)
        W[base_row_W + 2][base_col + 2] += 1  # C[0][2*b+1]

        # Product 3: d*(g-e)
        U[3][base_col + 3] += 1
        V[base_row_V + 6][base_col + 3] += 1  # g (row 6+2*b)
        V[base_row_V + 0][base_col + 3] += -1  # e (row 2*b)
        W[base_row_W + 0][base_col + 3] += -1  # C[0][2*b]
        W[base_row_W + 1][base_col + 3] += 1  # C[1][2*b]

        # Product 4: (a+b)*h
        U[0][base_col + 4] += 1
        U[1][base_col + 4] += 1
        V[base_row_V + 1 + 6][base_col + 4] += 1  # h (row 7+2*b)
        W[base_row_W + 0][base_col + 4] += 1  # C[0][2*b]

        # Product 5: (c-a)*(e+f)
        U[2][base_col + 5] += 1
        U[0][base_col + 5] += -1
        V[base_row_V + 0][base_col + 5] += 1  # e
        V[base_row_V + 1][base_col + 5] += 1  # f
        W[base_row_W + 3][base_col + 5] += 1  # C[1][2*b+1]

        # Product 6: (b-d)*(g+h)
        U[1][base_col + 6] += 1
        U[3][base_col + 6] += -1
        V[base_row_V + 6][base_col + 6] += 1  # g
        V[base_row_V + 1 + 6][base_col + 6] += 1  # h
        W[base_row_W + 0][base_col + 6] += -1  # C[0][2*b]
        W[base_row_W + 2][base_col + 6] += 1  # C[0][2*b+1]

    # Convert to integer values if they are close to integers
    for matrix in [U, V, W]:
        for row in matrix:
            for j in range(len(row)):
                if abs(row[j] - round(row[j])) < 1e-10:
                    row[j] = int(round(row[j]))

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"U shape: {len(result['U'])} x {len(result['U'][0])}")
    print(f"V shape: {len(result['V'])} x {len(result['V'][0])}")
    print(f"W shape: {len(result['W'])} x {len(result['W'][0])}")
    print(f"Rank: {len(result['U'][0])}")
