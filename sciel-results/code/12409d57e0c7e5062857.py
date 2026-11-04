"""Optimized matrix multiplication for 4x5 by 5x5 using Strassen on 4x4 blocks."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = 85
    U = [[0.0] * rank for _ in range(n * m)]  # 20 x 85
    V = [[0.0] * rank for _ in range(m * p)]  # 25 x 85
    W = [[0.0] * rank for _ in range(p * n)]  # 20 x 85

    col = 0

    # Strassen 4x4 block: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # Strassen 7 multiplications, each 2x2 block
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each of the 7 multiplications, we do 2x2 = 4 scalar multiplications
    # A block indices: (I, J) where I, J in {0,1} -> A[I, J] is 2x2
    # A[I, J] covers rows 2I, 2I+1 and cols 2J, 2J+1
    # B[I, J] covers rows 2I, 2I+1 and cols 2J, 2J+1

    # Precompute the Strassen combination coefficients
    # For each of the 7 mults, we need the linear combinations for A and B
    # A_combinations: list of 7 lists, each containing (coeff, row, col) for A
    # B_combinations: list of 7 lists, each containing (coeff, row, col) for B

    A_combos = [
        # M1: A00 + A11
        [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
        # M2: A10 + A11
        [(1, 1, 0), (1, 1, 1)],
        # M3: A00
        [(1, 0, 0), (1, 0, 1)],
        # M4: A11
        [(1, 1, 1)],
        # M5: A00 + A01
        [(1, 0, 0), (1, 0, 1)],
        # M6: A10 - A00
        [(-1, 0, 0), (-1, 0, 1), (1, 1, 0), (1, 1, 1)],
        # M7: A01 - A11
        [(1, 0, 1), (-1, 1, 1)],
    ]

    B_combos = [
        # M1: B00 + B11
        [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
        # M2: B00
        [(1, 0, 0), (1, 0, 1)],
        # M3: B10 - B11
        [(1, 1, 0), (-1, 1, 1)],
        # M4: B01 - B10
        [(-1, 1, 0), (1, 0, 1)],
        # M5: B11
        [(1, 1, 1)],
        # M6: B00 + B01
        [(1, 0, 0), (1, 0, 1)],
        # M7: B10 + B11
        [(1, 1, 0), (1, 1, 1)],
    ]

    # For each multiplication k (0..6), and each 2x2 element (a,b) of the result:
    # We add a rank-1 term
    # U contribution: for each (coeff_a, ar, ac) in A_combos[k], U[ar*5 + ac][col] += coeff_a
    # V contribution: for each (coeff_b, br, bc) in B_combos[k], V[br*5 + bc][col] += coeff_b
    # W contribution: C[2I+a, 2J+b] gets contribution from M_k with coefficient S_k[I,J]
    # But wait, the 2x2 multiplication gives a 2x2 result M_k[2x2]
    # C[I', J'] = sum_k S_k[I', J'] * M_k (as 2x2 blocks)
    # So C[2I'+a, 2J'+b] = sum_k S_k[I', J'] * M_k[a, b]

    # Strassen output coefficients S_k[I', J'] for 2x2 blocks:
    # I', J' in {0, 1}
    S = [
        # M1
        [[1, 0], [0, 1]],
        # M2
        [[0, 0], [1, 0]],
        # M3
        [[0, 1], [0, 1]],
        # M4
        [[1, 0], [0, 1]],
        # M5
        [[-1, 1], [0, 0]],
        # M6
        [[0, 0], [1, 1]],
        # M7
        [[1, 0], [0, 0]],
    ]

    for k in range(7):
        for a in range(2):  # row within 2x2 block
            for b in range(2):  # col within 2x2 block
                # This is the (a,b) element of M_k
                # U part
                for coeff_a, ar, ac in A_combos[k]:
                    U[ar * 5 + ac][col] += coeff_a
                # V part
                for coeff_b, br, bc in B_combos[k]:
                    V[br * 5 + bc][col] += coeff_b
                # W part: for each (I', J') block in C, if S[k][I'][J'] != 0
                for I_prime in range(2):
                    for J_prime in range(2):
                        coeff_c = S[k][I_prime][J_prime]
                        if coeff_c != 0:
                            # C row = 2*I_prime + a, C col = 2*J_prime + b
                            # W index = (p - 1 - Ccol) * n + Crow? No.
                            # Convention: W[k*n + i, r] where k is output col, i is output row?
                            # Task says: W[k*n+i,r]
                            # Output C is n x p = 4 x 5
                            # So output row i in [0,4), output col k in [0,5)
                            # W index = k * n + i
                            crow = 2 * I_prime + a
                            ccol = 2 * J_prime + b
                            w_idx = ccol * n + crow
                            W[w_idx][col] += coeff_c
                col += 1

    # Now handle the remaining terms:
    # A[0:4, 4] * B[4, 0:5] -> C[0:4, 0:5] (16 terms)
    # A[4, 0:5] * B[0:4, 0:5] -> C[4, 0:5] (20 terms) - wait, n=4 so no row 4 in C
    # Let me re-examine: A is 4x5, B is 5x5, C is 4x5
    # C[i, j] = sum_l A[i, l] * B[l, j]
    # We handled l in [0,4) for i in [0,4), j in [0,4) via Strassen
    # Remaining:
    # 1. i in [0,4), l=4, j in [0,5): A[i,4] * B[4,j] -> 4*5=20 terms
    # 2. i in [0,4), l in [0,4), j=4: A[i,l] * B[l,4] -> 4*4=16 terms
    # But wait, in Strassen we covered i in [0,4), l in [0,4), j in [0,4)
    # So we need:
    # a) i in [0,4), l=4, j in [0,5): 4*5=20 terms
    # b) i in [0,4), l in [0,4), j=4: 4*4=16 terms
    # Total remaining: 36 terms. 49 + 36 = 85. Correct.

    # Part a: A[i, 4] * B[4, j] for i in [0,4), j in [0,5)
    for i in range(4):
        for j in range(5):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + j][col] = 1
            W[j * n + i][col] = 1
            col += 1

    # Part b: A[i, l] * B[l, 4] for i in [0,4), l in [0,4)
    for i in range(4):
        for l in range(4):
            U[i * 5 + l][col] = 1
            V[l * 5 + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
