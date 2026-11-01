"""Matrix multiplication 2,2,14 — rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds; leave margin before 150s kill

    n, m, p = 2, 2, 14

    # Strassen decomposition for 2x2 x 2x2 (rank 7)
    # A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]]
    # 7 products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Build the 7 rank-1 terms for one 2x2 block
    # U indices: i*m + j = i*2 + j for i in {0,1}, j in {0,1} -> 0,1,2,3
    # V indices: j*p_block + k where p_block=2: j*2+k -> 0,1,2,3
    # W indices: k*n + i = k*2 + i for k in {0,1}, i in {0,1} -> 0,1,2,3

    # For each of the 7 Strassen products, define:
    # (a_coeffs, b_coeffs, c_coeffs)
    strassen_products = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1},
         {(0,1): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1},
         {(0,0): 1, (1,0): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1},
         {(0,0): -1, (1,0): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
    ]

    # Apply to 7 column-pairs (since 14 = 7*2)
    rank = 7 * 7  # 49

    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49

    for block in range(7):
        for s in range(7):
            col = block * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_products[s]

            # U: A indices (i,j) -> U[i*m+j, col]
            for (i, j), coeff in a_coeffs.items():
                U[i * m + j][col] = coeff

            # V: B indices (j,k_local) -> global column = block*2 + k_local
            # V row index = j * p + (block*2 + k_local)
            for (j, k_local), coeff in b_coeffs.items():
                k_global = block * 2 + k_local
                V[j * p + k_global][col] = coeff

            # W: C indices (k_local,i) -> global column = block*2 + k_local
            # W row index = (block*2 + k_local) * n + i
            for (k_local, i), coeff in c_coeffs.items():
                k_global = block * 2 + k_local
                W[k_global * n + i][col] = coeff

    # Quick validation check
    if time.time() - start_time < DEADLINE:
        import random
        random.seed(42)
        ok = True
        for _ in range(5):
            A = [[random.randint(-5, 5) for _ in range(2)] for _ in range(2)]
            B = [[random.randint(-5, 5) for _ in range(14)] for _ in range(2)]
            for i in range(2):
                for k in range(14):
                    expected = 0
                    for j in range(2):
                        expected += A[i][j] * B[j][k]
                    actual = 0
                    for c in range(rank):
                        sum_U = 0
                        for j in range(m):
                            sum_U += U[i * m + j][c] * A[i][j]
                        sum_V = 0
                        for j in range(m):
                            sum_V += V[j * p + k][c] * B[j][k]
                        actual += W[k * n + i][c] * sum_U * sum_V
                    if abs(actual - expected) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
        if not ok:
            # Fall back to schoolbook
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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
