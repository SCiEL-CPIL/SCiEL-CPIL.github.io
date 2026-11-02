"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 3 blocks * 7 (Strassen) = 21
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen factors for 2x2 multiplication
    # U_s[i, j], V_s[j, k], W_s[k, i]
    # Note: W index in problem is k*n + i. Here k is local k' in {0,1}, i in {0,1}.

    # M1: (A00+A11)(B00+B11)
    U0 = [[1, 0], [0, 1]]
    V0 = [[1, 0], [0, 1]]
    W0 = [[1, 0], [0, 1]]

    # M2: (A10+A11)B00
    U1 = [[0, 0], [1, 1]]
    V1 = [[1, 0], [0, 0]]
    W1 = [[0, 1], [0, -1]]

    # M3: A00(B01-B11)
    U2 = [[1, 0], [0, 0]]
    V2 = [[0, 1], [0, -1]]
    W2 = [[0, 0], [1, 1]]

    # M4: A11(B10-B00)
    U3 = [[0, 0], [0, 1]]
    V3 = [[-1, 0], [1, 0]]
    W3 = [[1, 1], [0, 0]]

    # M5: (A00+A01)B11
    U4 = [[1, 1], [0, 0]]
    V4 = [[0, 0], [0, 1]]
    W4 = [[-1, 0], [1, 0]]

    # M6: (A10-A00)(B00+B01)
    U5 = [[-1, 0], [1, 0]]
    V5 = [[1, 1], [0, 0]]
    W5 = [[0, 0], [0, 1]]

    # M7: (A01-A11)(B10+B11)
    U6 = [[0, 1], [0, -1]]
    V6 = [[0, 0], [1, 1]]
    W6 = [[1, 0], [0, 0]]

    factors = [
        (U0, V0, W0),
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6)
    ]

    for q in range(3):
        for s in range(7):
            col = 7 * q + s
            Us, Vs, Ws = factors[s]

            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][col] = Us[i][j]

            # Fill V
            for j in range(2):
                for kp in range(2):
                    k_global = 2 * q + kp
                    V[j * 6 + k_global][col] = Vs[j][kp]

            # Fill W
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * q + kp
                    W[k_global * 2 + i][col] = Ws[kp][i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
