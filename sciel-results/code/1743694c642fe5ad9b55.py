"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen on 2x2x2 blocks
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # k corresponds to the half of the inner dimension (0 or 1)
    # l corresponds to the block of the output dimension (0, 1, 2)
    for k in range(2):
        for l in range(3):
            # M1: (X11 + X22)(Y11 + Y22)
            # U: A[0, 2k] + A[1, 2k+1]
            U[0 * m + 2 * k][col] += 1
            U[1 * m + 2 * k + 1][col] += 1
            # V: B[2k, 2l] + B[2k+1, 2l+1]
            V[2 * k * p + 2 * l][col] += 1
            V[(2 * k + 1) * p + 2 * l + 1][col] += 1
            # W: Z11(+1), Z22(+1)
            W[2 * l * n + 0][col] += 1
            W[(2 * l + 1) * n + 1][col] += 1
            col += 1

            # M2: (X21 + X22)Y11
            # U: A[1, 2k] + A[1, 2k+1]
            U[1 * m + 2 * k][col] += 1
            U[1 * m + 2 * k + 1][col] += 1
            # V: B[2k, 2l]
            V[2 * k * p + 2 * l][col] += 1
            # W: Z21(+1)
            W[2 * l * n + 1][col] += 1
            col += 1

            # M3: X11(Y12 - Y22)
            # U: A[0, 2k]
            U[0 * m + 2 * k][col] += 1
            # V: B[2k, 2l+1] - B[2k+1, 2l+1]
            V[2 * k * p + 2 * l + 1][col] += 1
            V[(2 * k + 1) * p + 2 * l + 1][col] -= 1
            # W: Z12(+1)
            W[(2 * l + 1) * n + 0][col] += 1
            col += 1

            # M4: X22(Y21 - Y11)
            # U: A[1, 2k+1]
            U[1 * m + 2 * k + 1][col] += 1
            # V: B[2k+1, 2l] - B[2k, 2l]
            V[(2 * k + 1) * p + 2 * l][col] += 1
            V[2 * k * p + 2 * l][col] -= 1
            # W: Z11(+1), Z21(+1)
            W[2 * l * n + 0][col] += 1
            W[2 * l * n + 1][col] += 1
            col += 1

            # M5: (X11 + X12)Y22
            # U: A[0, 2k] + A[0, 2k+1]
            U[0 * m + 2 * k][col] += 1
            U[0 * m + 2 * k + 1][col] += 1
            # V: B[2k+1, 2l+1]
            V[(2 * k + 1) * p + 2 * l + 1][col] += 1
            # W: Z11(-1), Z12(+1)
            W[2 * l * n + 0][col] -= 1
            W[(2 * l + 1) * n + 0][col] += 1
            col += 1

            # M6: (X21 - X11)(Y11 + Y12)
            # U: A[1, 2k] - A[0, 2k]
            U[1 * m + 2 * k][col] += 1
            U[0 * m + 2 * k][col] -= 1
            # V: B[2k, 2l] + B[2k, 2l+1]
            V[2 * k * p + 2 * l][col] += 1
            V[2 * k * p + 2 * l + 1][col] += 1
            # W: Z22(+1)
            W[(2 * l + 1) * n + 1][col] += 1
            col += 1

            # M7: (X12 - X22)(Y21 + Y22)
            # U: A[0, 2k+1] - A[1, 2k+1]
            U[0 * m + 2 * k + 1][col] += 1
            U[1 * m + 2 * k + 1][col] -= 1
            # V: B[2k+1, 2l] + B[2k+1, 2l+1]
            V[(2 * k + 1) * p + 2 * l][col] += 1
            V[(2 * k + 1) * p + 2 * l + 1][col] += 1
            # W: Z11(+1), Z12(-1)
            W[2 * l * n + 0][col] += 1
            W[(2 * l + 1) * n + 0][col] -= 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
