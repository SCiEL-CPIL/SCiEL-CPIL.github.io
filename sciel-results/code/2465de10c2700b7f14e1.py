"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use a 3x3 block partition of the 6x6 matrices.
    # Each block is 2x2, so each block multiplication uses Strassen (rank 7).
    # There are 3*3 = 9 block positions for (i,j) but we need to handle all
    # (i, s, j) triples for the tensor: 3*3*3 = 27 block multiplications.
    # Total rank = 27 * 7 = 189.
    rank = 189

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(3):
        for s in range(3):
            for j in range(3):
                # Global indices for the 2x2 block of A at rows (2i, 2i+1), cols (2s, 2s+1)
                a = (2 * i) * 6 + (2 * s)
                b = (2 * i) * 6 + (2 * s + 1)
                c = (2 * i + 1) * 6 + (2 * s)
                d = (2 * i + 1) * 6 + (2 * s + 1)

                # Global indices for the 2x2 block of B at rows (2s, 2s+1), cols (2j, 2j+1)
                e = (2 * s) * 6 + (2 * j)
                f = (2 * s) * 6 + (2 * j + 1)
                g = (2 * s + 1) * 6 + (2 * j)
                h = (2 * s + 1) * 6 + (2 * j + 1)

                # Global indices for the 2x2 block of C at rows (2i, 2i+1), cols (2j, 2j+1)
                # W index convention: W[k*n + r, t] where k is col of C, r is row of C
                # So C[row, col] -> W[col * 6 + row]
                w00 = (2 * j) * 6 + (2 * i)
                w01 = (2 * j + 1) * 6 + (2 * i)
                w10 = (2 * j) * 6 + (2 * i + 1)
                w11 = (2 * j + 1) * 6 + (2 * i + 1)

                # Strassen's 7 products for this 2x2 block multiplication
                # M1 = (a+d)(e+h)
                U[a][col] = 1
                U[d][col] = 1
                V[e][col] = 1
                V[h][col] = 1
                W[w00][col] = 1
                W[w11][col] = 1
                col += 1

                # M2 = (c+d)e
                U[c][col] = 1
                U[d][col] = 1
                V[e][col] = 1
                W[w10][col] = 1
                W[w11][col] = -1
                col += 1

                # M3 = a(f-h)
                U[a][col] = 1
                V[f][col] = 1
                V[h][col] = -1
                W[w01][col] = 1
                W[w11][col] = 1
                col += 1

                # M4 = d(g-e)
                U[d][col] = 1
                V[g][col] = 1
                V[e][col] = -1
                W[w00][col] = 1
                W[w10][col] = 1
                col += 1

                # M5 = (a+b)h
                U[a][col] = 1
                U[b][col] = 1
                V[h][col] = 1
                W[w00][col] = -1
                W[w01][col] = 1
                col += 1

                # M6 = (c-a)(e+f)
                U[c][col] = 1
                U[a][col] = -1
                V[e][col] = 1
                V[f][col] = 1
                W[w11][col] = 1
                col += 1

                # M7 = (b-d)(g+h)
                U[b][col] = 1
                U[d][col] = -1
                V[g][col] = 1
                V[h][col] = 1
                W[w00][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
