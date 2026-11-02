"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56

    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56

    # Seven rank-1 tensors for the 2x2x2 Strassen core.
    # Each: (U_part, V_part, W_part) with local indices.
    # U_part entries: (i, j, coeff) where i,j in {0,1}
    # V_part entries: (j, k, coeff) where j in {0,1}, k in {0,1}
    # W_part entries: (i, k, coeff) where i in {0,1}, k in {0,1}

    S = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],         [(1,0,1),(1,1,-1)]),
        ([(0,0,1)],         [(0,1,1),(1,1,-1)],[(0,1,1),(1,1,1)]),
        ([(1,1,1)],         [(1,0,1),(0,0,-1)],[(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)],         [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)],[(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)],[(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    for t in range(num_blocks):
        k0 = 2 * t  # global column offset for this 2-col block
        for s in range(7):
            col = t * 7 + s
            u_part, v_part, w_part = S[s]

            for (i, j, c) in u_part:
                U[i * m + j][col] += c

            for (j, kl, c) in v_part:
                V[j * p + k0 + kl][col] += c

            for (i, kl, c) in w_part:
                W[(k0 + kl) * n + i][col] += c

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
