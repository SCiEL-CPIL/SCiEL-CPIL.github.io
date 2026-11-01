"""Matrix multiplication 2x4x8 - rank 56 via Strassen blocks."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    # Strassen 2x2x2 decomposition
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    strassen = [
        # (u_dict, v_dict, w_dict)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    col = 0
    for h in range(2):  # 2 column-pairs of A (j dimension split into 2 blocks of 2)
        for c in range(4):  # 4 column-pairs of B (k dimension split into 4 blocks of 2)
            for s in range(7):
                u_c, v_c, w_c = strassen[s]

                for idx, val in u_c.items():
                    i = idx // 2
                    j_l = idx % 2
                    j_g = 2 * h + j_l
                    U[i * m + j_g][col] += val

                for idx, val in v_c.items():
                    j_l = idx // 2
                    k_l = idx % 2
                    j_g = 2 * h + j_l
                    k_g = 2 * c + k_l
                    V[j_g * p + k_g][col] += val

                for idx, val in w_c.items():
                    i = idx // 2
                    k_l = idx % 2
                    k_g = 2 * c + k_l
                    W[k_g * n + i][col] += val

                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
