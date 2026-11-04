"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44

    # Initialize U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    cols = []

    def add_term(u_entries, v_entries, w_entries):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        cols.append((u_vec, v_vec, w_vec))

    def add_strassen(a, b, c, d, e, f, g, h, w11, w12, w21, w22):
        # M1 = (a+d)(e+h)
        add_term([(a, 1), (d, 1)], [(e, 1), (h, 1)], [(w11, 1), (w22, 1)])
        # M2 = (c+d)e
        add_term([(c, 1), (d, 1)], [(e, 1)], [(w21, 1), (w22, -1)])
        # M3 = a(f-h)
        add_term([(a, 1)], [(f, 1), (h, -1)], [(w12, 1), (w22, 1)])
        # M4 = d(g-e)
        add_term([(d, 1)], [(g, 1), (e, -1)], [(w11, 1), (w21, 1)])
        # M5 = (a+b)h
        add_term([(a, 1), (b, 1)], [(h, 1)], [(w12, 1), (w11, -1)])
        # M6 = (c-a)(e+f)
        add_term([(c, 1), (a, -1)], [(e, 1), (f, 1)], [(w22, 1)])
        # M7 = (b-d)(g+h)
        add_term([(b, 1), (d, -1)], [(g, 1), (h, 1)], [(w11, 1)])

    def add_rank4(a, b, c, d, e, f, w1, w2):
        add_term([(a, 1)], [(c, 1)], [(w1, 1)])
        add_term([(b, 1)], [(e, 1)], [(w1, 1)])
        add_term([(a, 1)], [(d, 1)], [(w2, 1)])
        add_term([(b, 1)], [(f, 1)], [(w2, 1)])

    # Indices for A (3x4)
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # Row 2: 8, 9, 10, 11

    # Indices for B (4x4)
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # Row 2: 8, 9, 10, 11
    # Row 3: 12, 13, 14, 15

    # Indices for C (3x4) -> W (k*3+i)
    # Col 0: 0, 1, 2
    # Col 1: 3, 4, 5
    # Col 2: 6, 7, 8
    # Col 3: 9, 10, 11

    # Case 1: A_LL (R0,1 C0,1) * B_LL1 (R0,1 C0,1) -> C(R0,1 C0,1)
    add_strassen(
        0, 1, 4, 5,   # A
        0, 1, 4, 5,   # B
        0, 3, 1, 4    # W: C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4
    )

    # Case 2: A_LL (R0,1 C0,1) * B_LL2 (R0,1 C2,3) -> C(R0,1 C2,3)
    add_strassen(
        0, 1, 4, 5,   # A
        2, 3, 6, 7,   # B
        6, 9, 7, 10   # W: C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10
    )

    # Case 3: A_LR (R2 C0,1) * B_LL1 (R0,1 C0,1) -> C(R2 C0,1)
    add_rank4(
        8, 9,         # A
        0, 1, 4, 5,   # B
        2, 5          # W: C[2,0]=2, C[2,1]=5
    )

    # Case 4: A_LR (R2 C0,1) * B_LL2 (R0,1 C2,3) -> C(R2 C2,3)
    add_rank4(
        8, 9,         # A
        2, 3, 6, 7,   # B
        8, 11         # W: C[2,2]=8, C[2,3]=11
    )

    # Case 5: A_RL (R0,1 C2,3) * B_RL1 (R2,3 C0,1) -> C(R0,1 C0,1)
    add_strassen(
        2, 3, 6, 7,   # A
        8, 9, 12, 13, # B
        0, 3, 1, 4    # W: C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4
    )

    # Case 6: A_RL (R0,1 C2,3) * B_RL2 (R2,3 C2,3) -> C(R0,1 C2,3)
    add_strassen(
        2, 3, 6, 7,   # A
        10, 11, 14, 15, # B
        6, 9, 7, 10   # W: C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10
    )

    # Case 7: A_RR (R2 C2,3) * B_RL1 (R2,3 C0,1) -> C(R2 C0,1)
    add_rank4(
        10, 11,       # A
        8, 9, 12, 13, # B
        2, 5          # W: C[2,0]=2, C[2,1]=5
    )

    # Case 8: A_RR (R2 C2,3) * B_RL2 (R2,3 C2,3) -> C(R2 C2,3)
    add_rank4(
        10, 11,       # A
        10, 11, 14, 15, # B
        8, 11         # W: C[2,2]=8, C[2,3]=11
    )

    # Fill U, V, W
    for r in range(rank):
        u_vec, v_vec, w_vec = cols[r]
        for i in range(n * m):
            U[i][r] = u_vec[i]
        for j in range(m * p):
            V[j][r] = v_vec[j]
        for k in range(p * n):
            W[k][r] = w_vec[k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
