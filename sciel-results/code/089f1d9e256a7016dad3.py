"""Matrix multiplication 5x5x5 with block decomposition: Strassen(2x2) + schoolbook(3x3) + cross terms."""

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    U_terms = []
    V_terms = []
    W_terms = []

    def make_vec(length, idx_val_pairs):
        vec = [0] * length
        for idx, val in idx_val_pairs:
            vec[idx] = val
        return vec

    def add_rank1(u_vec, v_vec, w_vec):
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)

    # --- Strassen 2x2x2 for block (rows 0,1) x (cols 0,1) ---
    # A idx = i*5+j, B idx = j*5+k, C idx = k*5+i
    # p1 = (a00+a11)(b00+b11) -> c00 + c11
    add_rank1(
        make_vec(nm, [(0, 1), (6, 1)]),
        make_vec(mp, [(0, 1), (6, 1)]),
        make_vec(pn, [(0, 1), (6, 1)])
    )
    # p2 = (a10+a11)(b00) -> c10 + (-c11)
    add_rank1(
        make_vec(nm, [(5, 1), (6, 1)]),
        make_vec(mp, [(0, 1)]),
        make_vec(pn, [(5, 1), (6, -1)])
    )
    # p3 = a00(b10-b11) -> c01 + c11
    add_rank1(
        make_vec(nm, [(0, 1)]),
        make_vec(mp, [(5, 1), (6, -1)]),
        make_vec(pn, [(1, 1), (6, 1)])
    )
    # p4 = a11(b01-b10) -> c00 + c10
    add_rank1(
        make_vec(nm, [(6, 1)]),
        make_vec(mp, [(1, 1), (5, -1)]),
        make_vec(pn, [(0, 1), (5, 1)])
    )
    # p5 = (a00+a01)(b11) -> -c00 + c01
    add_rank1(
        make_vec(nm, [(0, 1), (1, 1)]),
        make_vec(mp, [(6, 1)]),
        make_vec(pn, [(0, -1), (1, 1)])
    )
    # p6 = (a10-a00)(b00+b10) -> c11
    add_rank1(
        make_vec(nm, [(5, 1), (0, -1)]),
        make_vec(mp, [(0, 1), (5, 1)]),
        make_vec(pn, [(6, 1)])
    )
    # p7 = (a01-a11)(b01+b11) -> c00
    add_rank1(
        make_vec(nm, [(1, 1), (6, -1)]),
        make_vec(mp, [(1, 1), (6, 1)]),
        make_vec(pn, [(0, 1)])
    )

    # --- Schoolbook 3x3x3 for block (rows 2,3,4) x (cols 2,3,4) ---
    for i in range(3):
        for j in range(3):
            for k in range(3):
                gi, gj, gk = 2 + i, 2 + j, 2 + k
                add_rank1(
                    make_vec(nm, [(gi * 5 + gj, 1)]),
                    make_vec(mp, [(gj * 5 + gk, 1)]),
                    make_vec(pn, [(gk * 5 + gi, 1)])
                )

    # --- Cross-block: A[0:2,:]*B[:,2:5] -> C[0:2,2:5] (all-ones 2x5x3, rank 1) ---
    add_rank1(
        make_vec(nm, [(i * 5 + j, 1) for i in range(2) for j in range(5)]),
        make_vec(mp, [(j * 5 + k, 1) for j in range(5) for k in range(2, 5)]),
        make_vec(pn, [(k * 5 + i, 1) for k in range(2, 5) for i in range(2)])
    )

    # --- Cross-block: A[2:5,:]*B[:,0:2] -> C[2:5,0:2] (all-ones 3x5x2, rank 1) ---
    add_rank1(
        make_vec(nm, [(i * 5 + j, 1) for i in range(2, 5) for j in range(5)]),
        make_vec(mp, [(j * 5 + k, 1) for j in range(5) for k in range(2)]),
        make_vec(pn, [(k * 5 + i, 1) for k in range(2) for i in range(2, 5)])
    )

    # --- Cross-block: A[0:2,2:5]*B[2:5,0:2] -> C[0:2,0:2] (all-ones 2x3x2, rank 1) ---
    add_rank1(
        make_vec(nm, [(i * 5 + j, 1) for i in range(2) for j in range(2, 5)]),
        make_vec(mp, [(j * 5 + k, 1) for j in range(2, 5) for k in range(2)]),
        make_vec(pn, [(k * 5 + i, 1) for k in range(2) for i in range(2)])
    )

    # --- Cross-block: A[2:5,0:2]*B[0:2,2:5] -> C[2:5,2:5] (all-ones 3x2x3, rank 1) ---
    add_rank1(
        make_vec(nm, [(i * 5 + j, 1) for i in range(2, 5) for j in range(2)]),
        make_vec(mp, [(j * 5 + k, 1) for j in range(2) for k in range(2, 5)]),
        make_vec(pn, [(k * 5 + i, 1) for k in range(2, 5) for i in range(2, 5)])
    )

    rank = len(U_terms)
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    for c in range(rank):
        for idx in range(nm):
            if U_terms[c][idx] != 0:
                U[idx][c] = U_terms[c][idx]
        for idx in range(mp):
            if V_terms[c][idx] != 0:
                V[idx][c] = V_terms[c][idx]
        for idx in range(pn):
            if W_terms[c][idx] != 0:
                W[idx][c] = W_terms[c][idx]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
