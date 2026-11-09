import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Strassen 2x2x2 decomposition terms
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),  # M1
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),               # M2
        ([(0,1)], [(1,1),(3,-1)], [(1,1)]),              # M3
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),        # M4
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),        # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),        # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(3,-1)]), # M7
    ]

    rank = 0
    U_dict = {}
    V_dict = {}
    W_dict = {}

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal rank
        col = rank
        for (idx, coeff) in u_coeffs:
            U_dict[(idx, col)] = U_dict.get((idx, col), 0) + coeff
        for (idx, coeff) in v_coeffs:
            V_dict[(idx, col)] = V_dict.get((idx, col), 0) + coeff
        for (idx, coeff) in w_coeffs:
            W_dict[(idx, col)] = W_dict.get((idx, col), 0) + coeff
        rank += 1

    # Add Strassen terms for 8 subproblems covering k=0,1,2,3
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                if time.time() > deadline:
                    break
                for (u_local, v_local, w_local) in strassen_terms:
                    u_global = []
                    for (local_idx, coeff) in u_local:
                        li = local_idx // 2
                        lj = local_idx % 2
                        gi = bi * 2 + li
                        gj = bj * 2 + lj
                        u_global.append((gi * m + gj, coeff))

                    v_global = []
                    for (local_idx, coeff) in v_local:
                        lj = local_idx // 2
                        lk = local_idx % 2
                        gj = bj * 2 + lj
                        gk = bk * 2 + lk
                        v_global.append((gj * p + gk, coeff))

                    w_global = []
                    for (local_idx, coeff) in w_local:
                        lk = local_idx // 2
                        li = local_idx % 2
                        gk = bk * 2 + lk
                        gi = bi * 2 + li
                        w_global.append((gk * n + gi, coeff))

                    add_term(u_global, v_global, w_global)

    # Add schoolbook terms for k=4
    for i in range(n):
        for j in range(m):
            if time.time() > deadline:
                break
            u_idx = i * m + j
            v_idx = j * p + 4
            w_idx = 4 * n + i
            add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    # Convert to matrices
    nU = n * m
    nV = m * p
    nW = p * n

    U_mat = [[0] * rank for _ in range(nU)]
    V_mat = [[0] * rank for _ in range(nV)]
    W_mat = [[0] * rank for _ in range(nW)]

    for (idx, col), val in U_dict.items():
        U_mat[idx][col] = val
    for (idx, col), val in V_dict.items():
        V_mat[idx][col] = val
    for (idx, col), val in W_dict.items():
        W_mat[idx][col] = val

    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
