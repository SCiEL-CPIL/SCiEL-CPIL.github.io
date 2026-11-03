import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 8 blocks of 2x2x2 using Strassen (rank 7 each) -> 56
    # 4x1x4 part (rank 16) -> 16
    # 4x4x1 part (rank 16) -> 16
    # 4x1x1 part (rank 4) -> 4
    # Total: 92
    rank = 92
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        if col >= rank:
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
        col += 1

    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[c00, c01], [c10, c11]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Part 1: 4x4x4 (indices i,j,k in 0..3)
    # Split into 8 blocks of 2x2x2
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                # Local indices 0,1 map to global 2*i1+i, 2*j1+j, 2*k1+k
                for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                    u_indices = []
                    u_vals = []
                    for li in range(2):
                        for lj in range(2):
                            val = u_coeffs[li * 2 + lj]
                            if val != 0:
                                gi = 2 * i1 + li
                                gj = 2 * j1 + lj
                                u_indices.append(gi * m + gj)
                                u_vals.append(val)
                    
                    v_indices = []
                    v_vals = []
                    for lj in range(2):
                        for lk in range(2):
                            val = v_coeffs[lj * 2 + lk]
                            if val != 0:
                                gj = 2 * j1 + lj
                                gk = 2 * k1 + lk
                                v_indices.append(gj * p + gk)
                                v_vals.append(val)
                    
                    w_indices = []
                    w_vals = []
                    for li in range(2):
                        for lk in range(2):
                            val = w_coeffs[li * 2 + lk]
                            if val != 0:
                                gi = 2 * i1 + li
                                gk = 2 * k1 + lk
                                w_indices.append(gk * n + gi)
                                w_vals.append(val)
                    
                    add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)

    # Part 2: 4x1x4 (A[i,4], B[4,k], C[i,k]) for i in 0..3, k in 0..3
    for i in range(4):
        for k in range(4):
            u_indices = [i * m + 4]
            u_vals = [1]
            v_indices = [4 * p + k]
            v_vals = [1]
            w_indices = [k * n + i]
            w_vals = [1]
            add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)

    # Part 3: 4x4x1 (A[i,j], B[j,4], C[i,4]) for i in 0..3, j in 0..3
    for i in range(4):
        for j in range(4):
            u_indices = [i * m + j]
            u_vals = [1]
            v_indices = [j * p + 4]
            v_vals = [1]
            w_indices = [4 * n + i]
            w_vals = [1]
            add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)

    # Part 4: 4x1x1 (A[i,4], B[4,4], C[i,4]) for i in 0..3
    for i in range(4):
        u_indices = [i * m + 4]
        u_vals = [1]
        v_indices = [4 * p + 4]
        v_vals = [1]
        w_indices = [4 * n + i]
        w_vals = [1]
        add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
