import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    rank = 88

    u_rows = n * m  # 12
    v_rows = m * p  # 32
    w_rows = p * n  # 24

    U = [[0] * rank for _ in range(u_rows)]
    V = [[0] * rank for _ in range(v_rows)]
    W = [[0] * rank for _ in range(w_rows)]

    col = 0

    # Part 1: i=0, schoolbook 1x4x8 = 32 terms
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Part 2: i=1,2 using Strassen 2x2x2 blocks
    # Strassen local factors for 2x2x2
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_U = [
        [1, 0, 1, 0], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0], # M6: c-a
        [0, 1, 0, -1]  # M7: b-d
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1], # M3: f-h
        [-1, 0, 1, 0], # M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1: C00+1, C11+1
        [0, 0, 1, -1], # M2: C10+1, C11-1
        [0, 1, 0, 1], # M3: C01+1, C11+1
        [1, 0, 1, 0], # M4: C00+1, C10+1
        [-1, 1, 0, 0], # M5: C00-1, C01+1
        [0, 0, 0, 1], # M6: C11+1
        [1, 0, 0, 0]  # M7: C00+1
    ]

    # Iterate over j_blocks (2) and k_blocks (4)
    for j_block in range(m // 2):
        j_start = j_block * 2
        for k_block in range(p // 2):
            k_start = k_block * 2
            for t in range(7):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

                # U part: rows i=1,2
                for l in range(4):
                    val = strassen_U[t][l]
                    if val != 0:
                        local_i_prime = l // 2
                        local_j_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_j = j_start + local_j_prime
                        U[global_i * m + global_j][col] += val

                # V part
                for l in range(4):
                    val = strassen_V[t][l]
                    if val != 0:
                        local_j_prime = l // 2
                        local_k_prime = l % 2
                        global_j = j_start + local_j_prime
                        global_k = k_start + local_k_prime
                        V[global_j * p + global_k][col] += val

                # W part
                for l in range(4):
                    val = strassen_W[t][l]
                    if val != 0:
                        local_i_prime = l // 2
                        local_k_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_k = k_start + local_k_prime
                        W[global_k * n + global_i][col] += val

                col += 1

    return {"U": U, "V": V, "W": W}
