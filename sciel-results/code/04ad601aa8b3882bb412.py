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

    # Initialize with zeros
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
    # Strassen U factors (A)
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    # Strassen V factors (B)
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    # Strassen W factors (C) - Scaled by 0.5 to correct the factor of 2
    strassen_W = [
        [0.5, 0, 0, 0.5],
        [0, 0, 0.5, -0.5],
        [0, 0.5, 0, 0.5],
        [0.5, 0, 0.5, 0],
        [-0.5, 0.5, 0, 0],
        [0, 0, 0, 0.5],
        [0.5, 0, 0, 0]
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
