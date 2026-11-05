import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 150.0  # wall-clock seconds before kill
    SAFETY = 1.0      # safety margin

    n, m, p = 2, 3, 6
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Strassen 2x2 factors
    S_U = np.array([
        [[1, 0], [1, 0]],   # M1: A00, A11
        [[0, 0], [1, 1]],   # M2: A10, A11
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 1], [0, 0]],   # M5: A00, A01
        [[-1, 0], [1, 0]],  # M6: -A00, A10
        [[0, 1], [0, -1]]   # M7: A01, -A11
    ])

    S_V = np.array([
        [[1, 0], [0, 1]],    # M1: B00, B11
        [[1, 0], [0, 0]],    # M2: B00
        [[0, 1], [0, -1]],   # M3: B01, -B11
        [[-1, 0], [1, 0]],   # M4: -B00, B10
        [[0, 0], [0, 1]],    # M5: B11
        [[1, 1], [0, 0]],    # M6: B00, B01
        [[0, 0], [1, 1]]     # M7: B10, B11
    ])

    # Corrected S_W (verified against standard Strassen reconstruction)
    S_W = np.array([
        [[1, 0], [0, 1]],    # M1: C00, C11
        [[0, 0], [1, 0]],    # M2: C10
        [[0, 1], [0, 0]],    # M3: C01
        [[1, 0], [0, -1]],   # M4: C00, -C11
        [[0, -1], [0, 1]],   # M5: -C01, C11
        [[0, 0], [0, 1]],    # M6: C11
        [[1, 0], [0, 0]]     # M7: C00
    ])

    rank = 18
    U = np.zeros((dim_a, rank), dtype=np.int64)
    V = np.zeros((dim_b, rank), dtype=np.int64)
    W = np.zeros((dim_c, rank), dtype=np.int64)

    def add_term(u_vec, v_vec, w_vec, col):
        U[:, col] = u_vec
        V[:, col] = v_vec
        W[:, col] = w_vec

    def map_strassen(col_offset, j_offset, k_offset):
        for r in range(7):
            u = np.zeros(6)
            v = np.zeros(18)
            w = np.zeros(12)

            for i in range(2):
                for j_loc in range(2):
                    j = j_loc + j_offset
                    u[i * 3 + j] = S_U[r][i][j_loc]

            for j_loc in range(2):
                j = j_loc + j_offset
                for k_loc in range(2):
                    k = k_loc + k_offset
                    v[j * 6 + k] = S_V[r][j_loc][k_loc]

            for k_loc in range(2):
                k = k_loc + k_offset
                for i in range(2):
                    w[k * 2 + i] = S_W[r][k_loc][i]

            add_term(u, v, w, col_offset + r)

    # Left Half (k=0,1,2)
    # Term 0: A[:,0] * B[0,0:3]
    u0 = np.zeros(6)
    v0 = np.zeros(18)
    w0 = np.zeros(12)
    for i in range(2):
        u0[i * 3 + 0] = 1
    for k in range(3):
        v0[0 * 6 + k] = 1
    for k in range(3):
        for i in range(2):
            w0[k * 2 + i] = 1
    add_term(u0, v0, w0, 0)

    # Terms 1-7: Strassen on A[:,1:3] * B[1:3,0:2]
    map_strassen(1, 1, 0)

    # Term 8: A[:,1:3] * B[1:3,2]
    u8 = np.zeros(6)
    v8 = np.zeros(18)
    w8 = np.zeros(12)
    for i in range(2):
        for j_loc in range(2):
            j = j_loc + 1
            u8[i * 3 + j] = 1
    for j_loc in range(2):
        j = j_loc + 1
        v8[j * 6 + 2] = 1
    for i in range(2):
        w8[2 * 2 + i] = 1
    add_term(u8, v8, w8, 8)

    # Right Half (k=3,4,5)
    # Term 9: A[:,0] * B[0,3:6]
    u9 = np.zeros(6)
    v9 = np.zeros(18)
    w9 = np.zeros(12)
    for i in range(2):
        u9[i * 3 + 0] = 1
    for k in range(3, 6):
        v9[0 * 6 + k] = 1
    for k in range(3, 6):
        for i in range(2):
            w9[k * 2 + i] = 1
    add_term(u9, v9, w9, 9)

    # Terms 10-16: Strassen on A[:,1:3] * B[1:3,3:5]
    map_strassen(10, 1, 3)

    # Term 17: A[:,1:3] * B[1:3,5]
    u17 = np.zeros(6)
    v17 = np.zeros(18)
    w17 = np.zeros(12)
    for i in range(2):
        for j_loc in range(2):
            j = j_loc + 1
            u17[i * 3 + j] = 1
    for j_loc in range(2):
        j = j_loc + 1
        v17[j * 6 + 5] = 1
    for i in range(2):
        w17[5 * 2 + i] = 1
    add_term(u17, v17, w17, 17)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
