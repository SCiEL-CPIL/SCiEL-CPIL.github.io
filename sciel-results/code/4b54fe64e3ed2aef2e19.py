import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = 2, 4, 6

    def make_A_idx(i, j):
        return i * m + j

    def make_B_idx(j, k):
        return j * p + k

    def make_C_idx(k, i):
        return k * n + i

    # Strassen steps with correct signs in C contributions
    # M1 = (A11+A22)(B11+B22): C11 += M1, C22 += M1
    # M2 = (A21+A22)B11: C21 += M2, C22 -= M2
    # M3 = A11(B12-B22): C12 += M3, C22 += M3
    # M4 = A22(B21-B11): C11 += M4, C21 += M4
    # M5 = (A11+A12)B22: C11 -= M5, C12 += M5
    # M6 = (A21-A11)(B11+B12): C22 += M6
    # M7 = (A12-A22)(B21+B22): C11 += M7, C22 += M7

    steps = []

    # Step 1: M1
    X1 = [
        {make_A_idx(0,0): 1, make_A_idx(1,2): 1},
        {make_A_idx(0,1): 1, make_A_idx(1,3): 1},
    ]
    Y1 = [
        [
            {make_B_idx(0,0): 1, make_B_idx(2,3): 1},
            {make_B_idx(0,1): 1, make_B_idx(2,4): 1},
            {make_B_idx(0,2): 1, make_B_idx(2,5): 1},
        ],
        [
            {make_B_idx(1,0): 1, make_B_idx(3,3): 1},
            {make_B_idx(1,1): 1, make_B_idx(3,4): 1},
            {make_B_idx(1,2): 1, make_B_idx(3,5): 1},
        ],
    ]
    C1 = [
        {make_C_idx(0,0): 1, make_C_idx(3,1): 1},  # C11[0], C22[0]
        {make_C_idx(1,0): 1, make_C_idx(4,1): 1},  # C11[1], C22[1]
        {make_C_idx(2,0): 1, make_C_idx(5,1): 1},  # C11[2], C22[2]
    ]
    steps.append((X1, Y1, C1))

    # Step 2: M2
    X2 = [
        {make_A_idx(1,0): 1, make_A_idx(1,2): 1},
        {make_A_idx(1,1): 1, make_A_idx(1,3): 1},
    ]
    Y2 = [
        [
            {make_B_idx(0,0): 1},
            {make_B_idx(0,1): 1},
            {make_B_idx(0,2): 1},
        ],
        [
            {make_B_idx(1,0): 1},
            {make_B_idx(1,1): 1},
            {make_B_idx(1,2): 1},
        ],
    ]
    C2 = [
        {make_C_idx(0,1): 1, make_C_idx(3,1): -1},  # C21[0], C22[0] with -1
        {make_C_idx(1,1): 1, make_C_idx(4,1): -1},  # C21[1], C22[1] with -1
        {make_C_idx(2,1): 1, make_C_idx(5,1): -1},  # C21[2], C22[2] with -1
    ]
    steps.append((X2, Y2, C2))

    # Step 3: M3
    X3 = [
        {make_A_idx(0,0): 1},
        {make_A_idx(0,1): 1},
    ]
    Y3 = [
        [
            {make_B_idx(0,3): 1, make_B_idx(2,3): -1},
            {make_B_idx(0,4): 1, make_B_idx(2,4): -1},
            {make_B_idx(0,5): 1, make_B_idx(2,5): -1},
        ],
        [
            {make_B_idx(1,3): 1, make_B_idx(3,3): -1},
            {make_B_idx(1,4): 1, make_B_idx(3,4): -1},
            {make_B_idx(1,5): 1, make_B_idx(3,5): -1},
        ],
    ]
    C3 = [
        {make_C_idx(3,0): 1, make_C_idx(3,1): 1},  # C12[0], C22[0]
        {make_C_idx(4,0): 1, make_C_idx(4,1): 1},  # C12[1], C22[1]
        {make_C_idx(5,0): 1, make_C_idx(5,1): 1},  # C12[2], C22[2]
    ]
    steps.append((X3, Y3, C3))

    # Step 4: M4
    X4 = [
        {make_A_idx(1,2): 1},
        {make_A_idx(1,3): 1},
    ]
    Y4 = [
        [
            {make_B_idx(2,0): 1, make_B_idx(0,0): -1},
            {make_B_idx(2,1): 1, make_B_idx(0,1): -1},
            {make_B_idx(2,2): 1, make_B_idx(0,2): -1},
        ],
        [
            {make_B_idx(3,0): 1, make_B_idx(1,0): -1},
            {make_B_idx(3,1): 1, make_B_idx(1,1): -1},
            {make_B_idx(3,2): 1, make_B_idx(1,2): -1},
        ],
    ]
    C4 = [
        {make_C_idx(0,0): 1, make_C_idx(0,1): 1},  # C11[0], C21[0]
        {make_C_idx(1,0): 1, make_C_idx(1,1): 1},  # C11[1], C21[1]
        {make_C_idx(2,0): 1, make_C_idx(2,1): 1},  # C11[2], C21[2]
    ]
    steps.append((X4, Y4, C4))

    # Step 5: M5
    X5 = [
        {make_A_idx(0,0): 1, make_A_idx(0,2): 1},
        {make_A_idx(0,1): 1, make_A_idx(0,3): 1},
    ]
    Y5 = [
        [
            {make_B_idx(2,3): 1},
            {make_B_idx(2,4): 1},
            {make_B_idx(2,5): 1},
        ],
        [
            {make_B_idx(3,3): 1},
            {make_B_idx(3,4): 1},
            {make_B_idx(3,5): 1},
        ],
    ]
    C5 = [
        {make_C_idx(0,0): -1, make_C_idx(3,0): 1},  # C11[0] with -1, C12[0]
        {make_C_idx(1,0): -1, make_C_idx(4,0): 1},  # C11[1] with -1, C12[1]
        {make_C_idx(2,0): -1, make_C_idx(5,0): 1},  # C11[2] with -1, C12[2]
    ]
    steps.append((X5, Y5, C5))

    # Step 6: M6
    X6 = [
        {make_A_idx(1,0): 1, make_A_idx(0,0): -1},
        {make_A_idx(1,1): 1, make_A_idx(0,1): -1},
    ]
    Y6 = [
        [
            {make_B_idx(0,0): 1, make_B_idx(0,3): 1},
            {make_B_idx(0,1): 1, make_B_idx(0,4): 1},
            {make_B_idx(0,2): 1, make_B_idx(0,5): 1},
        ],
        [
            {make_B_idx(1,0): 1, make_B_idx(1,3): 1},
            {make_B_idx(1,1): 1, make_B_idx(1,4): 1},
            {make_B_idx(1,2): 1, make_B_idx(1,5): 1},
        ],
    ]
    C6 = [
        {make_C_idx(3,1): 1},  # C22[0]
        {make_C_idx(4,1): 1},  # C22[1]
        {make_C_idx(5,1): 1},  # C22[2]
    ]
    steps.append((X6, Y6, C6))

    # Step 7: M7
    X7 = [
        {make_A_idx(0,2): 1, make_A_idx(1,2): -1},
        {make_A_idx(0,3): 1, make_A_idx(1,3): -1},
    ]
    Y7 = [
        [
            {make_B_idx(2,0): 1, make_B_idx(2,3): 1},
            {make_B_idx(2,1): 1, make_B_idx(2,4): 1},
            {make_B_idx(2,2): 1, make_B_idx(2,5): 1},
        ],
        [
            {make_B_idx(3,0): 1, make_B_idx(3,3): 1},
            {make_B_idx(3,1): 1, make_B_idx(3,4): 1},
            {make_B_idx(3,2): 1, make_B_idx(3,5): 1},
        ],
    ]
    C7 = [
        {make_C_idx(0,0): 1, make_C_idx(3,1): 1},  # C11[0], C22[0]
        {make_C_idx(1,0): 1, make_C_idx(4,1): 1},  # C11[1], C22[1]
        {make_C_idx(2,0): 1, make_C_idx(5,1): 1},  # C11[2], C22[2]
    ]
    steps.append((X7, Y7, C7))

    rank = 7 * 6  # 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for step_idx, (X_def, Y_def, C_def) in enumerate(steps):
        for r in range(2):
            for c in range(3):
                for a_idx, coeff in X_def[r].items():
                    U[a_idx, col] = coeff
                for b_idx, coeff in Y_def[r][c].items():
                    V[b_idx, col] = coeff
                for c_idx, coeff in C_def[c].items():
                    W[c_idx, col] = coeff
                col += 1

    # Verify
    if time.time() < deadline - 10:
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for k in range(rank):
            T[:, :, k] = np.outer(U[:, k], V[:, k]) * W[:, k]

        T_expected = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_expected[a, b, c] = 1

        if not np.allclose(T, T_expected, atol=1e-10):
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, column] = 1
                        V[j * p + k, column] = 1
                        W[k * n + i, column] = 1
                        column += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
