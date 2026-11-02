import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 2, 14
    rank = 49

    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)

    # Strassen's 7 products for 2x2 multiplication:
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)

    # U coefficients: [a, b, c, d]
    U_coeff = [
        (1, 0, 0, 1),
        (0, 0, 1, 1),
        (1, 0, 0, 0),
        (0, 0, 0, 1),
        (1, 1, 0, 0),
        (-1, 0, 1, 0),
        (0, 1, 0, -1),
    ]

    # V coefficients: [e, f, g, h]
    V_coeff = [
        (1, 0, 0, 1),
        (1, 0, 0, 0),
        (0, 1, 0, -1),
        (-1, 0, 1, 0),
        (0, 0, 0, 1),
        (1, 1, 0, 0),
        (0, 0, 1, 1),
    ]

    # W coefficients: [C00, C01, C10, C11]
    W_coeff = [
        (1, 0, 0, 1),
        (0, 0, 1, -1),
        (0, 1, 0, 1),
        (1, 0, 1, 0),
        (-1, 1, 0, 0),
        (0, 0, 0, 1),
        (1, 0, 0, 0),
    ]

    for b in range(7):
        for s in range(7):
            col = b * 7 + s

            U[0, col] = U_coeff[s][0]
            U[1, col] = U_coeff[s][1]
            U[2, col] = U_coeff[s][2]
            U[3, col] = U_coeff[s][3]

            row0_c0 = 2 * b
            row0_c1 = 2 * b + 1
            row1_c0 = 14 + 2 * b
            row1_c1 = 14 + 2 * b + 1
            V[row0_c0, col] = V_coeff[s][0]
            V[row0_c1, col] = V_coeff[s][1]
            V[row1_c0, col] = V_coeff[s][2]
            V[row1_c1, col] = V_coeff[s][3]

            # W: C = 2 rows x 14 cols, index = k*n + i
            W[4 * b, col] = W_coeff[s][0]
            W[4 * b + 2, col] = W_coeff[s][1]
            W[4 * b + 1, col] = W_coeff[s][2]
            W[4 * b + 3, col] = W_coeff[s][3]

    # Convert to lists to match interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
