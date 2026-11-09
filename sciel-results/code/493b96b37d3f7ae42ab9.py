import numpy as np

def construct():
    n, m, p = 3, 4, 5
    rank = n * m * p

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0
    for r in range(n):
        for k in range(p):
            for j in range(m):
                U[r * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + r, col] = 1
                col += 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
