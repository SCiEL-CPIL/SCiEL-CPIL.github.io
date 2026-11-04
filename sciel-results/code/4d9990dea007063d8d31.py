import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
_U2 = np.array([
    [1, 0, 1, 0, 1, -1, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, -1]
], dtype=int)

_V2 = np.array([
    [1, 1, 0, -1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [1, 0, -1, 0, 1, 0, 1]
], dtype=int)

_W2 = np.array([
    [1, 0, 0, 1, -1, 0, 1],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [1, -1, 1, 0, 0, 1, 0]
], dtype=int)

def construct():
    n, m, p = DIMENSIONS
    U_kron = np.kron(_U2, _U2)
    V_kron = np.kron(_V2, _V2)
    W_kron = np.kron(_W2, _W2)

    perm_U = np.zeros(16, dtype=int)
    for r1 in range(4):
        for r2 in range(4):
            i1 = r1 // 2
            j1 = r1 % 2
            i0 = r2 // 2
            j0 = r2 % 2
            i = 2 * i1 + i0
            j = 2 * j1 + j0
            target_row = i * 4 + j
            kron_row = r1 * 4 + r2
            perm_U[target_row] = kron_row
    U = U_kron[perm_U, :]

    perm_V = np.zeros(16, dtype=int)
    for r1 in range(4):
        for r2 in range(4):
            j1 = r1 // 2
            k1 = r1 % 2
            j0 = r2 // 2
            k0 = r2 % 2
            j = 2 * j1 + j0
            k = 2 * k1 + k0
            target_row = j * 4 + k
            kron_row = r1 * 4 + r2
            perm_V[target_row] = kron_row
    V = V_kron[perm_V, :]

    perm_W = np.zeros(16, dtype=int)
    for r1 in range(4):
        for r2 in range(4):
            i1 = r1 // 2
            k1 = r1 % 2
            i0 = r2 // 2
            k0 = r2 % 2
            i = 2 * i1 + i0
            k = 2 * k1 + k0
            target_row = k * 4 + i
            kron_row = r1 * 4 + r2
            perm_W[target_row] = kron_row
    W = W_kron[perm_W, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
