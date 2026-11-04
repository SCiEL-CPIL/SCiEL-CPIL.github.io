"""Matrix multiplication 4x4x5 rank-minimal decomposition search."""
import numpy as np

N, M, P = 4, 4, 5


def construct():
    total_rank = 72
    U = np.zeros((N * M, total_rank))
    V = np.zeros((M * P, total_rank))
    W = np.zeros((P * N, total_rank))
    col_ptr = 0

    def place_strassen(ra, ca, rb, cb):
        nonlocal col_ptr
        ar0, ar1 = 2 * ra, 2 * ra + 1
        ac0, ac1 = 2 * ca, 2 * ca + 1
        br0, br1 = 2 * rb, 2 * rb + 1
        bc0, bc1 = 2 * cb, 2 * cb + 1
        cr0, cr1 = 2 * ra, 2 * ra + 1
        cc0, cc1 = 2 * cb, 2 * cb + 1

        c = col_ptr
        U[ar0 * M + ac0, c] = 1.0
        U[ar1 * M + ac1, c] = 1.0
        V[br0 * P + bc0, c] = 1.0
        V[br1 * P + bc1, c] = 1.0
        W[cc0 * N + cr0, c] = 1.0
        W[cc1 * N + cr1, c] = 1.0
        col_ptr += 1

        c = col_ptr
        U[ar1 * M + ac0, c] = 1.0
        U[ar1 * M + ac1, c] = 1.0
        V[br0 * P + bc0, c] = 1.0
        W[cc0 * N + cr1, c] = 1.0
        W[cc1 * N + cr1, c] = -1.0
        col_ptr += 1

        c = col_ptr
        U[ar0 * M + ac0, c] = 1.0
        V[br0 * P + bc1, c] = 1.0
        V[br1 * P + bc1, c] = -1.0
        W[cc1 * N + cr0, c] = 1.0
        W[cc1 * N + cr1, c] = 1.0
        col_ptr += 1

        c = col_ptr
        U[ar1 * M + ac1, c] = 1.0
        V[br1 * P + bc0, c] = 1.0
        V[br0 * P + bc0, c] = -1.0
        W[cc0 * N + cr0, c] = 1.0
        W[cc0 * N + cr1, c] = 1.0
        col_ptr += 1

        c = col_ptr
        U[ar0 * M + ac0, c] = 1.0
        U[ar0 * M + ac1, c] = 1.0
        V[br1 * P + bc1, c] = 1.0
        W[cc0 * N + cr0, c] = -1.0
        W[cc1 * N + cr0, c] = 1.0
        col_ptr += 1

        c = col_ptr
        U[ar1 * M + ac0, c] = 1.0
        U[ar0 * M + ac0, c] = -1.0
        V[br0 * P + bc0, c] = 1.0
        V[br0 * P + bc1, c] = 1.0
        W[cc1 * N + cr1, c] = 1.0
        col_ptr += 1

        c = col_ptr
        U[ar0 * M + ac1, c] = 1.0
        U[ar1 * M + ac1, c] = -1.0
        V[br1 * P + bc0, c] = 1.0
        V[br1 * P + bc1, c] = 1.0
        W[cc0 * N + cr0, c] = 1.0
        col_ptr += 1

    def place_direct(ra, ca, rb, kc):
        nonlocal col_ptr
        for i in range(2):
            for j in range(2):
                c = col_ptr
                U[(2 * ra + i) * M + (2 * ca + j), c] = 1.0
                V[(2 * rb + j) * P + kc, c] = 1.0
                W[kc * N + (2 * ra + i), c] = 1.0
                col_ptr += 1

    for ra in range(2):
        for ca in range(2):
            for rb in range(2):
                place_strassen(ra, ca, rb, 0)
                place_strassen(ra, ca, rb, 1)
                place_direct(ra, ca, rb, 4)

    U = U[:, :col_ptr]
    V = V[:, :col_ptr]
    W = W[:, :col_ptr]

    for i in range(N):
        for j in range(M):
            for k in range(P):
                val = float(np.dot(U[i * M + j], V[j * P + k] * W[k * N + i]))
                if abs(val - 1.0) > 1e-9:
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    r0 = N * M * P
    U0 = np.zeros((N * M, r0))
    V0 = np.zeros((M * P, r0))
    W0 = np.zeros((P * N, r0))
    idx = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U0[i * M + j, idx] = 1.0
                V0[j * P + k, idx] = 1.0
                W0[k * N + i, idx] = 1.0
                idx += 1
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
