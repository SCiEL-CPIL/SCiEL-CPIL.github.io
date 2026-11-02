import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 2, 4
    R = 14

    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]

    # Define the 7 Strassen bilinear forms as lists of (index, coeff) per factor.
    # Indices are expressed in terms of a local 2x2 block:
    #   A: a[i][j] -> flat index i*2+j  (i,j in {0,1})
    #   B: b[j][k] -> local flat index j*2+k, then global j*4 + (k + k_off)
    #   C: c[k][i] -> local flat index k*2+i, then global (k + k_off)*2 + i
    #
    # Each entry: (u_list, v_list, w_list) where each _list is [(local_idx, coeff), ...]
    # For V and W, local_idx encodes (j, k_local) or (k_local, i) with k_local in {0,1}.

    forms = [
        # M1 = (a00+a11)(b00+b11)
        ([(0, 1), (3, 1)],
         [(0, 1), (3, 1)],
         [(0, 1), (3, 1)]),
        # M2 = (a10+a11)b00
        ([(2, 1), (3, 1)],
         [(0, 1)],
         [(1, 1), (3, -1)]),
        # M3 = a00(b01-b11)
        ([(0, 1)],
         [(1, 1), (3, -1)],
         [(2, 1), (3, 1)]),
        # M4 = a11(b10-b00)
        ([(3, 1)],
         [(2, 1), (0, -1)],
         [(0, 1), (1, 1)]),
        # M5 = (a00+a01)b11
        ([(0, 1), (1, 1)],
         [(3, 1)],
         [(0, -1), (2, 1)]),
        # M6 = (a10-a00)(b00+b01)
        ([(2, 1), (0, -1)],
         [(0, 1), (1, 1)],
         [(3, 1)]),
        # M7 = (a01-a11)(b10+b11)
        ([(1, 1), (3, -1)],
         [(2, 1), (3, 1)],
         [(0, 1)]),
    ]

    # Two blocks: k_off = 0 (B cols 0,1 -> C cols 0,1) and k_off = 2 (B cols 2,3 -> C cols 2,3)
    for k_off in (0, 2):
        base = (k_off >> 1) * 7  # 0 for first block, 7 for second

        for s in range(7):
            col = base + s
            u_list, v_list, w_list = forms[s]

            for u_idx, u_coeff in u_list:
                U[u_idx][col] = u_coeff

            for v_local, v_coeff in v_list:
                # v_local encodes j*2 + k_local where j in {0,1}, k_local in {0,1}
                v_j = v_local >> 1
                v_k_local = v_local & 1
                v_global = v_j * 4 + v_k_local + k_off
                V[v_global][col] = v_coeff

            for w_local, w_coeff in w_list:
                # w_local encodes k_local*2 + i where k_local in {0,1}, i in {0,1}
                w_k_local = w_local >> 1
                w_i = w_local & 1
                w_global = (w_k_local + k_off) * 2 + w_i
                W[w_global][col] = w_coeff

    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
