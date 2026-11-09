import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    # Part 1: A1 (2x1) * B1 (1x5) — trivial
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    # Part 2: A2 (2x2) * B2a (2x2) via Strassen
    # A2: rows 0,1 of A; cols 1,2
    # B2a: rows 1,2 of B; cols 0,1
    # U indices: a=U[1], b=U[2], c=U[4], d=U[5]
    # V indices: e=V[5], f=V[6], g=V[10], h=V[11]
    # W: C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]

    strassen2 = [
        ({1: 1, 5: 1}, {5: 1, 11: 1}, {0: 1, 3: 1}),
        ({4: 1, 5: 1}, {5: 1}, {1: 1, 3: -1}),
        ({1: 1}, {6: 1, 11: -1}, {2: 1, 3: 1}),
        ({5: 1}, {10: 1, 5: -1}, {0: 1, 1: 1}),
        ({1: 1, 2: 1}, {11: 1}, {0: -1, 2: 1}),
        ({4: 1, 1: -1}, {5: 1, 6: 1}, {3: 1}),
        ({2: 1, 5: -1}, {10: 1, 11: 1}, {0: 1}),
    ]

    for u_c, v_c, w_c in strassen2:
        for idx, val in u_c.items():
            U[idx, col] = val
        for idx, val in v_c.items():
            V[idx, col] = val
        for idx, val in w_c.items():
            W[idx, col] = val
        col += 1

    # Part 3a: A2 (2x2) * B2b1 (2x2) via Strassen
    # B2b1: rows 1,2 of B; cols 2,3
    # V: e=V[7], f=V[8], g=V[12], h=V[13]
    # W: C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]

    strassen3a = [
        ({1: 1, 5: 1}, {7: 1, 13: 1}, {4: 1, 7: 1}),
        ({4: 1, 5: 1}, {7: 1}, {5: 1, 7: -1}),
        ({1: 1}, {8: 1, 13: -1}, {6: 1, 7: 1}),
        ({5: 1}, {12: 1, 7: -1}, {4: 1, 5: 1}),
        ({1: 1, 2: 1}, {13: 1}, {4: -1, 6: 1}),
        ({4: 1, 1: -1}, {7: 1, 8: 1}, {7: 1}),
        ({2: 1, 5: -1}, {12: 1, 13: 1}, {4: 1}),
    ]

    for u_c, v_c, w_c in strassen3a:
        for idx, val in u_c.items():
            U[idx, col] = val
        for idx, val in v_c.items():
            V[idx, col] = val
        for idx, val in w_c.items():
            W[idx, col] = val
        col += 1

    # Part 3b: A2 (2x2) * B2b2 (2x1) trivial
    # B2b2: rows 1,2 of B; col 4
    # V: B[1,4]=V[9], B[2,4]=V[14]
    # W: C[0,4]=W[8], C[1,4]=W[9]

    U[1, col] = 1; V[9, col] = 1; W[8, col] = 1; col += 1
    U[2, col] = 1; V[14, col] = 1; W[8, col] = 1; col += 1
    U[4, col] = 1; V[9, col] = 1; W[9, col] = 1; col += 1
    U[5, col] = 1; V[14, col] = 1; W[9, col] = 1; col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
