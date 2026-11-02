import time
import numpy as np

def construct():
    deadline = time.time() + 120.0

    n, m, p = 2, 2, 10
    num_blocks = p // 2  # 5

    # Strassen 7 terms for 2x2 multiplication.
    # U[i*m + j, col], V[j*p + k, col], W[k*n + i, col]
    # Local indices i,j in 0..1, k in 0..1.
    terms = [
        # M1 = (a+d)(e+h)
        ({(0, 0): 1, (1, 1): 1},
         {(0, 0): 1, (1, 1): 1},
         {(0, 0): 1, (1, 1): 1}),
        # M2 = (c+d)e
        ({(1, 0): 1, (1, 1): 1},
         {(0, 0): 1},
         {(0, 1): 1, (1, 1): -1}),
        # M3 = a(f-h)
        ({(0, 0): 1},
         {(0, 1): 1, (1, 1): -1},
         {(1, 0): 1, (1, 1): 1}),
        # M4 = d(g-e)
        ({(1, 1): 1},
         {(1, 0): 1, (0, 0): -1},
         {(0, 0): 1, (0, 1): 1}),
        # M5 = (a+b)h
        ({(0, 0): 1, (0, 1): 1},
         {(1, 1): 1},
         {(0, 0): -1, (1, 0): 1}),
        # M6 = (c-a)(e+f)
        ({(1, 0): 1, (0, 0): -1},
         {(0, 0): 1, (0, 1): 1},
         {(1, 1): 1}),
        # M7 = (b-d)(g+h)
        ({(0, 1): 1, (1, 1): -1},
         {(1, 0): 1, (1, 1): 1},
         {(0, 0): 1}),
    ]

    rank = num_blocks * 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for r in range(num_blocks):
        for t in range(7):
            col = r * 7 + t
            u_dict, v_dict, w_dict = terms[t]

            for (i, j), val in u_dict.items():
                U[i * m + j, col] = val

            for (j, k), val in v_dict.items():
                V[j * p + (2 * r + k), col] = val

            for (k, i), val in w_dict.items():
                W[(2 * r + k) * n + i, col] = val

    # Fast self-check: verify the tensor decomposition on a random input.
    if time.time() < deadline - 10.0:
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.arange(1, 2 * p + 1, dtype=float).reshape(m, p)
        C = A @ B
        Ua = np.asarray(U) @ np.asarray(V)  # rank x rank x rank check not needed;
        # Reconstruct: C[i, K] = sum_col U[i*m+j, col] * V[j*p+K, col] ...
        # Do it properly:
        Uc = np.asarray(U)
        Vc = np.asarray(V)
        Wc = np.asarray(W)
        # C = (U * V) contracted with W
        # C[i,K] = sum_{j} sum_{col} U[i*m+j,col]*V[j*p+K,col] * W[K*n+i,col]
        Crec = np.zeros((n, p))
        for i in range(n):
            for K in range(p):
                s = 0.0
                for col in range(rank):
                    # sum over j
                    term = 0.0
                    for j in range(m):
                        term += Uc[i * m + j, col] * Vc[j * p + K, col]
                    s += term * Wc[K * n + i, col]
                Crec[i, K] = s
        if not np.allclose(Crec, C, atol=1e-9):
            # Fallback: schoolbook rank 40 (correct by construction)
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for i in range(n):
                for j in range(m):
                    for K in range(p):
                        col = i * m * p + j * p + K
                        U[i * m + j, col] = 1
                        V[j * p + K, col] = 1
                        W[K * n + i, col] = 1

    if time.time() < deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
