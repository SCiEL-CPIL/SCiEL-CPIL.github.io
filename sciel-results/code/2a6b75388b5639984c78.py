import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=2, m=2, p=12

    # Rank-4 decomposition using rank-1 factorization of A:
    # A = [[a,b],[c,d]] = [1,0]^T*[a,b] + [0,1]^T*[c,d]
    # C = A*B = [1,0]^T*([a,b]*B) + [0,1]^T*([c,d]*B)
    #
    # [a,b]*B is 1x2 x 2x12 multiplication: C[k] = a*B[0,k] + b*B[1,k]
    # Rank-2 decomposition:
    #   Term 1: A-side=a, B-side=B[0,:], C-side=C[0,:]
    #   Term 2: A-side=b, B-side=B[1,:], C-side=C[0,:]
    #
    # [c,d]*B is 1x2 x 2x12 multiplication: C[k] = c*B[0,k] + d*B[1,k]
    # Rank-2 decomposition:
    #   Term 3: A-side=c, B-side=B[0,:], C-side=C[1,:]
    #   Term 4: A-side=d, B-side=B[1,:], C-side=C[1,:]
    #
    # Total rank = 4
    rank = 4

    U = [[0] * rank for _ in range(n * m)]  # (4, 4)
    V = [[0] * rank for _ in range(m * p)]  # (24, 4)
    W = [[0] * rank for _ in range(p * n)]  # (24, 4)

    # Term 0: a * B[0,k] -> C[0,k]
    # U: A[0,0] = index 0
    U[0][0] = 1
    # V: B[0,k] for all k -> rows 0 to 11 (j=0, k=0..11)
    for k in range(p):
        V[0 * p + k][0] = 1
    # W: C[0,k] for all k -> W[k*n + 0] = W[k*2 + 0]
    for k in range(p):
        W[k * n + 0][0] = 1

    # Term 1: b * B[1,k] -> C[0,k]
    # U: A[0,1] = index 1
    U[1][1] = 1
    # V: B[1,k] for all k -> rows 12 to 23 (j=1, k=0..11)
    for k in range(p):
        V[1 * p + k][1] = 1
    # W: C[0,k] for all k
    for k in range(p):
        W[k * n + 0][1] = 1

    # Term 2: c * B[0,k] -> C[1,k]
    # U: A[1,0] = index 2
    U[2][2] = 1
    # V: B[0,k] for all k
    for k in range(p):
        V[0 * p + k][2] = 1
    # W: C[1,k] for all k -> W[k*n + 1] = W[k*2 + 1]
    for k in range(p):
        W[k * n + 1][2] = 1

    # Term 3: d * B[1,k] -> C[1,k]
    # U: A[1,1] = index 3
    U[3][3] = 1
    # V: B[1,k] for all k
    for k in range(p):
        V[1 * p + k][3] = 1
    # W: C[1,k] for all k
    for k in range(p):
        W[k * n + 1][3] = 1

    # Verify the decomposition is exact
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((n, m))
    B_test = rng.standard_normal((m, p))
    C_expected = A_test @ B_test

    C_computed = np.zeros((n, p))
    for r in range(rank):
        a_lin = U_arr[:, r]
        b_lin = V_arr[:, r]
        w_lin = W_arr[:, r]

        A_lin = a_lin.reshape(n, m)
        B_lin = b_lin.reshape(m, p)

        prod_a = np.sum(A_lin * A_test)
        prod_b = np.sum(B_lin * B_test)

        for k in range(p):
            for i in range(n):
                C_computed[i, k] += prod_a * prod_b * w_lin[k * n + i]

    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to naive rank-48 decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1

    return {"U": U, "V": V, "W": W}
