import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-44 matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    rank = 44

    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A1 * B1 (2x2 * 2x8)
    # A1 is A[0:2, 0:2]
    # B1 is B[0:2, 0:8]
    # We treat B1 as four 2x2 blocks: B1[:, 2j:2j+2] for j in 0..3

    # Strassen coefficients for 2x2x2:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)

    # U indices for A1 (2x2), using A[i, j] -> U[i*m + j] where m=3:
    # a: A[0,0] -> U[0]
    # b: A[0,1] -> U[1]
    # c: A[1,0] -> U[3]
    # d: A[1,1] -> U[4]

    for j in range(4):
        # V indices for B1 block j (2x2): B[i, k] -> V[i*p + k] where p=8
        # e: B[0, 2j]     -> V[0*8 + 2j]   = V[2j]
        # f: B[0, 2j+1]   -> V[0*8 + 2j+1] = V[2j+1]
        # g: B[1, 2j]     -> V[1*8 + 2j]   = V[8+2j]
        # h: B[1, 2j+1]   -> V[1*8 + 2j+1] = V[8+2j+1] = V[9+2j]

        v_e = 2 * j
        v_f = 2 * j + 1
        v_g = 8 + 2 * j
        v_h = 9 + 2 * j

        # W indices for C block j (2x2): C[i, k] -> W[k*n + i] where n=2
        # R00 = C[0, 2j]   -> W[2j*2 + 0]   = W[4j]
        # R01 = C[0, 2j+1] -> W[(2j+1)*2 + 0] = W[4j+2]
        # R10 = C[1, 2j]   -> W[2j*2 + 1]   = W[4j+1]
        # R11 = C[1, 2j+1] -> W[(2j+1)*2 + 1] = W[4j+3]

        w_r00 = 4 * j
        w_r01 = 4 * j + 2
        w_r10 = 4 * j + 1
        w_r11 = 4 * j + 3

        # P1 = (a+d)(e+h)
        U[0][col] = 1
        U[4][col] = 1
        V[v_e][col] = 1
        V[v_h][col] = 1
        W[w_r00][col] = 1
        W[w_r11][col] = 1
        col += 1

        # P2 = (c+d)e
        U[3][col] = 1
        U[4][col] = 1
        V[v_e][col] = 1
        W[w_r10][col] = 1
        W[w_r11][col] = -1
        col += 1

        # P3 = a(f-h)
        U[0][col] = 1
        V[v_f][col] = 1
        V[v_h][col] = -1
        W[w_r01][col] = 1
        W[w_r11][col] = 1
        col += 1

        # P4 = d(g-e)
        U[4][col] = 1
        V[v_g][col] = 1
        V[v_e][col] = -1
        W[w_r00][col] = 1
        W[w_r10][col] = 1
        col += 1

        # P5 = (a+b)h
        U[0][col] = 1
        U[1][col] = 1
        V[v_h][col] = 1
        W[w_r00][col] = -1
        W[w_r01][col] = 1
        col += 1

        # P6 = (c-a)(e+f)
        U[3][col] = 1
        U[0][col] = -1
        V[v_e][col] = 1
        V[v_f][col] = 1
        W[w_r11][col] = 1
        col += 1

        # P7 = (b-d)(g+h)
        U[1][col] = 1
        U[4][col] = -1
        V[v_g][col] = 1
        V[v_h][col] = 1
        W[w_r00][col] = 1
        col += 1

    # Part 2: A2 * B2 (2x1 * 1x8)
    # A2 is A[0:2, 2], i.e. A[0,2] and A[1,2]
    # B2 is B[2, 0:8], i.e. B[2, k] for k in 0..7

    for i in range(2):
        for k in range(8):
            # U index for A[i, 2] -> U[i*3 + 2]
            u_idx = i * 3 + 2
            # V index for B[2, k] -> V[2*8 + k] = V[16 + k]
            v_idx = 16 + k
            # W index for C[i, k] -> W[k*2 + i]
            w_idx = k * 2 + i

            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Verify col == rank
    if col != rank:
        raise ValueError(f"Internal error: expected {rank} columns, got {col}")

    # Quick sanity check using numpy if time permits
    if time.time() < deadline - 5:
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        # Build the tensor T_{i,k,l} = sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r]
        # and check against the identity tensor for matrix multiplication.
        # Tensor M_{i,l,k} should equal 1 if l == l (i.e. for all i,l,k it's 1), actually
        # the multiplication tensor has T_{i,k,l} = 1 for all i,k,l.
        # Let's verify: for each (i, l, k), compute sum_r U[i, r] * V[l, r] * W[k, r]
        # where U[i,r] is actually U[i*m + l, r], etc.
        # T[i, l, k] = sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r]
        # We expect T[i, l, k] == 1 for all i in [0,n), l in [0,m), k in [0,p).
        T = np.zeros((n, m, p))
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i * m + l, r] * V_arr[l * p + k, r] * W_arr[k * n + i, r]
                    T[i, l, k] = s
        expected = np.ones((n, m, p))
        if not np.allclose(T, expected, atol=1e-10):
            # If mismatch, raise
            raise ValueError("Decomposition verification failed")

    return {"U": U, "V": V, "W": W}
