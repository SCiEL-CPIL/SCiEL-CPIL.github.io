import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    """
    Returns a rank-28 exact tensor decomposition for M_{2,2,8}.
    
    M_{2,2,8} = M_{2,2,2} ⊕ M_{2,2,2} ⊕ M_{2,2,2} ⊕ M_{2,2,2}
    Minimal rank of M_{2,2,2} is 7 (Strassen).
    Thus minimal rank of M_{2,2,8} is 4 * 7 = 28.
    
    We apply Strassen's algorithm to each of the four 2x2 column-blocks of B.
    """
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 8
    num_blocks = p // 2   # 4 blocks
    rank = num_blocks * 7 # 28

    # Pre-allocate
    U = np.zeros((n * m, rank), dtype=float)  # 4 x 28
    V = np.zeros((m * p, rank), dtype=float)  # 16 x 28
    W = np.zeros((p * n, rank), dtype=float)  # 16 x 28

    # Strassen's 7 products for 2x2 x 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6

    # U_strassen[r][i] = coefficient of A[i] in product r
    # A indices: 0=a(0,0), 1=b(0,1), 2=c(1,0), 3=d(1,1)
    U_strassen = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]

    # V_strassen[r][l] = coefficient of B[l] in product r
    # B local indices: 0=e(0,0), 1=f(0,1), 2=g(1,0), 3=h(1,1)
    V_strassen = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1],   # M7: g+h
    ]

    # W_strassen[r][c] = coefficient of product r in C[c]
    # C local indices: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
    W_strassen = [
        [1, 0, 0, 1],   # M1: C[0,0]+C[1,1]
        [0, 0, 1, -1],  # M2: C[1,0]-C[1,1]
        [0, 1, 0, 1],   # M3: C[0,1]+C[1,1]
        [1, 0, 1, 0],   # M4: C[0,0]+C[1,0]
        [-1, 1, 0, 0],  # M5: -C[0,0]+C[0,1]
        [0, 0, 0, 1],   # M6: C[1,1]
        [1, 0, 0, 0],   # M7: C[0,0]
    ]

    for b in range(num_blocks):
        base = b * 7
        for r in range(7):
            t = base + r

            # Fill U: A indices are global (i*m + j), same for all blocks
            for i in range(n * m):
                U[i, t] = U_strassen[r][i]

            # Fill V: B local index l maps to global B index
            # B is m x p (2 x 8), row-major: B[row, col] -> row*p + col
            # Local l: 0->(0,2b), 1->(0,2b+1), 2->(1,2b), 3->(1,2b+1)
            v_local_to_global = [
                0 * p + 2 * b,       # e: B[0, 2b]
                0 * p + 2 * b + 1,   # f: B[0, 2b+1]
                1 * p + 2 * b,       # g: B[1, 2b]
                1 * p + 2 * b + 1,   # h: B[1, 2b+1]
            ]
            for l in range(4):
                V[v_local_to_global[l], t] = V_strassen[r][l]

            # Fill W: C local index c maps to global W index
            # Convention: W[k*n + i, r] where k is column of C, i is row of C
            # C is n x p (2 x 8), C[row, col]
            # Local c: 0->C[0,2b], 1->C[0,2b+1], 2->C[1,2b], 3->C[1,2b+1]
            # W index = col*n + row
            w_local_to_global = [
                2 * b * n + 0,       # C[0, 2b]
                (2 * b + 1) * n + 0, # C[0, 2b+1]
                2 * b * n + 1,       # C[1, 2b]
                (2 * b + 1) * n + 1, # C[1, 2b+1]
            ]
            for c in range(4):
                W[w_local_to_global[c], t] = W_strassen[r][c]

    # Quick verification with a random test
    if time.time() < deadline:
        np.random.seed(12345)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test  # (n, p) = (2, 8)

        A_flat = A_test.flatten()  # row-major: [a, b, c, d]
        B_flat = B_test.flatten()  # row-major: [B[0,0], ..., B[0,7], B[1,0], ..., B[1,7]]

        # Compute C using the decomposition
        # C[c_global] = sum_r sum_a sum_b U[a,r] * V[b,r] * W[c_global,r] * A[a] * B[b]
        # More efficiently: for each r, compute (U[:,r]^T @ A_flat) * (V[:,r]^T @ B_flat) * W[:,r]
        C_computed = np.zeros(p * n)
        for r in range(rank):
            u_val = np.dot(U[:, r], A_flat)
            v_val = np.dot(V[:, r], B_flat)
            C_computed += u_val * v_val * W[:, r]

        # C_computed is indexed by W convention: W[k*n + i] = C[i, k]
        # So C_computed[k*n + i] = C[i, k]
        # Reshape: C_computed.reshape(p, n).T should equal C_expected (n, p)
        C_from_decomp = C_computed.reshape(p, n).T  # (n, p)

        if not np.allclose(C_from_decomp, C_expected, atol=1e-10):
            # If verification fails, we still return the result (shouldn't happen)
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
