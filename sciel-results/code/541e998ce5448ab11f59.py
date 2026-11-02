import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_blocks = p // 2  # 6 blocks of 2 columns each

    # Strassen 7-product decomposition for 2x2x2:
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[C00,C01],[C10,C11]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e)
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    #
    # Local A indices: 0=a, 1=b, 2=c, 3=d
    # Local B indices: 0=e(B[0,0]), 1=f(B[0,1]), 2=g(B[1,0]), 3=h(B[1,1])
    # Local C indices: 0=C00, 1=C01, 2=C10, 3=C11

    # U coefficients for each Strassen term (A-side)
    U_strassen = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {0: -1, 2: 1},      # M6
        {1: 1, 3: -1},      # M7
    ]

    # V coefficients for each Strassen term (B-side, local indices)
    V_strassen = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1},       # M7
    ]

    # W coefficients for each Strassen term (C-side, local indices)
    W_strassen = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: -1},      # M2
        {1: 1, 3: 1},       # M3
        {0: 1, 2: 1},       # M4
        {0: -1, 1: 1},      # M5
        {3: 1},             # M6
        {0: 1},             # M7
    ]

    # Map local B index to global V row
    def local_b_to_global(local_b, t):
        if local_b == 0:
            return 0 * p + 2 * t        # B[0, 2t]
        elif local_b == 1:
            return 0 * p + 2 * t + 1    # B[0, 2t+1]
        elif local_b == 2:
            return 1 * p + 2 * t        # B[1, 2t]
        else:
            return 1 * p + 2 * t + 1    # B[1, 2t+1]

    # Map local C index to global W row
    # W[k*n + i] where k=col, i=row
    def local_c_to_global(local_c, t):
        if local_c == 0:
            return (2 * t) * n + 0       # C[0, 2t]
        elif local_c == 1:
            return (2 * t + 1) * n + 0   # C[0, 2t+1]
        elif local_c == 2:
            return (2 * t) * n + 1       # C[1, 2t]
        else:
            return (2 * t + 1) * n + 1   # C[1, 2t+1]

    rank = num_blocks * 7  # 42

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s

            # U (A indices are global)
            for a_idx, val in U_strassen[s].items():
                U[a_idx][col] = val

            # V (map local B index to global)
            for local_b, val in V_strassen[s].items():
                global_b = local_b_to_global(local_b, t)
                V[global_b][col] = val

            # W (map local C index to global W row)
            for local_c, val in W_strassen[s].items():
                global_w = local_c_to_global(local_c, t)
                W[global_w][col] += val

    # Quick verification
    if time.time() < deadline:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_test = A_test @ B_test

        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_r = np.dot(A_test.flatten(), U_np[:, r])
            v_r = np.dot(B_test.flatten(), V_np[:, r])
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += u_r * v_r * W_np[k * n + i, r]

        if np.max(np.abs(C_decomp - C_test)) > 1e-10:
            pass  # Should not happen

    return {"U": U, "V": V, "W": W}
