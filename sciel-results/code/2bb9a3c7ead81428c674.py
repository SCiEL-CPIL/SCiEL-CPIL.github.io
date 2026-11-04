"""Block-partition + local Strassen decomposition for 3x5 by 5x7 matrix multiplication."""
import time
_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = _START_TIME + (_DEADLINE - (time.time() - _START_TIME))
    if time.time() > deadline:
        deadline = _START_TIME + 5.0
    n, m, p = DIMENSIONS  # 3, 5, 7

    # Schoolbook fallback: rank = n*m*p = 105
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Local Strassen decomposition for a 2x2 x 2x2 x 2x2 block.
    # Local indices: a in {0,1} (A col), b in {0,1} (B row), i in {0,1} (A row), k in {0,1} (B col)
    # U_local[a*2 + i][c] : coefficient of A[i, a] in product c
    # V_local[b*2 + k][c] : coefficient of B[b, k] in product c
    # W_local[k*2 + i][c] : coefficient of product c in C[i, k]
    #
    # Strassen:
    #  M0 = (A00+A11)(B00+B11)
    #  M1 = (A10+A11)B00
    #  M2 = A00(B01-B11)
    #  M3 = A11(B10-B00)
    #  M4 = (A00+A01)B11
    #  M5 = (A10-A00)(B00+B01)
    #  M6 = (A01-A11)(B10+B11)
    #  C00 = M0 - M3 + M5
    #  C01 = M2 + M4
    #  C10 = M1 + M3
    #  C11 = M0 - M2 + M6
    def strassen_local():
        # U_local indexed by [a*2 + i] for (a,i) in {0,1}^2, 7 columns
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        # (a, i, b, k, c, u, v, w) where u,v,w are coefficients
        terms = [
            # M0: (A00+A11)(B00+B11)
            (0, 0, 0, 0, 0, 1, 1, 0),
            (1, 1, 0, 0, 0, 1, 1, 0),
            (0, 0, 1, 1, 0, 1, 1, 0),
            (1, 1, 1, 1, 0, 1, 1, 0),
            # M1: (A10+A11)B00
            (0, 1, 0, 0, 1, 1, 1, 0),
            (1, 1, 0, 0, 1, 1, 1, 0),
            # M2: A00(B01-B11)
            (0, 0, 0, 1, 2, 1, 1, 0),
            (0, 0, 1, 1, 2, 1, -1, 0),
            # M3: A11(B10-B00)
            (1, 1, 1, 0, 3, 1, 1, 0),
            (1, 1, 0, 0, 3, 1, -1, 0),
            # M4: (A00+A01)B11
            (0, 0, 1, 1, 4, 1, 1, 0),
            (1, 0, 1, 1, 4, 1, 1, 0),
            # M5: (A10-A00)(B00+B01)
            (0, 1, 0, 0, 5, 1, 1, 0),
            (0, 1, 0, 1, 5, 1, 1, 0),
            (0, 0, 0, 0, 5, 1, -1, 0),
            (0, 0, 0, 1, 5, 1, -1, 0),
            # M6: (A01-A11)(B10+B11)
            (1, 0, 1, 0, 6, 1, 1, 0),
            (1, 0, 1, 1, 6, 1, 1, 0),
            (1, 1, 1, 0, 6, 1, -1, 0),
            (1, 1, 1, 1, 6, 1, -1, 0),
        ]
        for (a, i, b, k, c, u, v, w) in terms:
            U[a * 2 + i][c] += u
            V[b * 2 + k][c] += v
        # W: C[i,k] = sum_c W[k*2+i, c] M_c
        # C00 = M0 - M3 + M5
        W[0 * 2 + 0][0] += 1   # C[0,0] from M0
        W[0 * 2 + 0][3] += -1  # C[0,0] from M3
        W[0 * 2 + 0][5] += 1   # C[0,0] from M5
        # C01 = M2 + M4
        W[1 * 2 + 0][2] += 1   # C[0,1] from M2
        W[1 * 2 + 0][4] += 1   # C[0,1] from M4
        # C10 = M1 + M3
        W[0 * 2 + 1][1] += 1   # C[1,0] from M1
        W[0 * 2 + 1][3] += 1   # C[1,0] from M3
        # C11 = M0 - M2 + M6
        W[1 * 2 + 1][0] += 1   # C[1,1] from M0
        W[1 * 2 + 1][2] += -1  # C[1,1] from M2
        W[1 * 2 + 1][6] += 1   # C[1,1] from M6
        return U, V, W

    # Verify local Strassen is correct
    import numpy as np
    U_loc, V_loc, W_loc = strassen_local()
    np.random.seed(0)
    A2 = np.random.randint(0, 5, size=(2, 2)).astype(float)
    B2 = np.random.randint(0, 5, size=(2, 2)).astype(float)
    C2 = A2 @ B2
    C2_comp = np.zeros((2, 2))
    for c in range(7):
        za = 0.0
        for i in range(2):
            for a in range(2):
                za += U_loc[a * 2 + i][c] * A2[i, a]
        zb = 0.0
        for b in range(2):
            for k in range(2):
                zb += V_loc[b * 2 + k][c] * B2[b, k]
        for k in range(2):
            for i in range(2):
                C2_comp[i, k] += W_loc[k * 2 + i][c] * za * zb
    if not np.allclose(C2_comp, C2, atol=1e-9):
        # Strassen local is wrong; fall back to schoolbook
        return schoolbook()

    # Block partitions
    # I blocks (n=3): list of (start, size)
    I_blocks = [(0, 2), (2, 1)]
    # J blocks (m=5):
    J_blocks = [(0, 2), (2, 2), (4, 1)]
    # K blocks (p=7):
    K_blocks = [(0, 2), (2, 2), (4, 2), (6, 1)]

    # Accumulate contributions: dict (u_idx, v_idx, w_idx) -> coeff
    contributions = {}

    def add_contrib(u_idx, v_idx, w_idx, coeff):
        if coeff == 0:
            return
        key = (u_idx, v_idx, w_idx)
        contributions[key] = contributions.get(key, 0) + coeff

    for (i_start, i_sz) in I_blocks:
        for (j_start, j_sz) in J_blocks:
            for (k_start, k_sz) in K_blocks:
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    # Use local Strassen
                    U_loc, V_loc, W_loc = strassen_local()
                    for c in range(7):
                        # U side: A[i, a] for i in i_start..i_start+1, a in j_start..j_start+1
                        for li in range(2):
                            for la in range(2):
                                u_coeff = U_loc[la * 2 + li][c]
                                if u_coeff == 0:
                                    continue
                                gi = i_start + li
                                gj = j_start + la
                                u_idx = gi * m + gj
                                # V side: B[b, k] for b in j_start..j_start+1, k in k_start..k_start+1
                                for lb in range(2):
                                    for lk in range(2):
                                        v_coeff = V_loc[lb * 2 + lk][c]
                                        if v_coeff == 0:
                                            continue
                                        gb = j_start + lb
                                        gk = k_start + lk
                                        v_idx = gb * p + gk
                                        # W side: C[gi, gk]
                                        w_coeff = W_loc[lk * 2 + li][c]
                                        if w_coeff == 0:
                                            continue
                                        w_idx = gk * n + gi
                                        add_contrib(u_idx, v_idx, w_idx, u_coeff * v_coeff * w_coeff)
                else:
                    # Schoolbook for this block
                    for li in range(i_sz):
                        for lj in range(j_sz):
                            for lk in range(k_sz):
                                gi = i_start + li
                                gj = j_start + lj
                                gk = k_start + lk
                                u_idx = gi * m + gj
                                v_idx = gj * p + gk
                                w_idx = gk * n + gi
                                add_contrib(u_idx, v_idx, w_idx, 1)

    rank = len(contributions)

    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_idx, v_idx, w_idx) in enumerate(sorted(contributions.keys())):
        coeff = contributions[(u_idx, v_idx, w_idx)]
        U[u_idx][col] = coeff
        V[v_idx][col] = coeff
        W[w_idx][col] = coeff

    # Verify with numpy
    import numpy as np
    np.random.seed(42)
    A_test = np.random.randint(0, 10, size=(n, m)).astype(float)
    B_test = np.random.randint(0, 10, size=(m, p)).astype(float)
    C_test = A_test @ B_test

    C_comp = np.zeros((n, p))
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)

    for c in range(rank):
        za = 0.0
        for i in range(n):
            for j in range(m):
                za += U_arr[i * m + j, c] * A_test[i, j]
        zb = 0.0
        for j in range(m):
            for k in range(p):
                zb += V_arr[j * p + k, c] * B_test[j, k]
        for k in range(p):
            for i in range(n):
                C_comp[i, k] += W_arr[k * n + i, c] * za * zb

    if not np.allclose(C_comp, C_test, atol=1e-9):
        # Decomposition is wrong, fall back to schoolbook
        return schoolbook()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
