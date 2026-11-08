import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# Strassen rank-7 decomposition of M_{2,2,2}
# alpha_s[i,j] (i=row of A, j=col of A), beta_s[j,k] (j=row of B, k=col of B)
# recon_s: list of (i, k, gamma) meaning C[i,k] += gamma * (product)
ALPHA = [
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [-1, 0, 1, 0],
    [0, 1, 0, -1]
]
BETA = [
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, -1],
    [-1, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1]
]
RECON = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, -1)],
    [(0, 1, 1), (1, 1, 1)],
    [(0, 0, 1), (1, 0, 1)],
    [(0, 0, -1), (0, 1, 1)],
    [(1, 1, 1)],
    [(0, 0, 1)]
]


def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6

    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s/150s kill

    # True Kronecker product: M_{4,4,6} = M_{2,2,3} ⊗ M_{2,2,2}
    #   First factor  M_{2,2,3}:  A1(2x2) * B1(2x3) = C1(2x3),  rank 6
    #   Second factor M_{2,2,2}:  A2(2x2) * B2(2x2) = C2(2x2),  rank 7
    #   Total rank = 6 * 7 = 42
    #
    # Index maps (A row = (i1,i2), A col = (j1,j2), B row = (j1,j2),
    #             B col = (k1,k2), C row = (i1,i2), C col = (k1,k2)):
    #   U idx = (i1*2+i2)*4 + (j1*2+j2)
    #   V idx = (j1*2+j2)*6 + (k1*3+k2)
    #   W idx = (i1*2+i2)*6 + (k1*3+k2)
    #
    # For block s = (s1, s2):
    #   U[(i1,j1,i2,j2), s] = ALPHA[s1][i1,j1] * ALPHA[s2][i2,j2]
    #   V[(j1,k1,j2,k2), s] = BETA[s1][j1,k1]  * BETA[s2][j2,k2]
    #   W[(i1,k1,i2,k2), s] = sum_{(i1,k1) in RECON[s1]} g1
    #                          * sum_{(i2,k2) in RECON[s2]} g2

    R = 42
    U = np.zeros((16, R), dtype=np.float64)
    V = np.zeros((24, R), dtype=np.float64)
    W = np.zeros((24, R), dtype=np.float64)

    col = 0
    for s1 in range(7):
        for s2 in range(7):
            # ---- U: outer product of two alpha vectors ----
            for i1 in range(2):
                for j1 in range(2):
                    a1 = ALPHA[s1][i1 * 2 + j1]
                    if a1 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            a2 = ALPHA[s2][i2 * 2 + j2]
                            if a2 == 0:
                                continue
                            u_idx = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            U[u_idx, col] += a1 * a2

            # ---- V: outer product of two beta vectors ----
            for j1 in range(2):
                for k1 in range(3):
                    b1 = BETA[s1][j1 * 3 + k1]
                    if b1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            b2 = BETA[s2][j2 * 2 + k2]
                            if b2 == 0:
                                continue
                            v_idx = (j1 * 2 + j2) * 6 + (k1 * 3 + k2)
                            V[v_idx, col] += b1 * b2

            # ---- W: tensor (outer) product of two recon lists ----
            for (i1, k1, g1) in RECON[s1]:
                for (i2, k2, g2) in RECON[s2]:
                    w_idx = (i1 * 2 + i2) * 6 + (k1 * 3 + k2)
                    W[w_idx, col] += g1 * g2

            col += 1

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    def to_list(mat):
        out = []
        for row in mat:
            r = []
            for x in row:
                rx = round(x)
                r.append(int(rx) if abs(x - rx) < 1e-9 else x)
            out.append(r)
        return out

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
