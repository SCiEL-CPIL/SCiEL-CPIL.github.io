import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen 2x2x2 coefficients
    # A blocks: 0=00, 1=01, 2=10, 3=11
    # B blocks: 0=00, 1=01, 2=10, 3=11
    alpha_s = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta_s = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon_s = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2 (cols 0,1 of Y_s)
        for si in range(7):
            # U
            for a in range(4):
                ia = alpha_s[si].get(a, 0)
                if ia != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        sa = alpha_s[s].get(r, 0)
                        if sa != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += ia * sa
            # V
            for b in range(4):
                ib = beta_s[si].get(b, 0)
                if ib != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        sb = beta_s[s].get(c, 0)
                        if sb != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += ib * sb
            # W
            for a, b, gamma in recon_s[s]:
                for za, zb, zgamma in recon_s[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1
            
        # Part 2: Schoolbook for 2x2 x 2x1 (col 2 of Y_s)
        for u in range(2):
            for v in range(2):
                # U
                for r in range(4):
                    sa = alpha_s[s].get(r, 0)
                    if sa != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += sa
                # V
                for c in range(4):
                    sb = beta_s[s].get(c, 0)
                    if sb != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += sb
                # W
                for a, b, gamma in recon_s[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                col += 1

    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                r = round(x)
                if abs(x - r) < 1e-9:
                    new_row.append(int(r))
                else:
                    r2 = round(2*x) / 2
                    if abs(x - r2) < 1e-9:
                        new_row.append(r2)
                    else:
                        new_row.append(float(x))
            result.append(new_row)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
