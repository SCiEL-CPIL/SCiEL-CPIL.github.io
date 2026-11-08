import numpy as np
import time

DIMENSIONS = [4, 4, 6]


def _nz(d):
    """Return list of (idx, coeff) for non-zero entries of dict d."""
    return [(i, c) for i, c in d.items() if c != 0]


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s (evaluator kills at 150s)

    n, m, p = DIMENSIONS  # n=4, m=4, p=6

    # Top-level Strassen for (2x2) x (2x3), 7 terms.
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},   # A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},   # A00
        {0: 0, 1: 0, 2: 0, 3: 1},   # A11
        {0: 1, 1: 1, 2: 0, 3: 0},   # A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0},  # A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1},  # A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},   # B00
        {0: 0, 1: 1, 2: 0, 3: -1},  # B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0},  # B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},   # B11
        {0: 1, 1: 1, 2: 0, 3: 0},   # B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1},   # B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    # Precompute non-zero coefficient lists for speed and clarity.
    alpha_nz = [_nz(d) for d in alpha]
    beta_nz = [_nz(d) for d in beta]

    # Index helpers.
    def u_idx(ar, br, x_row, x_col):
        return (ar * 2 + x_row) * 4 + (br * 2 + x_col)

    def v_idx(ac, bc, y_row, y_inner_col):
        return (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)

    def w_idx(a, b, za, zb):
        return (b * 3 + zb) * 4 + (a * 2 + za)

    R = 77
    U = np.zeros((16, R), dtype=int)
    V = np.zeros((24, R), dtype=int)
    W = np.zeros((24, R), dtype=int)

    col = 0
    for s in range(7):
        if time.time() > deadline:
            break
        a_s = alpha_nz[s]
        b_s = beta_nz[s]
        rc_s = recon[s]

        # Part 1: inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            a_si = alpha_nz[si]
            b_si = beta_nz[si]
            rc_si = recon[si]

            # U entries
            for (a, ca) in a_si:
                x_row, x_col = a // 2, a % 2
                for (r, cr) in a_s:
                    ar, br = r // 2, r % 2
                    U[u_idx(ar, br, x_row, x_col), col] += ca * cr

            # V entries
            for (b, cb) in b_si:
                y_row, y_inner_col = b // 2, b % 2
                for (c, cc) in b_s:
                    ac, bc = c // 2, c % 2
                    V[v_idx(ac, bc, y_row, y_inner_col), col] += cb * cc

            # W entries
            for (a, b, gamma) in rc_s:
                for (za, zb, zgamma) in rc_si:
                    W[w_idx(a, b, za, zb), col] += gamma * zgamma

            col += 1

        # Part 2: schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for (r, cr) in a_s:
                    ar, br = r // 2, r % 2
                    U[u_idx(ar, br, u, v), col] += cr

                for (c, cc) in b_s:
                    ac, bc = c // 2, c % 2
                    V[v_idx(ac, bc, v, 2), col] += cc

                for (a, b, gamma) in rc_s:
                    W[(b * 3 + 2) * 4 + (a * 2 + u), col] += gamma

                col += 1

    def to_list(mat):
        out = []
        for row in mat:
            r = []
            for x in row:
                xi = int(round(float(x)))
                if abs(float(x) - xi) < 1e-9:
                    r.append(xi)
                else:
                    r.append(float(x))
            out.append(r)
        return out

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
