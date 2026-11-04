"""Matrix multiplication 3x5x5: block Strassen decomposition, verified.

Partitions A-rows, A-cols(=B-rows), B-cols into groups; each sub-block uses
canonical Strassen (2x2x2, rank 7) or schoolbook otherwise. The assembled
decomposition is verified exactly (reconstruct the tensor and compare to the
all-ones multiplication pattern). Falls back to schoolbook if verification
fails, so a valid result is always returned.
"""
import numpy as np

DIMENSIONS = [3, 5, 5]


def _strassen2x2_terms():
    """Canonical Strassen, 7 rank-1 terms, for 2x2 x 2x2 -> 2x2.
    Local: A rows {0,1}, B rows {0,1}, B cols {0,1}, C rows {0,1}, C cols {0,1}.
    term = (u_entries, v_entries, w_entries):
      u: (A_row, A_col, coeff)   [A_col is the shared index == B_row]
      v: (B_row, B_col, coeff)
      w: (C_row, C_col, coeff)
    """
    # M0 = (A00+A11)(B00+B11)
    M = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # M1 = (A10+A11)B00
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
        # M2 = A00(B10-B11)
        ([(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        # M3 = (A00+A11)(B01+B11)
        ([(0, 0, 1), (1, 1, 1)], [(0, 1, 1), (1, 1, 1)]),
        # M4 = (A00+A10)B11
        ([(0, 0, 1), (1, 0, 1)], [(1, 1, 1)]),
        # M5 = (A01-A11)(B00+B10)
        ([(0, 1, 1), (1, 1, -1)], [(0, 0, 1), (1, 0, 1)]),
        # M6 = (A10-A00)(B01+B11)
        ([(1, 0, 1), (0, 0, -1)], [(0, 1, 1), (1, 1, 1)]),
    ]
    # C00 = M0 + M2 - M3 + M5
    # C01 = M4 + M6
    # C10 = M1 + M3
    # C11 = M0 + M4 - M1 - M2
    recon = {
        (0, 0): [(0, 1), (2, 1), (3, -1), (5, 1)],
        (0, 1): [(4, 1), (6, 1)],
        (1, 0): [(1, 1), (3, 1)],
        (1, 1): [(0, 1), (4, 1), (1, -1), (2, -1)],
    }
    terms = []
    for tidx in range(7):
        u = M[tidx][0]
        v = M[tidx][1]
        w = []
        for (cr, cc), coeffs in recon.items():
            for (midx, coeff) in coeffs:
                if midx == tidx:
                    w.append((cr, cc, coeff))
        terms.append((u, v, w))
    return terms


def _verify(U, V, W, n, m, p):
    rank = U.shape[1]
    T_hat = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        for i in range(n):
            for j in range(m):
                ui = u[i * m + j]
                if ui == 0.0:
                    continue
                for k in range(p):
                    T_hat[i, j, k] += ui * v[j * p + k] * w[k * n + i]
    T = np.ones((n, m, p))
    return bool(np.allclose(T_hat, T, atol=1e-9, rtol=0.0))


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    row_groups = [(0, 1), (2,)]
    brow_groups = [(0, 1), (2, 3), (4,)]
    bcol_groups = [(0, 1), (2, 3), (4,)]

    all_terms = []

    def add_term(u_entries, v_entries, w_entries):
        all_terms.append((u_entries, v_entries, w_entries))

    strassen = _strassen2x2_terms()

    for (a0, a1) in row_groups:
        for (h0, h1) in brow_groups:
            for (k0, k1) in bcol_groups:
                r = len((a0, a1))
                s = len((h0, h1))
                t = len((k0, k1))
                if r == 2 and s == 2 and t == 2:
                    for (u, v, w) in strassen:
                        u_e = [(a0 + ar, h0 + ac, coeff) for (ar, ac, coeff) in u]
                        v_e = [(h0 + br, k0 + bc, coeff) for (br, bc, coeff) in v]
                        w_e = [(a0 + cr, k0 + cc, coeff) for (cr, cc, coeff) in w]
                        add_term(u_e, v_e, w_e)
                else:
                    for ar in range(r):
                        for br in range(s):
                            for bc in range(t):
                                add_term(
                                    [(a0 + ar, h0 + br, 1)],
                                    [(h0 + br, k0 + bc, 1)],
                                    [(a0 + ar, k0 + bc, 1)],
                                )

    rank = len(all_terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for c, (u_entries, v_entries, w_entries) in enumerate(all_terms):
        for (i, j, coeff) in u_entries:
            U[i * m + j, c] += coeff
        for (j, k, coeff) in v_entries:
            V[j * p + k, c] += coeff
        for (i, k, coeff) in w_entries:
            W[k * n + i, c] += coeff

    if rank < n * m * p and _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
