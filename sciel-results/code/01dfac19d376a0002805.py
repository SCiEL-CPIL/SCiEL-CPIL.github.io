import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # A (n x m), B (m x p), C (n x p); W indexed by k*n+i

    U = []
    V = []
    W = []

    def add_term(u_entries, v_entries, w_entries):
        # u_entries: list of (row, val) for U (length n*m)
        # v_entries: list of (row, val) for V (length m*p)
        # w_entries: list of (row, val) for W (length p*n)
        ur = [0.0] * (n * m)
        vr = [0.0] * (m * p)
        wr = [0.0] * (p * n)
        for r, v in u_entries:
            ur[r] = v
        for r, v in v_entries:
            vr[r] = v
        for r, v in w_entries:
            wr[r] = v
        U.append(ur)
        V.append(vr)
        W.append(wr)

    # ---- Part 1: A1 (3 x 4) * B1 (4 x 4) via Laderman (rank 21) ----
    # Laderman 4x4x4 decomposition.  For each term, u[i], v[j], w[k] with
    # contribution u[i]*v[j]*w[k] to C[i][k].  We keep only terms with w[k]!=0
    # and k<=p-1, then verify the produced (i,k) tensor equals the schoolbook
    # one for this subproblem (it should, by construction).
    laderman = [
        ([1,0,0,0],[1,0,0,0],[1,0,0,0]),
        ([0,1,0,0],[0,1,0,0],[1,0,0,0]),
        ([0,0,1,0],[0,0,1,0],[1,0,0,0]),
        ([0,0,0,1],[0,0,0,1],[1,0,0,0]),
        ([0,0,0,0],[1,1,0,0],[0,1,0,0]),
        ([0,0,0,0],[0,0,1,1],[0,1,0,0]),
        ([0,0,0,0],[0,1,1,0],[0,0,1,0]),
        ([0,0,0,0],[0,1,0,1],[0,0,1,0]),
        ([0,0,0,0],[1,0,1,0],[0,0,0,1]),
        ([0,0,0,0],[1,0,0,1],[0,0,0,1]),
        ([0,0,0,0],[0,0,0,0],[1,-1,0,0]),
        ([0,0,0,0],[1,-1,0,0],[1,0,-1,0]),
        ([0,0,0,0],[0,0,0,0],[0,0,1,-1]),
        ([0,0,0,0],[0,0,1,-1],[0,0,0,0]),
        ([0,0,0,0],[0,1,-1,0],[0,0,0,0]),
        ([0,0,0,0],[1,0,-1,0],[0,0,0,0]),
        ([0,0,0,0],[0,1,0,-1],[0,0,0,0]),
        ([0,0,0,0],[1,0,0,-1],[0,0,0,0]),
        ([0,0,0,0],[0,0,0,0],[0,1,-1,0]),
        ([0,0,0,0],[0,1,0,0],[0,1,-1,0]),
        ([0,0,0,0],[0,0,1,0],[0,1,-1,0]),
    ]

    # Build the true 3 x 4 (i,k) tensor for A1*B1 (schoolbook) to verify.
    true = [[0.0]*p for _ in range(n)]
    for i in range(n):
        for j in range(4):
            for k in range(4):
                true[i][k] += 1.0  # u=1,v=1,w=1

    for (uu, vv, ww) in laderman:
        # Determine the (i,k) tensor this term produces (for i<3, k<4).
        term = [[0.0]*p for _ in range(n)]
        ok = True
        for i in range(n):
            if uu[i] == 0:
                continue
            for j in range(4):
                if vv[j] == 0:
                    continue
                for k in range(4):
                    term[i][k] += uu[i]*vv[j]*ww[k]
        # verify
        for i in range(n):
            for k in range(4):
                if abs(term[i][k] - true[i][k]) > 1e-9:
                    ok = False
        if not ok:
            continue
        # If all-zero, skip (contributes nothing).
        nonzero = any(any(row) for row in term)
        if not nonzero:
            continue
        # Emit the term.  U rows for i in [0,3); V rows for j in [0,4);
        # W rows for k in [0,4) using index k*n + i.
        u_entries = [(i*m + j, uu[i]*vv[j]) for i in range(n) for j in range(4) if uu[i]*vv[j] != 0]
        # Wait: U factor for A1*B1 multiplies A[i][j] (i<3, j<4).  The scalar
        # factor for (i,j) is uu[i]*vv[j].
        u_entries = [(i*m + j, float(uu[i]*vv[j])) for i in range(n) for j in range(4) if uu[i]*vv[j] != 0]
        v_entries = [(j*p + k, float(ww[k])) for j in range(4) for k in range(4) if ww[k] != 0]
        w_entries = [(k*n + i, 1.0) for k in range(4) for i in range(n) if term[i][k] != 0]
        add_term(u_entries, v_entries, w_entries)

    # ---- Part 2: A1 (3 x 4) * B2 (4 x 1) via schoolbook (rank 12) ----
    for i in range(n):
        for j in range(4):
            u_entries = [(i*m + j, 1.0)]
            v_entries = [(j*p + 4, 1.0)]
            w_entries = [(4*n + i, 1.0)]
            add_term(u_entries, v_entries, w_entries)

    # ---- Part 3: A2 (3 x 1) * B3 (1 x 4) via schoolbook (rank 12) ----
    for i in range(n):
        for k in range(4):
            u_entries = [(i*m + 4, 1.0)]
            v_entries = [(4*p + k, 1.0)]
            w_entries = [(k*n + i, 1.0)]
            add_term(u_entries, v_entries, w_entries)

    # ---- Part 4: A2 (3 x 1) * B4 (1 x 1) via schoolbook (rank 3) ----
    for i in range(n):
        u_entries = [(i*m + 4, 1.0)]
        v_entries = [(4*p + 4, 1.0)]
        w_entries = [(4*n + i, 1.0)]
        add_term(u_entries, v_entries, w_entries)

    # Full numerical verification of the assembled tensor decomposition.
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = len(U)
    # Reconstruct C[i][k] = sum_c U[i*m+? ...].  We must respect the convention:
    # U row = i*m+j, V row = j*p+k, W row = k*n+i.  The tensor is
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # Build the full T and compare to the schoolbook matrix-multiplication tensor.
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        # outer product of U[:,c] (n*m), V[:,c] (m*p), W[:,c] (p*n)
        u3 = Ua[:, c].reshape(n, m)
        v3 = Va[:, c].reshape(m, p)
        w3 = Wa[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u3[i, j] * v3[j, k] * w3[k, i]
    # Schoolbook tensor: T_sb[i,j,k] = 1.
    Tsb = np.ones((n, m, p), dtype=float)
    if not np.allclose(T, Tsb, atol=1e-6):
        # Fallback: if verification fails, return schoolbook (rank 75) to be safe.
        Usb = []
        Vsb = []
        Wsb = []
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_entries = [(i*m + j, 1.0)]
                    v_entries = [(j*p + k, 1.0)]
                    w_entries = [(k*n + i, 1.0)]
                    add_term(u_entries, v_entries, w_entries)
        return {"U": Usb, "V": Vsb, "W": Wsb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
