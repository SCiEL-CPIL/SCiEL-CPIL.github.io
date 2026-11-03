import numpy as np

N, M, P = 5, 5, 5


def _strassen_pairs(ia, ib, ja, jb):
    """Yield (U_entries, V_entries, W_entries) for Strassen 2x2
    multiplying sub-block rows ia..ib x cols ja..jb."""
    # local 2x2 -> global mapping
    ai = [ia, ib]  # A row indices (global i)
    aj = [ja, jb]  # A col indices (global j) = B row indices

    def e(d):
        return d

    out = []
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    u = {(ai[0], aj[0]): 1, (ai[1], aj[1]): 1}
    v = {(aj[0], aj[0]): 1, (aj[1], aj[1]): 1}
    w = {(ai[0], aj[0]): 1, (ai[1], aj[1]): 1}
    out.append((u, v, w))
    # M2 = (A10+A11) B00 -> C10 += M2, C11 -= M2
    u = {(ai[1], aj[0]): 1, (ai[1], aj[1]): 1}
    v = {(aj[0], aj[0]): 1}
    w = {(ai[1], aj[0]): 1, (ai[1], aj[1]): -1}
    out.append((u, v, w))
    # M3 = A00 (B01-B11) -> C01 += M3
    u = {(ai[0], aj[0]): 1}
    v = {(aj[0], aj[1]): 1, (aj[1], aj[1]): -1}
    w = {(ai[0], aj[1]): 1}
    out.append((u, v, w))
    # M4 = A11 (B10-B00) -> C10 -= M4
    u = {(ai[1], aj[1]): 1}
    v = {(aj[1], aj[0]): 1, (aj[0], aj[0]): -1}
    w = {(ai[1], aj[0]): -1}
    out.append((u, v, w))
    # M5 = (A00+A01) B11 -> C00 -= M5, C10 += M5
    u = {(ai[0], aj[0]): 1, (ai[0], aj[1]): 1}
    v = {(aj[1], aj[1]): 1}
    w = {(ai[0], aj[0]): -1, (ai[1], aj[0]): 1}
    out.append((u, v, w))
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    u = {(ai[1], aj[0]): 1, (ai[0], aj[0]): -1}
    v = {(aj[0], aj[0]): 1, (aj[0], aj[1]): 1}
    w = {(ai[1], aj[1]): 1}
    out.append((u, v, w))
    # M7 = (A01-A11)(B10+B11) -> C01 -= M7
    u = {(ai[0], aj[1]): 1, (ai[1], aj[1]): -1}
    v = {(aj[1], aj[0]): 1, (aj[1], aj[1]): 1}
    w = {(ai[0], aj[1]): -1}
    out.append((u, v, w))
    return out


def _strassen3x3(ii, jj):
    """Yield (U_entries, V_entries, W_entries) for a 3x3 block (indices ii, jj = [a,b,c])
    using 2+1 split: top-left 2x2 via Strassen, plus cross terms and bottom-right 1x1."""
    a, b, c = ii
    pairs = _strassen_pairs(a, b, jj[0], jj[1])
    out = list(pairs)
    # Cross terms: C_ab gets A_ab B_bc; A has i in {a,b}, j=c; B has j=c, k in {a,b}; C has i in {a,b}, k in {a,b}
    for r in (a, b):
        for s in (a, b):
            out.append((
                {(r, c): 1},
                {(jj[2], jj[0] if s == a else jj[1]): 1},
                {(jj[0] if s == a else jj[1], r): 1},
            ))
    # C_ac (i in {a,b}, k=c): A_a{a,b} B_{j=c,k=c}; A_{i in {a,b},j=c} B_{c,c}
    for r in (a, b):
        for s in (jj[0], jj[1]):
            out.append((
                {(r, s): 1},
                {(s, jj[2]): 1},
                {(jj[2], r): 1},
            ))
        # A_{i in {a,b},j=c} * B_{c,c}
        out.append((
            {(r, c): 1},
            {(jj[2], jj[2]): 1},
            {(jj[2], r): 1},
        ))
    # C_ca (i=c, k in {a,b}): A_{c, j in {a,b}} B_{j, k in {a,b}}; A_{c,c} B_{c, k in {a,b}}
    for s in (jj[0], jj[1]):
        for t in (jj[0], jj[1]):
            out.append((
                {(c, s): 1},
                {(s, t): 1},
                {(t, c): 1},
            ))
        # A_{c,c} B_{c,k in {a,b}}
        out.append((
            {(c, c): 1},
            {(jj[2], jj[0] if t == jj[0] else jj[1]): 1},
            {(jj[0] if t == jj[0] else jj[1], c): 1},
        ))
    # C_cc (i=c, k=c): A_{c, j in {a,b}} B_{j, c}; A_{c,c} B_{c,c}
    for s in (jj[0], jj[1]):
        out.append((
            {(c, s): 1},
            {(s, jj[2]): 1},
            {(jj[2], c): 1},
        ))
    out.append((
        {(c, c): 1},
        {(jj[2], jj[2]): 1},
        {(jj[2], c): 1},
    ))
    return out


def _schoolbook(i_list, j_list, k_list, U_acc, V_acc, W_acc):
    """Append rank terms for C[i_list,k_list] += A[i_list,j_list] * B[j_list,k_list] via schoolbook."""
    for i in i_list:
        for j in j_list:
            for k in k_list:
                U_acc.append((i * N + j, 1))
                V_acc.append((j * M + k, 1))
                W_acc.append((k * N + i, 1))


def _emit(entries, col):
    """Yield (global_index, value) tuples where global index = row*5 + second index."""
    for (a, b), v in entries.items():
        yield (a, b), v


def construct():
    S = [0, 1]
    L = [2, 3, 4]

    U_rows = []
    V_rows = []
    W_rows = []

    def add_triplet(u_entries, v_entries, w_entries):
        # u_entries: list of (global_row, val)
        # We accumulate per-row lists to build a column vector later.
        U_rows.append(u_entries)
        V_rows.append(v_entries)
        W_rows.append(w_entries)

    def from_dicts(u_d, v_d, w_d):
        u_e = [(i * N + j, val) for (i, j), val in u_d.items()]
        v_e = [(j * M + k, val) for (j, k), val in v_d.items()]
        w_e = [(k * N + i, val) for (k, i), val in w_d.items()]
        add_triplet(u_e, v_e, w_e)

    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_triplet([(i * N + j, 1)], [(j * M + k, 1)], [(k * N + i, 1)])

    # --- C_SS (S x S) = A_SS B_SS + A_SL B_LS ---
    for (u_d, v_d, w_d) in _strassen_pairs(S[0], S[1], S[0], S[1]):
        from_dicts(u_d, v_d, w_d)
    schoolbook_block(S, L, S)

    # --- C_SL (S x L) = A_SS B_SL + A_SL B_LL ---
    schoolbook_block(S, S, L)
    schoolbook_block(S, L, L)

    # --- C_LS (L x S) = A_LS B_SS + A_LL B_LS ---
    schoolbook_block(L, S, S)
    schoolbook_block(L, L, S)

    # --- C_LL (L x L) = A_LS B_SL + A_LL B_LL ---
    schoolbook_block(L, S, L)
    for (u_d, v_d, w_d) in _strassen3x3(L, L):
        from_dicts(u_d, v_d, w_d)

    # Build factor matrices of shape (25, rank) with sparse entries
    rank = len(U_rows)
    U = np.zeros((N * M, rank), dtype=float)
    V = np.zeros((M * P, rank), dtype=float)
    W = np.zeros((P * N, rank), dtype=float)
    for r in range(rank):
        for (row, val) in U_rows[r]:
            U[row, r] = val
        for (row, val) in V_rows[r]:
            V[row, r] = val
        for (row, val) in W_rows[r]:
            W[row, r] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
