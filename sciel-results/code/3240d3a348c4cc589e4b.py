"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
_DEADLINE = 115.0  # safety margin under the 120s evaluator limit
_start = time.time()


def _elapsed():
    return time.time() - _start


def _schoolbook(n, m, p):
    """Exact rank n*m*p decomposition (guaranteed correct)."""
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


def _local_verify(u, v, w, ni, nj, nk, tol=1e-9):
    """Verify a block decomposition for local dims (ni,nj,nk).
    u: ni*nj x r, v: nj*nk x r, w: nk*ni x r.
    Constraint: for all i,a,b,k: sum_c w[k*ni+i,c]*u[i*nj+a,c]*v[b*nk+k,c] == (a==b).
    """
    r = len(u[0])
    for i in range(ni):
        for a in range(nj):
            for b in range(nj):
                for k in range(nk):
                    s = 0.0
                    for c in range(r):
                        s += w[k * ni + i][c] * u[i * nj + a][c] * v[b * nk + k][c]
                    if abs(s - (1.0 if a == b else 0.0)) > tol:
                        return False
    return True


def _strassen_2x2x2():
    """Return (u,v,w) for local 2x2x2 Strassen (rank 7).
    u indexed (i,a)->i*2+a, v (b,k)->b*2+k, w (k,i)->k*2+i.
    """
    # (u_entries, v_entries, w_entries); entries are (row, col) in local coords
    # Standard Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)(B00)
    # M2 = (A00)(B01-B11)
    # M3 = (A11)(B10-B00)
    # M4 = (A00+A01)(B11)
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0 - M3 + M5 ; C01 = M2 + M4
    # C10 = M1 + M3 ;     C11 = M0 - M2 + M6
    factors = [
        # M0
        ([(0, 0), (1, 1)], [(0, 0), (1, 1)], [(0, 0), (1, 1)]),
        # M1
        ([(1, 0), (1, 1)], [(0, 0)], [(1, 0)]),
        # M2
        ([(0, 0)], [(0, 1), (1, 1)], [(0, 1), (1, 1)]),
        # M3
        ([(1, 1)], [(1, 0), (0, 0)], [(1, 0), (0, 0)]),
        # M4
        ([(0, 0), (0, 1)], [(1, 1)], [(0, 1)]),
        # M5
        ([(1, 0), (0, 0)], [(0, 0), (0, 1)], [(1, 0)]),
        # M6
        ([(0, 1), (1, 1)], [(1, 0), (1, 1)], [(1, 1)]),
    ]
    # Now apply signs to W to realize C = combinations.
    # We need W[k*2+i, c] such that sum_c W*z_c = C[i,k].
    # Build per-factor W contributions with signs.
    # Represent W as list of (row, col, val).
    w_entries = []
    for c, (ue, ve, we) in enumerate(factors):
        for (i, k) in we:
            w_entries.append((k * 2 + i, c, 1.0))
    # Apply the combination signs:
    # C00 = M0 - M3 + M5  -> rows (k=0,i=0): M0 +, M3 -, M5 +
    # C01 = M2 + M4       -> rows (k=1,i=0): M2 +, M4 +
    # C10 = M1 + M3       -> rows (k=0,i=1): M1 +, M3 +
    # C11 = M0 - M2 + M6  -> rows (k=1,i=1): M0 +, M2 -, M6 +
    # We'll rebuild W cleanly.
    W = [[0] * 7 for _ in range(4)]
    # C00 (i=0,k=0) row index k*2+i = 0
    W[0][0] += 1   # M0
    W[0][3] += -1  # M3
    W[0][5] += 1   # M5
    # C01 (i=0,k=1) row index k*2+i = 2
    W[2][2] += 1   # M2
    W[2][4] += 1   # M4
    # C10 (i=1,k=0) row index k*2+i = 1
    W[1][1] += 1   # M1
    W[1][3] += 1   # M3
    # C11 (i=1,k=1) row index k*2+i = 3
    W[3][0] += 1   # M0
    W[3][2] += -1  # M2
    W[3][6] += 1   # M6

    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    for c, (ue, ve, _we) in enumerate(factors):
        for (i, a) in ue:
            U[i * 2 + a][c] = 1
        for (b, k) in ve:
            V[b * 2 + k][c] = 1
    return U, V, W


def _verify_full(U, V, W, n, m, p, tol=1e-9):
    r = len(U[0])
    for i in range(n):
        for a in range(m):
            for b in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i][c] * U[i * m + a][c] * V[b * p + k][c]
                    if abs(s - (1.0 if a == b else 0.0)) > tol:
                        return False
    return True


def _build_block_decomposition(n, m, p):
    """Partition dimensions into blocks; use Strassen for 2x2x2 blocks,
    schoolbook for the rest. Verify each block locally; fall back to schoolbook
    if any block fails.
    """
    # Block partitions: list of (start, size)
    def partition(total, sizes):
        blocks = []
        s = 0
        for sz in sizes:
            blocks.append((s, sz))
            s += sz
        return blocks

    # Choose sizes: split each dim into 2s then a 1 if odd.
    def sizes_for(total):
        sz = []
        while total >= 2:
            sz.append(2)
            total -= 2
        if total == 1:
            sz.append(1)
        return sz

    I_sizes = sizes_for(n)  # [2,1]
    J_sizes = sizes_for(m)  # [2,2,1]
    K_sizes = sizes_for(p)  # [2,2,2,1]

    I_blocks = partition(n, I_sizes)
    J_blocks = partition(m, J_sizes)
    K_blocks = partition(p, K_sizes)

    # Precompute Strassen local factors (verify once)
    sU, sV, sW = _strassen_2x2x2()
    strassen_ok = _local_verify(sU, sV, sW, 2, 2, 2)

    # Determine total rank
    rank = 0
    for (is_, isz) in I_blocks:
        for (js, jsz) in J_blocks:
            for (ks, ksz) in K_blocks:
                if strassen_ok and isz == 2 and jsz == 2 and ksz == 2:
                    rank += 7
                else:
                    rank += isz * jsz * ksz

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for (is_, isz) in I_blocks:
        for (js, jsz) in J_blocks:
            for (ks, ksz) in K_blocks:
                if strassen_ok and isz == 2 and jsz == 2 and ksz == 2:
                    # Strassen block: local u (2*2=4), v (2*2=4), w (2*2=4)
                    for li in range(2):
                        for la in range(2):
                            gi = is_ + li
                            ga = js + la
                            for c in range(7):
                                val = sU[li * 2 + la][c]
                                if val:
                                    U[gi * m + ga][col + c] += val
                    for lb in range(2):
                        for lk in range(2):
                            gb = js + lb
                            gk = ks + lk
                            for c in range(7):
                                val = sV[lb * 2 + lk][c]
                                if val:
                                    V[gb * p + gk][col + c] += val
                    for lk in range(2):
                        for li in range(2):
                            gk = ks + lk
                            gi = is_ + li
                            for c in range(7):
                                val = sW[lk * 2 + li][c]
                                if val:
                                    W[gk * n + gi][col + c] += val
                    col += 7
                else:
                    # Schoolbook for this block
                    for li in range(isz):
                        for la in range(jsz):
                            for lk in range(ksz):
                                gi = is_ + li
                                ga = js + la
                                gb = js + la
                                gk = ks + lk
                                U[gi * m + ga][col] = 1
                                V[gb * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    global _start, _DEADLINE
    n, m, p = 3, 5, 7
    _start = time.time()

    # Fast: try block decomposition with Strassen (rank 99).
    try:
        if _elapsed() < _DEADLINE - 5:
            cand = _build_block_decomposition(n, m, p)
            if _verify_full(cand["U"], cand["V"], cand["W"], n, m, p):
                return cand
    except Exception:
        pass

    # Fallback: guaranteed-correct schoolbook (rank 105).
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
