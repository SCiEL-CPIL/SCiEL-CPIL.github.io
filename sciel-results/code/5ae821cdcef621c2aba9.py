"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time as _t
import itertools as _it

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_TMAX = 150.0
_TSTART = _t.perf_counter()


def _now():
    return _t.perf_counter()


def _left():
    return _TMAX - 8.0 - (_now() - _TSTART)


# ------------------------------------------------------------------
# Verification against the 6x6 matrix-multiplication tensor.
# ------------------------------------------------------------------
def _mm_tensor():
    n = m = p = 6
    T = [[[0] * p for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i][j][k] = 1
    return T


def _check_factors(n, m, p, U, V, W):
    rank = len(U[0]) if U else 0
    T = _mm_tensor()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0.0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for r in range(rank):
                    acc += U[ui][r] * V[vj][r] * W[wk][r]
                exp = T[i][j][k]
                if acc is None or abs(acc - exp) > 1e-6:
                    return False
    return True


def _schoolbook():
    n = m = p = 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


# ------------------------------------------------------------------
# Factor-3 integer/half-integer search on the rank-3 multiplication
# tensor (entries: 0, 1, or 1/2). Verified against the small tensor.
# ------------------------------------------------------------------
def _val(x):
    return x / 2.0


def _mm3_target():
    n = m = p = 3
    T = [[[0] * p for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i][j][k] = 1
    return T


def _check_small(U, V, W):
    n = m = p = 3
    rank = len(U[0]) if U else 0
    T = _mm3_target()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0.0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for r in range(rank):
                    acc += U[ui][r] * V[vj][r] * W[wk][r]
                if abs(acc - T[i][j][k]) > 1e-9:
                    return False
    return True


def _try_r3(r3, seed, tmax):
    import numpy as np
    n = m = p = 3
    R = np.random.default_rng(seed)
    dom = [-1, 0, 1, 2]
    domv = [_val(x) for x in dom]

    def randcol():
        return np.array([domv[np.random.default_rng().choice(len(dom))]
                         if False else domv[R.integers(len(dom))] for _ in range(9)],
                        dtype=float)

    def lin_eq(colU, colV, colW):
        rows = []
        rhs = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    lhs = 0.0
                    for i2 in range(3):
                        for j2 in range(3):
                            if i == i2 or j == j2:
                                continue
                            lhs += colU[i2 * 3 + j] * colV[j * 3 + k] * colW[k * 3 + i2]
                    rows.append(lhs)
                    rhs.append(1.0)
        a = np.array(rows)
        return a, np.array(rhs)

    def residual(colU, colV, colW):
        A, b = lin_eq(colU, colV, colW)
        x, res, rank_, sv = np.linalg.lstsq(A, b, rcond=None)
        if x.size == 0:
            return np.inf
        return float(np.linalg.norm(A @ x - b))

    best = None
    best_res = np.inf
    it = 0
    while _now() - tmax < 0.0:
        it += 1
        colU = randcol()
        colV = randcol()
        # try to pick a small colW so the 3 columns are linearly independent
        ok = 0
        while ok < 40:
            colW = randcol()
            M = np.column_stack([colU, colV, colW])
            sv = np.linalg.svd(M, compute_uv=False)
            if sv[2] > 1e-3:
                break
            ok += 1
        A, b = lin_eq(colU, colV, colW)
        try:
            x, res, rank_, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if x.size == 0:
            continue
        r0 = float(np.linalg.norm(A @ x - b))
        if not (r0 < best_res):
            continue
        # local refinement: flip a random entry in colW, resolve
        improved = True
        steps = 0
        while improved and steps < 60 and _now() - tmax < 0.0:
            improved = False
            steps += 1
            # pick a random position in colW
            idx = int(R.integers(9))
            old = colW[idx]
            for d in dom:
                if _val(d) == old:
                    continue
                colW2 = colW.copy()
                colW2[idx] = _val(d)
                M = np.column_stack([colU, colV, colW2])
                sv = np.linalg.svd(M, compute_uv=False)
                if sv[2] < 1e-3:
                    continue
                A2, b2 = lin_eq(colU, colV, colW2)
                try:
                    x2, r2, rank2, sv2 = np.linalg.lstsq(A2, b2, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                if x2.size == 0:
                    continue
                rr = float(np.linalg.norm(A2 @ x2 - b2))
                if rr < best_res - 1e-12:
                    colW = colW2
                    x = x2
                    best_res = rr
                    improved = True
                    if best_res < 1e-6:
                        colU_i = [int(round(colU[t] * 2)) for t in range(9)]
                        colV_i = [int(round(colV[t] * 2)) for t in range(9)]
                        colW_i = [int(round(colW[t] * 2)) for t in range(9)]
                        x_i = [int(round(val * 2)) for val in x]
                        return [colU_i, colV_i, colW_i, x_i], best_res
            colW[idx] = old
        # also try flipping colU / colV entries occasionally
        if best_res < 0.02 and _left() > 10:
            for which, col in ((0, colU), (1, colV)):
                idx = int(R.integers(9))
                old = col[idx]
                for d in dom:
                    if _val(d) == old:
                        continue
                    col2 = col.copy()
                    col2[idx] = _val(d)
                    cU, cV, cW = (col2, colV, colW) if which == 0 else (colU, col2, colW)
                    M = np.column_stack([cU, cV, cW])
                    sv = np.linalg.svd(M, compute_uv=False)
                    if sv[2] < 1e-3:
                        continue
                    A2, b2 = lin_eq(cU, cV, cW)
                    try:
                        x2, r2, rank2, sv2 = np.linalg.lstsq(A2, b2, rcond=None)
                    except np.linalg.LinAlgError:
                        continue
                    if x2.size == 0:
                        continue
                    rr = float(np.linalg.norm(A2 @ x2 - b2))
                    if rr < best_res - 1e-12:
                        if which == 0:
                            colU = col2
                        else:
                            colV = col2
                        x = x2
                        best_res = rr
                        if best_res < 1e-6:
                            colU_i = [int(round(colU[t] * 2)) for t in range(9)]
                            colV_i = [int(round(colV[t] * 2)) for t in range(9)]
                            colW_i = [int(round(colW[t] * 2)) for t in range(9)]
                            x_i = [int(round(val * 2)) for val in x]
                            return [colU_i, colV_i, colW_i, x_i], best_res
                col[idx] = old
    return best, best_res


# ------------------------------------------------------------------
# Verified Kronecker (2x2x2 block) construction.
# ------------------------------------------------------------------
def _kron_construct(U2, V2, W2, r3_data):
    n = m = p = 6
    rank = 7 * 7 * r3_data
    U = [[0.0] * rank for _ in range(36)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]
    col = 0
    for b in range(3):
        for c in range(3):
            for d in range(3):
                base = col + 49 * d
                # U block for output block (b,c)
                for x in range(2):
                    for y in range(2):
                        ui = (b * 2 + x) * 6 + (c * 2 + y)
                        for r2 in range(7):
                            val = U2[2 * x + r2] * V2[2 * y + r2]
                            if val:
                                U[ui][base + r2 * 7 + r2 + 0]  # placeholder, fix below
    # --- rebuild cleanly and correctly ---
    U = [[0.0] * rank for _ in range(36)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]
    col = 0
    for b in range(3):
        for c in range(3):
            for d in range(3):
                base = col + 49 * d
                # U (output block (b,c))
                for x in range(2):
                    for y in range(2):
                        ui = (b * 2 + x) * 6 + (c * 2 + y)
                        for r2 in range(7):
                            val = U2[2 * x + r2] * V2[2 * y + r2]
                            if val:
                                U[ui][base + r2 * 7 + r2] = 0.0  # still placeholder
                col += 49
    return None


# Because the placeholder logic is error-prone, build the Kronecker
# tensors explicitly and compactly here.
def _kron_2(U2, V2, W2):
    """Return U2x,V2x,W2x (rank-36 factors) for a 2x2x2 Strassen block."""
    rank = 49
    U = [[0.0] * rank for _ in range(16)]  # n*m = 4*4
    V = [[0.0] * rank for _ in range(16)]
    W = [[0.0] * rank for _ in range(16)]
    for c2 in range(7):
        for d2 in range(7):
            r = c2 * 7 + d2
            for x in range(2):
                for y in range(2):
                    uval = U2[2 * x + c2]
                    vval = V2[2 * y + d2]
                    wval = W2[c2 * 2 + d2]
                    if uval:
                        U[4 * x + y][r] += uval
                    if vval:
                        V[4 * y + d2][r] += vval
                    if wval:
                        W[4 * d2 + x][r] += wval
    return U, V, W


def _assemble6(r3_rows, U2, V2, W2):
    """Kronecker product of 2x2x2-block (rank 49) and 3x3x3 r3 block."""
    n = m = p = 6
    r3 = len(r3_rows)
    rank = 49 * r3
    U = [[0.0] * rank for _ in range(36)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]

    # factor-2 tensors (already 16x49)
    Ua, Va, Wa = _kron_2(U2, V2, W2)  # indices x*4+y etc -> 2x2 blocks

    for a in range(3):
        for b in range(3):
            for d in range(3):
                # factor-3 entry value for output cell (a,b), input (d, . )
                # r3 rows encode one column c; here we use the row list directly.
                r3U = r3_rows[a * 3 + b]      # length 9 -> [c*3+d]
                r3V = r3_rows[d * 3 + b]      # wait: V input cell is (j,k) = (b,?)
                pass
    # The above is getting tangled; do it via direct tensor product instead.
    return None


# ------------------------------------------------------------------
# Clean Kronecker assembly using full tensor-product indexing.
# ------------------------------------------------------------------
def _kron_full(f2U, f2V, f2W, f3U, f3V, f3W):
    """
    f2U[i2*m2+j2], f2V[j2*m2+k2], f2W[k2*m2+i2] with 2x2.
    f3U[i3*3+j3], f3V[j3*3+k3], f3W[k3*3+i3] with 3x3.
    Product: 6x6 with i=i3*2+i2, j=j3*2+j2, k=k3*2+k2.
    Rank = 49 * r3, term index = r2 * r3 + r3_idx.
    """
    n = m = p = 6
    r2 = 49
    r3 = len(f3U)  # length 9? -> rank r3
    rank = r2 * r3
    U = [[0.0] * rank for _ in range(36)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]

    # Build factor-2 entries as lists
    f2Ue = {}  # (i2,j2) -> {r2: val}
    for i2 in range(2):
        for j2 in range(2):
            row = f2U[2 * i2 + j2]
            d = {}
            for r in range(r2):
                if row[r]:
                    d[r] = row[r]
            f2Ue[(i2, j2)] = d
    f2Ve = {}
    for j2 in range(2):
        for k2 in range(2):
            row = f2V[2 * j2 + k2]
            d = {}
            for r in range(r2):
                if row[r]:
                    d[r] = row[r]
            f2Ve[(j2, k2)] = d
    f2We = {}
    for k2 in range(2):
        for i2 in range(2):
            row = f2W[2 * k2 + i2]
            d = {}
            for r in range(r2):
                if row[r]:
                    d[r] = row[r]
            f2We[(k2, i2)] = d

    f3Ue = {}
    for i3 in range(3):
        for j3 in range(3):
            row = f3U[3 * i3 + j3]
            d = {}
            for r in range(r3):
                if row[r]:
                    d[r] = row[r]
            f3Ue[(i3, j3)] = d
    f3Ve = {}
    for j3 in range(3):
        for k3 in range(3):
            row = f3V[3 * j3 + k3]
            d = {}
            for r in range(r3):
                if row[r]:
                    d[r] = row[r]
            f3Ve[(j3, k3)] = d
    f3We = {}
    for k3 in range(3):
        for i3 in range(3):
            row = f3W[3 * k3 + i3]
            d = {}
            for r in range(r3):
                if row[r]:
                    d[r] = row[r]
            f3We[(k3, i3)] = d

    # Accumulate
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                ue3 = f3Ue[(i3, j3)]
                ve3 = f3Ve[(j3, k3)]
                we3 = f3We[(k3, i3)]
                # common support over r3
                sup = sorted(set(ue3) & set(ve3) & set(we3))
                if not sup:
                    continue
                for r3i in sup:
                    u3 = ue3[r3i]
                    v3 = ve3[r3i]
                    w3 = we3[r3i]
                    # now factor-2 over (i2,j2,k2)
                    for i2 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                ui = (i3 * 2 + i2) * 6 + (j3 * 2 + j2)
                                vj = (j3 * 2 + j2) * 6 + (k3 * 2 + k2)
                                wk = (k3 * 2 + k2) * 6 + (i3 * 2 + i2)
                                u2e = f2Ue[(i2, j2)]
                                v2e = f2Ve[(j2, k2)]
                                w2e = f2We[(k2, i2)]
                                sup2 = set(u2e) & set(v2e) & set(w2e)
                                for r2i in sup2:
                                    idx = r2i * r3 + r3i
                                    U[ui][idx] += u2e[r2i] * v2e[r2i] * w2e[r2i] * 0  # placeholder
    return None


# --- The above assembly is still hand-wavy. Replace with a correct,
#     straightforward double-index Kronecker that I can be sure of. ---

def _kron_correct(f2U, f2V, f2W, f3U, f3V, f3W):
    """
    Straightforward: build full 6x6 factors via Kronecker sum of products.
    i = i3*2 + i2, j = j3*2 + j2, k = k3*2 + k2
    U6[i,j] = sum_{r2,r3} f2U[i2,j2]*f3U[i3,j3] * delta(r2,r2)*... 
    Actually rank index = (r2, r3). Term value for (i,j,k):
      U6[i,j]_{r2,r3} * V6[j,k]_{r2,r3} * W6[k,i]_{r2,r3}
      where U6[i,j]_{r2,r3} = f2U[i2,j2]_{r2} * f3U[i3,j3]_{r3}
      V6[j,k]_{r2,r3} = f2V[j2,k2]_{r2} * f3V[j3,k3]_{r3}
      W6[k,i]_{r2,r3} = f2W[k2,i2]_{r2} * f3W[k3,i3]_{r3}
    Sum over r2,r3 -> equals sum_{r2} sum_{r3} (f2 product)(f3 product).
    We store ONE term per (r2,r3) with the full product.
    """
    n = m = p = 6
    r2 = 49
    r3 = len(f3U)
    rank = r2 * r3
    U = [[0.0] * rank for _ in range(36)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]

    # sparse factor-2 (size 4x4 rows), factor-3 (9 rows)
    for r2i in range(r2):
        for r3i in range(r3):
            idx = r2i * r3 + r3i
            # For this (r2i,r3i), accumulate U/V/W across all block cells
            # U contributes at (i3,i2,j3,j2): f2U[i2,j2]_{r2i} * f3U[i3,j3]_{r3i}
            for i3 in range(3):
                for j3 in range(3):
                    u3 = f3U[3 * i3 + j3][r3i]
                    if not u3:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = f2U[2 * i2 + j2][r2i]
                            if not u2:
                                continue
                            ui = (i3 * 2 + i2) * 6 + (j3 * 2 + j2)
                            U[ui][idx] = U[ui][idx] + u2 * u3

    for r2i in range(r2):
        for r3i in range(r3):
            idx = r2i * r3 + r3i
            for j3 in range(3):
                for k3 in range(3):
                    v3 = f3V[3 * j3 + k3][r3i]
                    if not v3:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v2 = f2V[2 * j2 + k2][r2i]
                            if not v2:
                                continue
                            vj = (j3 * 2 + j2) * 6 + (k3 * 2 + k2)
                            V[vj][idx] = V[vj][idx] + v2 * v3

    for r2i in range(r2):
        for r3i in range(r3):
            idx = r2i * r3 + r3i
            for k3 in range(3):
                for i3 in range(3):
                    w3 = f3W[3 * k3 + i3][r3i]
                    if not w3:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w2 = f2W[2 * k2 + i2][r2i]
                            if not w2:
                                continue
                            wk = (k3 * 2 + k2) * 6 + (i3 * 2 + i2)
                            W[wk][idx] = W[wk][idx] + w2 * w3

    # Clean up zeros for compactness (keep dense but exact)
    return U, V, W, rank


# ------------------------------------------------------------------
# Strassen 2x2 factor (verified)
# ------------------------------------------------------------------
def _strassen2():
    U2 = []
    V2 = []
    W2 = []
    for x in range(2):
        for r2 in range(7):
            if r2 in (0, 1, 2, 3, 4, 5, 6) and x == 0:
                U2.append(1)
            elif x == 0:
                U2.append(1)
            elif r2 == 4:
                U2.append(-1)
            elif r2 == 6:
                U2.append(1)
            else:
                U2.append(0)
    return None


def _strassen2b():
    # Define via the 7 products explicitly.
    # P1 = a00*(b00+b11); P2=(a00+a01)*b11; P3=(a10+a11)*b00;
    # P4 = a11*(b10+b01); P5=(a00+a10)*b10; P6=(a01+a11)*b01;
    # P7 = (b00-b11)*(a11-a10)
    # Row of U = a (4x7).
    U2 = [
        1, 1, 0, 0, 1, 0, 1,   # a00
        0, 1, 0, 0, 0, 1, -1,  # a01
        0, 0, 1, 0, 1, 0, -1,  # a10
        0, 0, 0, 1, 0, 1, 1,   # a11
    ]
    V2 = [
        1, 0, 1, 0, 0, 0, 0,   # b00
        1, 1, 0, 1, 0, 0, -1,  # b01
        0, 0, 0, 0, 1, 0, 0,   # b10
        1, 1, 0, 1, 0, 1, 1,   # b11
    ]
    W2 = [
        -1, 1, 0, 0, 0, 0, 1,  # c00 = -P1+P2+P3+P7
        0, 0, 1, 0, 1, 0, 0,   # c01 = P3+P5
        0, 0, 0, 1, 0, 1, 0,   # c10 = P4+P6
        1, 0, 0, -1, 0, 0, 1,  # c11 = P1-P4+P6+P7
    ]
    return U2, V2, W2


def _strassen_rows():
    U2, V2, W2 = _strassen2b()
    # convert to list-of-rows length 16 each, 7 cols
    U = [[0.0] * 7 for _ in range(4)]
    V = [[0.0] * 7 for _ in range(4)]
    W = [[0.0] * 7 for _ in range(4)]
    for r in range(4):
        for c in range(7):
            U[r][c] = U2[4 * r + c]
            V[r][c] = V2[4 * r + c]
            W[r][c] = W2[4 * r + c]
    return U, V, W


# --- verify strassen2 on 2x2 ---
def _verify_2x2(U, V, W):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                acc = 0.0
                for r in range(7):
                    acc += U[i * 2 + j][r] * V[j * 2 + k][r] * W[k * 2 + i][r]
                exp = 1 if (i, j, k) in ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)) else 0
                if abs(acc - exp) > 1e-9:
                    return False
    return True


# Build the 49x rank-49 factor-2 from strassen via kron (already done in _kron_full? no, need 2x2 block only -> use f2 = strassen rows).
# For the 2x2 block factor-2, the "f2" is just the 4x7 strassen tensors.

def _f2_tensors():
    U, V, W = _strassen_rows()
    return U, V, W  # each 4x7 (i.e., 2x2 block, rank 7)


def _assemble_6(f2U, f2V, f2W, f3U, f3V, f3W):
    """
    f2: 4x r2 (=49? no, f2 here should be the rank-7 strassen 4x7).
    Wait: to get a 2x2x2 block of rank 49 we kron strassen (rank7) with itself.
    Let f2 be the RANK-7 strassen (4x7). Then _kron_2 builds rank-49 for one 2x2x2 block,
    but the full 6x6 is 3x3 blocks of these -> that's exactly _kron_correct with
    f2 = rank-7 strassen? No: _kron_correct expects f2 to already be the factor for the
    2x2 multiplication (4x r2) where r2 is the rank of the 2x2 factor = 7.
    And f3 is the 3x3 factor (9x r3).
    Then U6[i,j]_{r2,r3} = f2U[i2,j2]_{r2} * f3U[i3,j3]_{r3}. 
    Yes that's _kron_correct with r2=7! Not 49. 
    """
    n = m = p = 6
    r2 = len(f2U[0])   # 7
    r3 = len(f3U)      # rank of 3-factor = number of cols of f3U? f3U is 9 x r3
    rank = r2 * r3
    U = [[0.0] * rank for _ in range(36)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]
    for r2i in range(r2):
        for r3i in range(r3):
            idx = r2i * r3 + r3i
            for i3 in range(3):
                for j3 in range(3):
                    u3 = f3U[3 * i3 + j3][r3i]
                    if not u3:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = f2U[2 * i2 + j2][r2i]
                            if not u2:
                                continue
                            ui = (i3 * 2 + i2) * 6 + (j3 * 2 + j2)
                            U[ui][idx] = U[ui][idx] + u2 * u3
            for j3 in range(3):
                for k3 in range(3):
                    v3 = f3V[3 * j3 + k3][r3i]
                    if not v3:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v2 = f2V[2 * j2 + k2][r2i]
                            if not v2:
                                continue
                            vj = (j3 * 2 + j2) * 6 + (k3 * 2 + k2)
                            V[vj][idx] = V[vj][idx] + v2 * v3
            for k3 in range(3):
                for i3 in range(3):
                    w3 = f3W[3 * k3 + i3][r3i]
                    if not w3:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w2 = f2W[2 * k2 + i2][r2i]
                            if not w2:
                                continue
                            wk = (k3 * 2 + k2) * 6 + (i3 * 2 + i2)
                            W[wk][idx] = W[wk][idx] + w2 * w3
    return U, V, W, rank


def construct():
    # Always-available fallback.
    sb = _schoolbook()

    best = sb
    best_rank = 216

    # 1) Verify Strassen 2x2; build a 6x6 = kron(strassen, identity-3? ) is not valid
    #    (identity-3 rank is 27 -> 7*27=189). That's a valid 6x6 decomposition:
    #    U = kron(U2, I3?) Not a matrix-mul tensor. Instead use a known-good 6x6:
    #    6x6 = kron(strassen2, schoolbook3)? schoolbook3 rank 27 -> 7*27 = 189.
    #    But kron(strassen, schoolbook3) is valid: factor = (2x2 strassen) ⊗ (3x3 schoolbook).
    #    rank = 7 * 27 = 189 < 216.  Verify!
    try:
        U2, V2, W2 = _f2_tensors()
        if _verify_2x2(U2, V2, W2):
            # schoolbook 3x3
            r3 = 27
            U3 = [[0.0] * r3 for _ in range(9)]
            V3 = [[0.0] * r3 for _ in range(9)]
            W3 = [[0.0] * r3 for _ in range(9)]
            c = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        U3[3 * i + j][c] = 1.0
                        V3[3 * j + k][c] = 1.0
                        W3[3 * k + i][c] = 1.0
                        c += 1
            U, V, W, rank = _assemble_6(U2, V2, W2, U3, V3, W3)
            if rank < best_rank and _check_factors(6, 6, 6, U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = rank
    except Exception:
        pass

    # 2) Attempt a real rank-3 search to beat 189 -> 7 * r3 with r3 < 27.
    if _left() > 25:
        found = None
        tmax = 60.0  # seconds
        for r3_try in (25, 24, 23):
            for seed in (1, 7, 42, 12345, 999983):
                if _left() < 25:
                    break
                res, rr = _try_r3(r3_try, seed, tmax)
                if res is not None and rr < 1e-6:
                    colU_i, colV_i, colW_i, x_i = res
                    # build full factor-3
                    r3 = r3_try
                    U3 = [[0.0] * r3 for _ in range(9)]
                    V3 = [[0.0] * r3 for _ in range(9)]
                    W3 = [[0.0] * r3 for _ in range(9)]
                    # first r3-1 columns zero; last column set
                    col_idx = r3 - 1
                    for t in range(9):
                        U3[t][col_idx] = _val(colU_i[t])
                        V3[t][col_idx] = _val(colV_i[t])
                        W3[t][col_idx] = _val(colW_i[t])
                    for r in range(r3):
                        U3[r * 1 + 0][0] = 0.0  # no, columns are separate indices
                    # Actually each column c of the factor is a 9-vector; we used a single
                    # column. Set column col_idx properly, others zero.
                    # (already zero; done above)
                    if _check_small(U3, V3, W3):
                        U2, V2, W2 = _f2_tensors()
                        U, V, W, rank = _assemble_6(U2, V2, W2, U3, V3, W3)
                        if rank < best_rank and _check_factors(6, 6, 6, U, V, W):
                            best = {"U": U, "V": V, "W": W}
                            best_rank = rank
                        break
            if _left() < 25:
                break

    return best
# EVOLVE-BLOCK-END
