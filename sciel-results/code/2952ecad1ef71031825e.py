"""Matrix multiplication 3,3,5 rank-minimal tensor decomposition."""
import time

DIMENSIONS = [3, 3, 5]
_START = time.time()
DEADLINE = 150.0
_SAFETY = 5.0  # seconds margin before kill

def _time_left():
    return DEADLINE - _SAFETY - (time.time() - _START)

def _target_tensor(n, m, p):
    """T[i*m+j, j*p+k, k*n+i] = 1 else 0."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i * m + j, j * p + k, k * n + i)] = 1
    return T

def _verify(U, V, W, T):
    """Exact integer verification that sum_r U[:,r]⊗V[:,r]⊗W[:,r] == T."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # build candidate tensor dict
    cand = {}
    for r in range(rank):
        uc = U[:, r]
        vc = V[:, r]
        wc = W[:, r]
        for a in range(n * m):
            uav = uc[a]
            if uav == 0:
                continue
            for b in range(m * p):
                ubv = uav * vc[b]
                if ubv == 0:
                    continue
                for c in range(p * n):
                    wcv = ubv * wc[c]
                    if wcv == 0:
                        continue
                    cand[(a, b, c)] = cand.get((a, b, c), 0) + wcv
    # compare
    if set(cand.keys()) != set(T.keys()):
        return False
    for key, val in T.items():
        if cand[key] != val:
            return False
    return True

def _schoolbook(n, m, p):
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

def _strassen_222():
    """Rank-7 Strassen factors for 2x2x2, indexed i*2+j, j*2+k, k*2+i."""
    # M1..M7
    M = {
        1: [(0, 0), (1, 1)],          # A00*(B11-B01)
        2: [(0, 1)],                   # A01*B00
        3: [(1, 0), (1, 1)],           # A10*(B01+B11)
        4: [(0, 0), (1, 0)],           # A11*(B10-B00)
        5: [(0, 0), (0, 1)],           # A00*(B00+B11)
        6: [(1, 0), (1, 1)],           # (A10-A00)*B01
        7: [(0, 1), (1, 1)],           # (A01-A11)*B10
    }
    signs = {
        1: {1: 1, 0: -1},   # for M1: B11 - B01
        2: {0: 1},
        3: {1: 1, 3: 1},    # B01+B11 (indices j*2+k: (1,0)->1, (1,1)->3)
        4: {2: 1, 0: -1},   # B10 - B00
        5: {0: 1, 3: 1},    # B00+B11
        6: {1: 1},          # B01
        7: {2: 1},          # B10
    }
    # U factors (A side)
    Ufac = {
        1: [(0, 1)],
        2: [(1, 1)],
        3: [(2, 1)],
        4: [(3, 1)],
        5: [(0, 1)],
        6: [(2, 1), (0, -1)],
        7: [(1, 1), (3, -1)],
    }
    # W factors (C side): C = M * (B combos)
    # C00 = M5 - M1 - M4 + M7
    # C01 = M2 + M3
    # C10 = M6 + M1
    # C11 = M5 - M2 + M3 - M7
    Wfac = {
        0: {5: 1, 1: -1, 4: -1, 7: 1},   # C00 (k*2+i = 0)
        1: {2: 1, 3: 1},                   # C01 (k*2+i = 1)
        2: {6: 1, 1: 1},                   # C10 (k*2+i = 2)
        3: {5: 1, 2: -1, 3: 1, 7: -1},     # C11 (k*2+i = 3)
    }
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    for r in range(1, 8):
        for (idx, val) in Ufac[r]:
            U[idx][r - 1] += val
        for (bidx, val) in signs[r].items():
            V[bidx][r - 1] += val
        for (cidx, val) in Wfac[cidx if False else 0].items():
            pass
    # fill W
    for cidx in range(4):
        for (r, val) in Wfac[cidx].items():
            W[cidx][r - 1] += val
    return U, V, W

def _block_candidate(n, m, p):
    """3x3x3 via Strassen(2x2x2)+schoolbook L-shape, plus 3x3x2 schoolbook tail."""
    # We'll just build full schoolbook then try to replace the 2x2x2 sub-block (i,j,k in {0,1}) with Strassen.
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    # Start with schoolbook
    sb = _schoolbook(n, m, p)
    U = [row[:] for row in sb["U"]]
    V = [row[:] for row in sb["V"]]
    W = [row[:] for row in sb["W"]]
    # Identify schoolbook columns for the 2x2x2 sub-block (i,j,k in {0,1})
    sub_cols = []
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    sub_cols.append(col)
                col += 1
    # Remove those columns (set to zero) and add Strassen columns
    for c in sub_cols:
        for row in U:
            row[c] = 0
        for row in V:
            row[c] = 0
        for row in W:
            row[c] = 0
    # Strassen factors for 2x2x2
    sU, sV, sW = _strassen_222()
    base = sub_cols[0]  # reuse first removed column index as base; we'll add 7 new columns at end
    # Actually simpler: append 7 columns
    rank0 = n * m * p
    for r in range(7):
        U.append([0] * rank0 + [0] * 7)
        V.append([0] * rank0 + [0] * 7)
        W.append([0] * rank0 + [0] * 7)
    # extend existing rows
    for row in U:
        row += [0] * 7
    for row in V:
        row += [0] * 7
    for row in W:
        row += [0] * 7
    for r in range(7):
        for a in range(4):
            U[a][rank0 + r] += sU[a][r]
        for b in range(4):
            V[b][rank0 + r] += sV[b][r]
        for c in range(4):
            W[c][rank0 + r] += sW[c][r]
    # Trim trailing zero columns
    new_rank = 0
    for c in range(rank0 + 7):
        if any(U[a][c] for a in range(len(U))) or any(V[b][c] for b in range(len(V))) or any(W[c2][c] for c2 in range(len(W))):
            new_rank += 1
    U = [row[:new_rank] for row in U]
    V = [row[:new_rank] for row in V]
    W = [row[:new_rank] for row in W]
    # remove fully-zero rows? keep shapes
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    T = _target_tensor(n, m, p)
    candidates = []
    # schoolbook
    candidates.append(_schoolbook(n, m, p))
    # block candidate
    try:
        if _time_left() > 2.0:
            candidates.append(_block_candidate(n, m, p))
    except Exception:
        pass
    best = None
    best_rank = None
    for cand in candidates:
        if _time_left() <= 0.0:
            break
        r = len(cand["U"][0])
        if best_rank is not None and r >= best_rank:
            continue
        try:
            if _verify(cand["U"], cand["V"], cand["W"], T):
                best = cand
                best_rank = r
        except Exception:
            continue
    if best is None:
        # fallback schoolbook (always valid)
        best = _schoolbook(n, m, p)
    return best
