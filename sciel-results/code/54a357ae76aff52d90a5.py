import time
import itertools

DIMENSIONS = [3, 3, 8]
DEADLINE = 140.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _strassen_2x2_2():
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    return U, V, W

def _try_strassen_placements(n, m, p):
    best_rank = n * m * p
    best_result = None
    all_positions = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_positions.append((i0, j0, k0))
    for pos in all_positions:
        if _time_left() < 5:
            break
        U, V, W = _build_strassen_embedding(n, m, p, [pos])
        if _verify(U, V, W, n, m, p):
            r = len(U[0])
            if r < best_rank:
                best_rank = r
                best_result = (U, V, W)
    for idx1 in range(len(all_positions)):
        if _time_left() < 10:
            return best_result
        pos1 = all_positions[idx1]
        covered1 = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered1.add((pos1[0]+di, pos1[1]+dj, pos1[2]+dk))
        for idx2 in range(idx1 + 1, len(all_positions)):
            if _time_left() < 10:
                return best_result
            pos2 = all_positions[idx2]
            overlap = False
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if (pos2[0]+di, pos2[1]+dj, pos2[2]+dk) in covered1:
                            overlap = True
                            break
                    if overlap:
                        break
                if overlap:
                    break
            if overlap:
                continue
            U, V, W = _build_strassen_embedding(n, m, p, [pos1, pos2])
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = (U, V, W)
    return best_result

def _try_trilinear_block(n, m, p):
    if p % 2 != 0:
        return None
    half = p // 2
    def _block_decomp(n, m, phalf):
        positions = []
        for i0 in range(n - 1):
            for j0 in range(m - 1):
                for k0 in range(phalf - 1):
                    positions.append((i0, j0, k0))
        chosen = []
        covered = set()
        for pos in positions:
            if _time_left() < 3:
                break
            ok = True
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if (pos[0]+di, pos[1]+dj, pos[2]+dk) in covered:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                chosen.append(pos)
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered.add((pos[0]+di, pos[1]+dj, pos[2]+dk))
        if not chosen:
            return _schoolbook(n, m, phalf)
        return _build_strassen_embedding(n, m, phalf, chosen)
    U1, V1, W1 = _block_decomp(n, m, half)
    U2, V2, W2 = _block_decomp(n, m, half)
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 + r2
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for a in range(n*m):
        for c in range(r1):
            U[a][c] = U1[a][c]
        for c in range(r2):
            U[a][r1+c] = U2[a][c]
    for a in range(m*p):
        for c in range(r1):
            V[a][c] = V1[a][c]
        for c in range(r2):
            V[a][r1+c] = V2[a][c]
    for a in range(p*n):
        for c in range(r1):
            W[a][c] = W1[a][c]
        for c in range(r2):
            W[a][r1+c] = W2[a][c]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p

    # 1) Fixed known-good Strassen placement (4 blocks along k).
    if _time_left() > 5:
        try:
            blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
            U, V, W = _build_strassen_embedding(n, m, p, blocks)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass

    # 2) Rectangular block decomposition.
    if _time_left() > 10:
        try:
            cand = _try_trilinear_block(n, m, p)
            if cand is not None:
                U, V, W = cand
                if _verify(U, V, W, n, m, p):
                    r = len(U[0])
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass

    # 3) Systematic Strassen placement search.
    if _time_left() > 15:
        try:
            found = _try_strassen_placements(n, m, p)
            if found is not None:
                U, V, W = found
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass

    return best_result
