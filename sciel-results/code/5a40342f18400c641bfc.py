import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    # P1 = (a+d)(e+h)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # P2 = (c+d)e
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=-1
    # P3 = a(f-h)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1; W[3][2]=1
    # P4 = d(g-e)
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[1][3]=1
    # P5 = (a+b)h
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    # P6 = (c-a)(e+f)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    # P7 = (b-d)(g+h)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col]=1; V[j*p+k][col]=1; W[k*n+i][col]=1
                col += 1
    return U, V, W

def _build_disjoint_strassen(n, m, p, blocks):
    """Embed disjoint 2x2x2 Strassen blocks; schoolbook elsewhere (exhaustive)."""
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    n_school = n*m*p - len(covered)
    rank = 7*len(blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for (i0, j0, k0) in blocks:
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
                    U[i*m+j][col]=1; V[j*p+k][col]=1; W[k*n+i][col]=1
                    col += 1
    return U, V, W

def _all_disjoint_block_sets(n, m, p):
    """Yield sets of disjoint 2x2x2 blocks (i0,j0,k0)."""
    i0s = list(range(n-1))
    j0s = list(range(m-1))
    k0s = list(range(p-1))
    # greedy: pick maximal disjoint sets via simple enumeration of k-groups
    # For 3x3x8: i0=0, j0=0, k0 in {0,2,4,6} gives 4 disjoint blocks.
    # General: try combinations of k0 with step>=2.
    import itertools
    results = []
    for k0_combo in itertools.combinations(k0s, 4):
        if all(abs(a-b)>=2 for a in k0_combo) and all(abs(a-b)>=2 for a in k0_combo for b in k0_combo):
            blocks = [(0,0,k) for k in sorted(k0_combo)]
            results.append(blocks)
    # Also single block and two-block variants for safety
    results.append([(0,0,0)])
    results.append([(0,0,0),(0,0,2)])
    results.append([(0,0,0),(0,0,2),(0,0,4),(0,0,6)])
    # dedupe
    seen = set()
    out = []
    for b in results:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out

def construct():
    n, m, p = DIMENSIONS
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n*m*p

    for blocks in _all_disjoint_block_sets(n, m, p):
        if _time_left() < 3:
            break
        try:
            U, V, W = _build_disjoint_strassen(n, m, p, blocks)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass

    return best_result
