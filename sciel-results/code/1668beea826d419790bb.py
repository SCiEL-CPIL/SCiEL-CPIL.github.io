"""Improved: Strassen on 2x2 grid of 3x3 blocks; 3x3 via rank-23 (Lader) if valid, else schoolbook."""
DIMENSIONS = [6, 6, 6]

import time

# EVOLVE-BLOCK-START
def _build_3x3_rank23():
    """Lader's rank-23 decomposition of 3x3 matrix multiplication.
    Term (i,j,k): A[i,j]*B[j,k] -> C[i,k].
    U3[i*3+j, c]=1, V3[j*3+k, c]=1, W3[k*3+i, c]=1."""
    terms = [
        (0,0,0),(0,0,1),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2),
        (1,0,0),(1,0,1),(1,1,1),(1,1,2),(1,2,0),(1,2,1),(1,2,2),
        (2,0,0),(2,0,1),(2,1,0),(2,1,1),(2,1,2),(2,2,1),(2,2,2),
    ]
    n = m = p = 3
    rank = len(terms)
    U3 = [[0]*rank for _ in range(9)]
    V3 = [[0]*rank for _ in range(9)]
    W3 = [[0]*rank for _ in range(9)]
    for c, (i, j, k) in enumerate(terms):
        U3[i*3+j][c] = 1
        V3[j*3+k][c] = 1
        W3[k*3+i][c] = 1
    return U3, V3, W3

def _build_3x3_schoolbook():
    n = m = p = 3
    rank = 27
    U3 = [[0]*rank for _ in range(9)]
    V3 = [[0]*rank for _ in range(9)]
    W3 = [[0]*rank for _ in range(9)]
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][c] = 1
                V3[j*3+k][c] = 1
                W3[k*3+i][c] = 1
                c += 1
    return U3, V3, W3

def _verify_3x3(U3, V3, W3, trials=60):
    import random
    n = m = p = 3
    rank = len(U3[0])
    for _ in range(trials):
        A = [[random.randint(-4,4) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-4,4) for _ in range(p)] for _ in range(m)]
        C = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j]*B[j][k]
                C[i][k] = s
        Cr = [[0]*p for _ in range(n)]
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    u = U3[i*m+j][c]
                    if u == 0: continue
                    for k in range(p):
                        v = V3[j*p+k][c]
                        if v == 0: continue
                        w = W3[k*n+i][c]
                        if w == 0: continue
                        Cr[i][k] += u*v*w
        for i in range(n):
            for k in range(p):
                if Cr[i][k] != C[i][k]:
                    return False
    return True

def _strassen_2x2_grid_of_3x3(U3, V3, W3):
    n = m = p = 6
    r3 = len(U3[0])
    rank = 7 * r3
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    # Strassen products M1..M7: ca (A blocks), cb (B blocks), cc (C blocks)
    M = [
        {"ca": {(0,0):1,(1,1):1}, "cb": {(0,0):1,(1,1):1}, "cc": {(0,0):1,(1,1):1}},
        {"ca": {(1,0):1,(1,1):1}, "cb": {(0,0):1},          "cc": {(1,0):1,(1,1):-1}},
        {"ca": {(0,0):1},          "cb": {(0,1):1,(1,1):-1}, "cc": {(0,1):1,(1,1):1}},
        {"ca": {(1,1):1},          "cb": {(1,0):1,(0,0):-1}, "cc": {(0,0):1,(1,0):1}},
        {"ca": {(0,0):1,(0,1):1},  "cb": {(1,1):1},          "cc": {(0,0):-1,(0,1):1}},
        {"ca": {(1,0):1,(0,0):-1}, "cb": {(0,0):1,(0,1):1},  "cc": {(1,1):1}},
        {"ca": {(0,1):1,(1,1):-1}, "cb": {(1,0):1,(1,1):1},  "cc": {(0,0):1}},
    ]
    for x, Mx in enumerate(M):
        ca = Mx["ca"]; cb = Mx["cb"]; cc = Mx["cc"]
        for t in range(r3):
            col = x * r3 + t
            for ii in range(3):
                for jj in range(3):
                    u3 = U3[ii*3+jj, t]
                    if u3 == 0: continue
                    for (bi,bj), ccoef in ca.items():
                        if ccoef == 0: continue
                        gi = bi*3 + ii
                        gj = bj*3 + jj
                        U[gi*6 + gj][col] += u3 * ccoef
            for jj in range(3):
                for jk in range(3):
                    v3 = V3[jj*3+jk, t]
                    if v3 == 0: continue
                    for (bj,bk), ccoef in cb.items():
                        if ccoef == 0: continue
                        gj = bj*3 + jj
                        gk = bk*3 + jk
                        V[gj*6 + gk][col] += v3 * ccoef
            for jk in range(3):
                for ii in range(3):
                    w3 = W3[jk*3+ii, t]
                    if w3 == 0: continue
                    for (bi,bk), ccoef in cc.items():
                        if ccoef == 0: continue
                        gi = bi*3 + ii
                        gk = bk*3 + jk
                        W[gk*6 + gi][col] += w3 * ccoef
    return U, V, W

def _build_6x6_schoolbook():
    n = m = p = 6
    rank = 216
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    c = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U[i*6+j][c] = 1
                V[j*6+k][c] = 1
                W[k*6+i][c] = 1
                c += 1
    return U, V, W

def _verify_full(U, V, W, trials=20):
    import random
    n = m = p = 6
    rank = len(U[0])
    for _ in range(trials):
        A = [[random.randint(-2,2) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-2,2) for _ in range(p)] for _ in range(m)]
        C = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j]*B[j][k]
                C[i][k] = s
        Cr = [[0]*p for _ in range(n)]
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    u = U[i*m+j][c]
                    if u == 0: continue
                    for k in range(p):
                        v = V[j*p+k][c]
                        if v == 0: continue
                        w = W[k*n+i][c]
                        if w == 0: continue
                        Cr[i][k] += u*v*w
        for i in range(n):
            for k in range(p):
                if Cr[i][k] != C[i][k]:
                    return False
    return True

def construct():
    deadline = time.time() + 100
    # Try rank-23 3x3 first
    U3, V3, W3 = _build_3x3_rank23()
    if _verify_3x3(U3, V3, W3, trials=50):
        U, V, W = _strassen_2x2_grid_of_3x3(U3, V3, W3)
    else:
        # Fallback to schoolbook 3x3 (rank 27) -> total 189
        U3, V3, W3 = _build_3x3_schoolbook()
        U, V, W = _strassen_2x2_grid_of_3x3(U3, V3, W3)
    if time.time() < deadline and not _verify_full(U, V, W, trials=15):
        U, V, W = _build_6x6_schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
