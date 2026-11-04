DIMENSIONS = [4, 4, 5]

def _strassen_terms():
    return [
        ([1,0,0,1],[1,0,0,1],[1,0,0,1]),
        ([0,0,1,1],[1,0,0,0],[0,0,1,-1]),
        ([1,0,0,0],[0,1,0,-1],[0,1,0,1]),
        ([0,0,0,1],[0,0,-1,1],[1,0,1,0]),
        ([1,1,0,0],[0,0,0,1],[-1,1,0,0]),
        ([0,-1,1,0],[1,1,0,0],[0,0,0,1]),
        ([0,1,-1,0],[0,0,1,1],[1,0,0,0]),
    ]

def construct():
    import time
    t0 = time.time()
    DEADLINE = 120.0
    n, m, p = DIMENSIONS
    rank = 72
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    def add(u4, v4, w4, ar, ac, br, bc, cr, cc):
        nonlocal col
        for lr in range(2):
            for lc in range(2):
                li = lr*2+lc
                if u4[li]:
                    U[(ar+lr)*m + (ac+lc)][col] += u4[li]
                if v4[li]:
                    V[(br+lr)*m + (bc+lc)][col] += v4[li]
                if w4[li]:
                    W[(cc+lc)*n + (cr+lr)][col] += w4[li]
        col += 1
    terms = _strassen_terms()
    # 8 block products for 4x4x4
    blocks = [
        (0,0, 0,0, 0,0), (0,2, 2,0, 0,0),   # C11
        (0,0, 0,2, 0,2), (0,2, 2,2, 0,2),   # C12
        (2,0, 0,0, 2,0), (2,2, 2,0, 2,0),   # C21
        (2,0, 0,2, 2,2), (2,2, 2,2, 2,2),   # C22
    ]
    for (ar,ac,br,bc,cr,cc) in blocks:
        for (u4,v4,w4) in terms:
            if time.time() > DEADLINE - 5:
                break
            add(u4,v4,w4,ar,ac,br,bc,cr,cc)
    # 4x4x1 schoolbook
    for i in range(n):
        for j in range(m):
            U[i*m + j][col] = 1
            V[j*p + 0][col] = 1
            W[0*n + i][col] = 1
            col += 1
    return {"U": U, "V": V, "W": W}
