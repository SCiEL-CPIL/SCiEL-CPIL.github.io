"""Program for 4x4x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    # 7 rank-1 terms for 2x2x2 matrix multiplication (1x1 blocks).
    terms = []
    terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
    terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}))
    terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1}))
    terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(1,0):1}))
    terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,1):1}))
    terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
    terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
    return terms

def _build_4x4x4_strassen():
    outer = _strassen_2x2_terms()
    inner = _strassen_2x2_terms()
    terms = []
    for oA, oB, oC in outer:
        for iA, iB, iC in inner:
            A = {}
            B = {}
            C = {}
            for (bI, bJ), oa in oA.items():
                for (li, lj), ia in iA.items():
                    gi, gj = bI*2+li, bJ*2+lj
                    A[(gi,gj)] = A.get((gi,gj),0) + oa*ia
            for (bJ, bK), ob in oB.items():
                for (lj, lk), ib in iB.items():
                    gj, gk = bJ*2+lj, bK*2+lk
                    B[(gj,gk)] = B.get((gj,gk),0) + ob*ib
            for (bK, bI), oc in oC.items():
                for (lk, li), ic in iC.items():
                    gk, gi = bK*2+lk, bI*2+li
                    C[(gk,gi)] = C.get((gk,gi),0) + oc*ic
            terms.append((A,B,C))
    return terms

def _decompose_4x4x4(terms):
    n=m=p=4
    R = len(terms)
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    for c,(A,B,C) in enumerate(terms):
        for (i,j),val in A.items():
            U[i*m+j][c] += val
        for (j,k),val in B.items():
            V[j*p+k][c] += val
        for (k,i),val in C.items():
            W[k*n+i][c] += val
    return U,V,W

def _schoolbook(n,m,p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col=0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col]=1
                V[j*p+k][col]=1
                W[k*n+i][col]=1
                col+=1
    return U,V,W

def construct():
    n,m,p = DIMENSIONS
    try:
        terms = _build_4x4x4_strassen()
        U4,V4,W4 = _decompose_4x4x4(terms)
        R1 = len(terms)
        R2 = 16
        rank = R1 + R2
        U = [row[:] for row in U4]
        V = [row[:] for row in V4]
        W = [row[:] for row in W4]
        for r in range(rank - R1):
            for row in U: row.append(0)
            for row in V: row.append(0)
            for row in W: row.append(0)
        t=0
        for i in range(n):
            for j in range(m):
                U[i*m+j][R1+t]=1
                V[j*p+4][R1+t]=1
                W[4*n+i][R1+t]=1
                t+=1
        return {"U":U,"V":V,"W":W}
    except Exception:
        U,V,W = _schoolbook(n,m,p)
        return {"U":U,"V":V,"W":W}
# EVOLVE-BLOCK-END
