import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    U=[[0]*7 for _ in range(4)]; V=[[0]*7 for _ in range(4)]; W=[[0]*7 for _ in range(4)]
    defs = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,-1),(1,0,-1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,-1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    for t,(ue,ve,we) in enumerate(defs):
        for (i,j,c) in ue: U[i*2+j][t]+=c
        for (j,k,c) in ve: V[j*2+k][t]+=c
        for (k,i,c) in we: W[k*2+i][t]+=c
    return U,V,W

def _build_4x4x4():
    U=[[0]*49 for _ in range(16)]; V=[[0]*49 for _ in range(16)]; W=[[0]*49 for _ in range(16)]
    U2,V2,W2 = _strassen_2x2()
    outer = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,-1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    for s,(ue,ve,we) in enumerate(outer):
        for t in range(7):
            col = s*7+t
            for (bi,bj,c) in ue:
                for i in range(2):
                    for j in range(2):
                        gi=bi*2+i; gj=bj*2+j
                        cc=U2[i*2+j][t]
                        if cc: U[gi*4+gj][col]+=cc*c
            for (bj,bk,c) in ve:
                for j in range(2):
                    for k in range(2):
                        gj=bj*2+j; gk=bk*2+k
                        cc=V2[j*2+k][t]
                        if cc: V[gj*4+gk][col]+=cc*c
            for (bk,bi,c) in we:
                for k in range(2):
                    for i in range(2):
                        gk=bk*2+k; gi=bi*2+i
                        cc=W2[k*2+i][t]
                        if cc: W[gk*4+gi][col]+=cc*c
    return U,V,W

def _build_4x4x2():
    U=[[0]*28 for _ in range(16)]; V=[[0]*28 for _ in range(8)]; W=[[0]*28 for _ in range(8)]
    U2,V2,W2 = _strassen_2x2()
    blocks = [(0,0,0,0,0),(0,2,2,0,0),(2,0,0,2,0),(2,2,2,2,0)]
    for bi,(ar,ac,br,cr,ck) in enumerate(blocks):
        base = bi*7
        for t in range(7):
            col=base+t
            for i in range(2):
                for j in range(2):
                    cc=U2[i*2+j][t]
                    if cc: U[(ar+i)*4+(ac+j)][col]+=cc
            for j in range(2):
                for k in range(2):
                    cc=V2[j*2+k][t]
                    if cc: V[(br+j)*2+(ck+k)][col]+=cc
            for k in range(2):
                for i in range(2):
                    cc=W2[k*2+i][t]
                    if cc: W[(ck+k)*4+(cr+i)][col]+=cc
    return U,V,W

def _build_1x1x2():
    U=[[0]*2 for _ in range(1)]; V=[[0]*2 for _ in range(2)]; W=[[0]*2 for _ in range(2)]
    for k in range(2):
        U[0][k]=1; V[k][k]=1; W[k][k]=1
    return U,V,W

def _build_1x1x4():
    U=[[0]*4 for _ in range(1)]; V=[[0]*4 for _ in range(4)]; W=[[0]*4 for _ in range(4)]
    for k in range(4):
        U[0][k]=1; V[k][k]=1; W[k][k]=1
    return U,V,W

def _combine(parts, n, m, p):
    total = sum(len(d[0][0]) for d in parts)
    U=[[0]*total for _ in range(n*m)]; V=[[0]*total for _ in range(m*p)]; W=[[0]*total for _ in range(p*n)]
    off=0
    for (Us,Vs,Ws, (i0,j0,ii,jj,k0,kk)) in parts:
        nA=ii; mA=jj; nB=kk
        for r in range(nA*mA):
            i=r//mA; j=r%mA
            for c in range(len(Us[0])):
                U[(i0+i)*m+(j0+j)][off+c]=Us[r][c]
        for r in range(mA*nB):
            j=r//nB; k=r%nB
            for c in range(len(Vs[0])):
                V[(j0+j)*p+(k0+k)][off+c]=Vs[r][c]
        for r in range(nB*nA):
            k=r//nA; i=r%nA
            for c in range(len(Ws[0])):
                W[(k0+k)*n+(i0+i)][off+c]=Ws[r][c]
        off+=len(Us[0])
    return U,V,W

def _verify(U,V,W,n,m,p, trials=10, seed=12345):
    rng=random.Random(seed)
    rank=len(U[0])
    for _ in range(trials):
        A=[[rng.randint(-5,5) for _ in range(m)] for _ in range(n)]
        B=[[rng.randint(-5,5) for _ in range(p)] for _ in range(m)]
        C=[[sum(A[i][j]*B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                val=0
                for t in range(rank):
                    au=sum(U[i*m+j][t]*A[i][j] for j in range(m))
                    bu=sum(V[j*p+k][t]*B[j][k] for j in range(m))
                    val+=W[k*n+i][t]*au*bu
                if val!=C[i][k]:
                    return False
    return True

def _build_schoolbook(n,m,p):
    rank=n*m*p
    U=[[0]*rank for _ in range(n*m)]; V=[[0]*rank for _ in range(m*p)]; W=[[0]*rank for _ in range(p*n)]
    c=0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c]=1; V[j*p+k][c]=1; W[k*n+i][c]=1; c+=1
    return U,V,W

def _build_tiling(n,m,p):
    # Tile A (4x5) into top 2 rows, bottom 2 rows.
    # Tile B (5x6) into top 2 rows, bottom 3 rows.
    # C (4x6) into 2x2 blocks: C11 (2x3), C12 (2x3), C21 (2x3), C22 (2x3).
    # Wait, A is 4x5, B is 5x6.
    # A_top: 2x5, A_bot: 2x5.
    # B_top: 5x3 (cols 0-2), B_bot: 5x3 (cols 3-5).
    # C11 = A_top * B_top: (2x5)*(5x3)
    # C12 = A_top * B_bot: (2x5)*(5x3)
    # C21 = A_bot * B_top: (2x5)*(5x3)
    # C22 = A_bot * B_bot: (2x5)*(5x3)
    
    # We need decompositions for (2x5)*(5x3).
    # Schoolbook rank: 2*5*3 = 30.
    # Can we do better?
    # Split A_top into A_top1 (2x2) and A_top2 (2x3).
    # Split B_top into B_top1 (2x3) and B_top2 (3x3).
    # C11 = A_top1*B_top1 + A_top2*B_top2.
    # A_top1*B_top1: (2x2)*(2x3). Rank?
    #   Split B_top1 into B_top1a (2x1) and B_top1b (2x2).
    #   A_top1*B_top1a: (2x2)*(2x1). Rank 2*2*1=4? Or Strassen?
    #   (2x2)*(2x1):
    #     A = [a b; c d], B = [e; f].
    #     C = [ae+bf; ce+df].
    #     Strassen:
    #       p1 = (a+d)(e+f) -> c11 +=, c21 +=
    #       p2 = (c+d)(e) -> c21 -=
    #       p3 = (a)(e+f) -> c11 -=
    #       p4 = (d)(f-e) -> c22? No, C is 2x1.
    #     Let's just use schoolbook for small blocks to be safe, then optimize if needed.
    #     Actually, (2x2)*(2x1) rank is 4 (schoolbook). Strassen doesn't help for rectangular 2x2 * 2x1?
    #     M_{2,2,1} = 4.
    #     M_{2,2,2} = 7.
    #     M_{2,3,3}?
    
    # Let's try a different tiling that aligns with known Strassen blocks.
    # A: 4x5. B: 5x6.
    # Split A: 2x2, 2x2, 2x1. (Rows 0-1, 2-3; Cols 0-1, 2-3, 4)
    # Split B: 2x3, 2x3, 1x3. (Rows 0-1, 2-3, 4; Cols 0-2, 3-5)
    
    # This is getting complicated. Let's stick to the 4x4x4 + 4x4x2 + small terms approach from inspiration 1,
    # which gave rank 101.
    # U1: 4x4x4 (rank 49) for A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
    # U2: 4x4x2 (rank 28) for A[0:4,0:4] * B[0:4,4:6] -> C[0:4,4:6]
    # U3: 4x1x6 (rank 24) for A[0:4,4:5] * B[4:5,0:6] -> C[0:4,0:6]
    
    # Total rank: 49 + 28 + 24 = 101.
    
    parts = []
    
    # Part 1: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    U1, V1, W1 = _build_4x4x4()
    parts.append((U1, V1, W1, (0, 0, 4, 4, 0, 4)))
    
    # Part 2: A[0:4, 0:4] * B[0:4, 4:6] -> C[0:4, 4:6]
    U2, V2, W2 = _build_4x4x2()
    parts.append((U2, V2, W2, (0, 0, 4, 4, 4, 2)))
    
    # Part 3: A[0:4, 4:5] * B[4:5, 0:6] -> C[0:4, 0:6]
    # A is 4x1, B is 1x6.
    # This is outer product of a column vector (4x1) and a row vector (1x6).
    # Rank = 4*6 = 24.
    U3 = [[0]*24 for _ in range(4)]
    V3 = [[0]*24 for _ in range(6)]
    W3 = [[0]*24 for _ in range(24)]
    c = 0
    for i in range(4):
        for k in range(6):
            U3[i][c] = 1
            V3[0][c] = 1
            W3[k*4+i][c] = 1
            c += 1
    parts.append((U3, V3, W3, (0, 4, 4, 1, 0, 6)))
    
    U, V, W = _combine(parts, n, m, p)
    return U, V, W

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 120
    
    try:
        U, V, W = _build_tiling(n, m, p)
        rank = len(U[0])
        
        if time.time() > deadline - 20:
            U, V, W = _build_schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}
        
        if _verify(U, V, W, n, m, p, trials=5):
            if rank < n * m * p:
                return {"U": U, "V": V, "W": W}
        
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    except Exception:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
