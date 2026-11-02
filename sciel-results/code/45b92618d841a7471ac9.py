import numpy as np

DIMENSIONS = [4, 4, 4]

def _s22_terms():
    # 7 Strassen products for 2x2: each (x_combo, y_combo) over entries 0..3=(00,01,10,11)
    return [
        ([(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)]),
        ([(0,1)], [(1,1),(3,-1)]),
        ([(3,1)], [(2,1),(0,-1)]),
        ([(0,1),(1,1)], [(3,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)]),
    ]

def _zmap():
    return [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
        [((1,0),1)],
        [((0,0),-1),((1,1),1)],
    ]

def _blockMs():
    return [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)]),
    ]

def _Csigns():
    return [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
        [((1,0),1)],
        [((0,0),-1),((1,1),1)],
    ]

def _build():
    n = m = p = 4
    s22 = _s22_terms(); zmap = _zmap()
    terms = []
    for mi, (a_combo, b_combo) in enumerate(_blockMs()):
        # build X (2x2 coeff) and Y (2x2 coeff)
        X = [[0,0],[0,0]]; Y = [[0,0],[0,0]]
        for (I,J,s) in a_combo: X[I][J] += s
        for (K,L,t) in b_combo: Y[K][L] += t
        for xi, (x_combo, y_combo) in enumerate(s22):
            u = {}
            for (xidx, xs) in x_combo:
                ii, jj = divmod(xidx, 2)
                xc = X[ii][jj]
                if xc == 0: continue
                for aii in range(2):
                    for ajj in range(2):
                        a_entry = (2*ii+aii)*4 + (2*jj+ajj)
                        u[a_entry] = u.get(a_entry, 0) + xs*xc
            v = {}
            for (yidx, ys) in y_combo:
                kk, ll = divmod(yidx, 2)
                yc = Y[kk][ll]
                if yc == 0: continue
                for bkk in range(2):
                    for bll in range(2):
                        b_entry = (2*kk+bkk)*4 + (2*ll+bll)
                        v[b_entry] = v.get(b_entry, 0) + ys*yc
            w = {}
            for (Iprime, Jprime, cs) in _Csigns()[mi]:
                for (zidx, zs) in zmap[xi]:
                    ii2, jj2 = divmod(zidx, 2)
                    c_row = 2*Iprime + ii2
                    c_col = 2*Jprime + jj2
                    c_entry = c_col*4 + c_row   # W row = k*n+i = col*4+row
                    w[c_entry] = w.get(c_entry, 0) + cs*zs
            terms.append((u, v, w))
    R = len(terms)
    U = [[0.0]*R for _ in range(n*m)]
    V = [[0.0]*R for _ in range(m*p)]
    W = [[0.0]*R for _ in range(p*n)]
    for r,(u,v,w) in enumerate(terms):
        for e,c in u.items(): U[e][r] += c
        for e,c in v.items(): V[e][r] += c
        for e,c in w.items(): W[e][r] += c
    return U, V, W

def _verify(U,V,W):
    n=m=p=4
    Ua=np.array(U); Va=np.array(V); Wa=np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(np.sum(Ua[i*4+j]*Va[j*4+k]*Wa[k*4+i]) - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n,m,p = DIMENSIONS
    U,V,W = _build()
    if _verify(U,V,W):
        return {"U":U,"V":V,"W":W}
    R=64
    U=[[0]*R for _ in range(n*m)]; V=[[0]*R for _ in range(m*p)]; W=[[0]*R for _ in range(p*n)]
    col=0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col]=1; V[j*p+k][col]=1; W[k*n+i][col]=1; col+=1
    return {"U":U,"V":V,"W":W}
