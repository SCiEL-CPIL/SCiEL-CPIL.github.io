import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    T_diag = [(i * m + j, j * p + k, k * n + i) for i in range(n) for j in range(m) for k in range(p)]
    T_off = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)
             if b // p == a % m and c % n == a // m and c // n == b % p
             and (a, b, c) not in T_diag]

    start_time = time.time()
    deadline = start_time + 118.0

    def check(U, V, W, tol=1e-7):
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        for (a, b, c) in T_diag:
            if abs(T[a, b, c] - 1.0) > tol:
                return False
        for (a, b, c) in T_off:
            if abs(T[a, b, c]) > tol:
                return False
        return True

    def to_halfint(U, V, W, tol=1e-6):
        U2, V2, W2 = np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2
        if check(U2, V2, W2, tol=tol):
            return U2, V2, W2
        return None

    def to_list(U, V, W):
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}

    # Strassen rank-36 baseline
    U36 = np.zeros((nm, 36))
    V36 = np.zeros((mp, 36))
    W36 = np.zeros((pn, 36))
    S = [([1,0,0,1],[1,0,0,1],[1,0,0,1]),([0,0,1,1],[1,0,0,0],[0,0,1,-1]),
         ([1,0,0,0],[0,1,0,-1],[0,1,0,1]),([0,0,0,1],[-1,0,1,0],[1,0,1,0]),
         ([1,1,0,0],[0,0,0,1],[-1,1,0,0]),([-1,0,1,0],[1,1,0,0],[0,0,0,1]),
         ([0,1,0,-1],[0,0,1,1],[1,0,0,0])]
    B = [([1,0,0,0],[1,0],[1,0]),([0,1,0,0],[0,1],[1,0]),
         ([0,0,1,0],[1,0],[0,1]),([0,0,0,1],[0,1],[0,1])]
    col = 0
    for (ai,aj,bj,bk,ci,ck) in [(0,0,0,0,0,0),(0,0,0,2,0,2),(0,2,2,0,0,0),(0,2,2,2,0,2)]:
        for up,vp,wp in S:
            for ix,v in enumerate(up):
                if v: U[(ai+ix//2)*m+(aj+ix%2),col]=v
            for ix,v in enumerate(vp):
                if v: V[(bj+ix//2)*p+(bk+ix%2),col]=v
            for ix,v in enumerate(wp):
                if v: W[(ck+ix%2)*n+(ci+ix//2),col]=v
            col+=1
    for (ai,aj,bj,bk,ci,ck) in [(0,0,0,4,0,4),(0,2,2,4,0,4)]:
        for up,vp,wp in B:
            for ix,v in enumerate(up):
                if v: U[(ai+ix//2)*m+(aj+ix%2),col]=v
            for ix,v in enumerate(vp):
                if v: V[(bj+ix)*p+bk,col]=v
            for ix,v in enumerate(wp):
                if v: W[ck*n+(ci+ix),col]=v
            col+=1

    if not check(U36, V36, W36):
        r = n*m*p
        Us = np.zeros((nm,r)); Vs = np.zeros((mp,r)); Ws = np.zeros((pn,r))
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j,c2]=1; Vs[j*p+k,c2]=1; Ws[k*n+i,c2]=1; c2+=1
        return to_list(Us, Vs, Ws)

    best = to_list(U36, V36, W36)
    best_r = 36

    # Rank-35: drop one column, solve linear system for W
    for dc in range(36):
        if time.time() > deadline - 12: break
        keep = [c for c in range(36) if c != dc]
        Ur, Vr = U36[:,keep], V36[:,keep]
        A = np.zeros((40,35)); b = np.ones(40); e=0
        for (a,b2,c) in T_diag:
            for t in range(35): A[e,t]=Ur[a,t]*Vr[b2,t]
            e+=1
        Wf,res,rk,sv = np.linalg.lstsq(A,b,rcond=None)
        if len(res) and res[0]>1e-7: continue
        Wr = Wf.reshape(pn,35)
        if check(Ur,Vr,Wr,1e-6):
            h = to_halfint(Ur,Vr,Wr)
            if h:
                best_r = 35; best = to_list(*h); break

    # Rank-34: drop two columns
    if best_r == 35:
        Uc = np.array(best["U"]); Vc = np.array(best["V"])
    else:
        Uc, Vc = U36, V36
    for d1 in range(Uc.shape[1]):
        if time.time() > deadline - 10: break
        for d2 in range(d1+1, Uc.shape[1]):
            if time.time() > deadline - 10: break
            keep = [c for c in range(Uc.shape[1]) if c!=d1 and c!=d2]
            Ur, Vr = Uc[:,keep], Vc[:,keep]
            r = len(keep)
            A = np.zeros((40,r)); b = np.ones(40); e=0
            for (a,b2,c) in T_diag:
                for t in range(r): A[e,t]=Ur[a,t]*Vr[b2,t]
                e+=1
            Wf,res,rk,sv = np.linalg.lstsq(A,b,rcond=None)
            if len(res) and res[0]>1e-7: continue
            Wr = Wf.reshape(pn,r)
            if check(Ur,Vr,Wr,1e-6):
                h = to_halfint(Ur,Vr,Wr)
                if h:
                    best_r = 34; best = to_list(*h); break
        if best_r < 35: break

    return best
# EVOLVE-BLOCK-END
