import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 130

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # ---------- Sub-decomposition builders ----------
    # Each returns (U_block, V_block, W_block) as 2D lists for the local indices,
    # plus we track which global indices they map to via caller.

    def lader_223():
        """M(2,2,3): A 2x2 (u 0..3), B 2x3 (v 0..5), C 2x3 (w 0..5). Rank 10."""
        R = 10
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(6)]
        W = [[0]*R for _ in range(6)]
        a,b,c,d = 0,1,2,3
        x,y,z,uu,vv,ww = 0,1,2,3,4,5
        C00,C01,C02,C10,C11,C12 = 0,1,2,3,4,5
        col=0
        def add(u,v,w):
            nonlocal col
            for i in u: U[i][col]+=1
            for j in v: V[j][col]+=1
            for k in w: W[k][col]+=1
            col+=1
        add([a],[y,uu],[C01,C10])
        add([d],[x,vv],[C00,C11])
        add([a,b],[z,uu],[C02,C10])
        add([c,d],[x,vv],[C00,C11])
        add([c,d],[y,vv],[C01,C11])
        add([a,b],[z,vv],[C02,C11])
        add([c,d],[z,vv],[C02,C12])
        add([a],[y,-vv],[C01,C11])
        add([d],[x,-uu],[C00,C10])
        add([a,b],[z,-uu],[C02,C12])
        return U,V,W

    def m233():
        """M(2,3,3): A 2x3 (u 0..5), B 3x3 (v 0..8), C 2x3 (w 0..5). Rank 9."""
        R = 9
        U = [[0]*R for _ in range(6)]
        V = [[0]*R for _ in range(9)]
        W = [[0]*R for _ in range(6)]
        a,b,c,d,e,f = 0,1,2,3,4,5
        x,y,z,uu,vv,ww,rr,ss,tt = 0,1,2,3,4,5,6,7,8
        C00,C01,C02,C10,C11,C12 = 0,1,2,3,4,5
        col=0
        def add(u,v,w):
            nonlocal col
            for i in u: U[i][col]+=1
            for j in v: V[j][col]+=1
            for k in w: W[k][col]+=1
            col+=1
        add([a,b,c],[x,uu,rr],[C00,C01,C02,C10,C11,C12])
        add([a,e,-f],[y,vv,ss],[C01,C02,C11,C12])
        add([b,f,-e],[z,ww,tt],[C02,C11,C12])
        add([a,e,-f],[x,uu,rr],[C00,C01,C02,C10,C11,C12])
        add([b,f,-e],[y,vv,ss],[C01,C02,C11,C12])
        add([a,e,-f],[z,ww,tt],[C02,C11,C12])
        add([a,d,e],[x,uu,rr],[C00,C01,C02,C10,C11,C12])
        add([b,d,f],[y,vv,ss],[C01,C02,C11,C12])
        add([c,d,e],[z,ww,tt],[C02,C11,C12])
        return U,V,W

    # ---------- Verify a block decomposition ----------
    def verify_block(Ub, Vb, Wb, ni, mi, pi):
        # Ub: (ni*mi, R), Vb: (mi*pi, R), Wb: (pi*ni, R)
        R = len(Ub[0])
        Ua = np.array(Ub, float); Va = np.array(Vb, float); Wa = np.array(Wb, float)
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    s = float(np.sum(Ua[i*mi+j,:]*Va[j*pi+k,:]*Wa[k*ni+i,:]))
                    if abs(s-1.0) > 1e-9:
                        return False
        return True

    # ---------- Assemble ----------
    # A1 = A[0:2,0:2] (2x2), A2 = A[0:2,2:5] (2x3)
    # B1 = B[0:2,:] (2x6), B2 = B[2:5,:] (3x6)
    # C = A1*B1 + A2*B2
    # Split output C cols into 0:3 and 3:6.

    # Global U rows: A1 uses [0,1,5,6] (a,b,c,d of 2x2 in A: row0col0=0,row0col1=1,row1col0=5,row1col1=6)
    # A2 uses [2,3,4,7,8,9] (row0col2=2,row0col3=3,row0col4=4,row1col2=7,row1col3=8,row1col4=9)
    # Global V rows: B1 row0 = [0,1,2,3,4,5], B1 row1=[6,7,8,9,10,11]; B2 row2=[12..17],row3=[18..23],row4=[24..29]
    # Global W rows: C col0=[0,1],col1=[2,3],col2=[4,5],col3=[6,7],col4=[8,9],col5=[10,11]

    # We'll build 4 block decompositions and merge into global.
    blocks = []

    # Block 1: A1(2x2) * B1[:,0:3](2x3) -> C[:,0:3](2x3)  [M(2,2,3)]
    # u_local->global: [0,1,5,6]; v_local->global: [0,1,2,6,7,8]; w_local->global: [0,1,2,3,4,5]
    blocks.append((lader_223(), [0,1,5,6], [0,1,2,6,7,8], [0,1,2,3,4,5]))
    # Block 2: A1(2x2) * B1[:,3:6](2x3) -> C[:,3:6](2x3)  [M(2,2,3)]
    blocks.append((lader_223(), [0,1,5,6], [3,4,5,9,10,11], [6,7,8,9,10,11]))
    # Block 3: A2(2x3) * B2[:,0:3](3x3) -> C[:,0:3](2x3)  [M(2,3,3)]
    blocks.append((m233(), [2,3,4,7,8,9], [12,13,14,18,19,20,24,25,26], [0,1,2,3,4,5]))
    # Block 4: A2(2x3) * B2[:,3:6](3x3) -> C[:,3:6](2x3)  [M(2,3,3)]
    blocks.append((m233(), [2,3,4,7,8,9], [15,16,17,21,22,23,27,28,29], [6,7,8,9,10,11]))

    # Verify each block
    for (Ub,Vb,Wb), umap, vmap, wmap in blocks:
        ni, mi, pi = len(umap)//2, len(umap)//2, len(wmap)//2  # for 223: ni=2,mi=2,pi=3 ; for 233: ni=2,mi=3,pi=3
        # determine dims from maps
        if len(umap)==4:
            ni,mi,pi = 2,2,3
        else:
            ni,mi,pi = 2,3,3
        if not verify_block(Ub,Vb,Wb,ni,mi,pi):
            if time.time() < deadline:
                return schoolbook()

    # Merge
    total_rank = sum(len(Ub[0]) for (Ub,Vb,Wb),_,_,_ in blocks)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    off = 0
    for (Ub,Vb,Wb), umap, vmap, wmap in blocks:
        R = len(Ub[0])
        for li in range(len(umap)):
            gi = umap[li]
            for r in range(R):
                U[gi][off+r] += Ub[li][r]
        for lj in range(len(vmap)):
            gj = vmap[lj]
            for r in range(R):
                V[gj][off+r] += Vb[lj][r]
        for lk in range(len(wmap)):
            gk = wmap[lk]
            for r in range(R):
                W[gk][off+r] += Wb[lk][r]
        off += R

    # Full verification
    if time.time() < deadline:
        Ua = np.array(U, float); Va = np.array(V, float); Wa = np.array(W, float)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.sum(Ua[i*m+j,:]*Va[j*p+k,:]*Wa[k*n+i,:]))
                    if abs(s-1.0) > 1e-9:
                        ok = False
                        break
                if not ok: break
            if not ok: break
        if not ok:
            return schoolbook()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
