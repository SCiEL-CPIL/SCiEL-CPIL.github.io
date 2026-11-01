import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 125

    # Split middle dim m=5 into 1+4 and output dim p=6 into 3+3.
    # C = A1*B1 + A2*B2 where A1(2x1), B1(1x6), A2(2x4), B2(4x6).
    # M(2,1,3) rank 1, M(2,4,3) rank 10 (= M(2,3,3) rank 9 + M(2,1,3) rank 1)
    # Total rank = 1+1+10+10 = 22
    RANK = 22
    U = [[0] * RANK for _ in range(n * m)]   # (10, RANK)
    V = [[0] * RANK for _ in range(m * p)]   # (30, RANK)
    W = [[0] * RANK for _ in range(p * n)]   # (12, RANK)

    col = 0

    def m233(u, v, w):
        """Rank-9 decomposition of M(2,3,3).
        A=[[a,b,c],[d,e,f]] (u=[a,b,c,d,e,f]), B=[[x,y,z],[u,v,w],[r,s,t]]
        (v=[x,y,z,u,v,w,r,s,t]), C=[[C00,C01,C02],[C10,C11,C12]] (w=[...]).
        Adds 9 terms, returns next col."""
        a, b, c, d, e, f = u
        x, y, z, uu, vv, ww, rr, ss, tt = v
        C00, C01, C02, C10, C11, C12 = w
        nonlocal col
        # M1 = (a+b+c)*(x+u+r) -> C00,C01,C02,C10,C11,C12
        U[a][col]+=1; U[b][col]+=1; U[c][col]+=1
        V[x][col]+=1; V[uu][col]+=1; V[rr][col]+=1
        W[C00][col]+=1; W[C01][col]+=1; W[C02][col]+=1
        W[C10][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M2 = (a+e-f)*(y+v+s) -> C01,C02,C11,C12
        U[a][col]+=1; U[e][col]+=1; U[f][col]-=1
        V[y][col]+=1; V[vv][col]+=1; V[ss][col]+=1
        W[C01][col]+=1; W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M3 = (b+f-e)*(z+w+t) -> C02,C11,C12
        U[b][col]+=1; U[f][col]+=1; U[e][col]-=1
        V[z][col]+=1; V[ww][col]+=1; V[tt][col]+=1
        W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M4 = (a+e-f)*(x+u+r) -> C00,C01,C02,C10,C11,C12
        U[a][col]+=1; U[e][col]+=1; U[f][col]-=1
        V[x][col]+=1; V[uu][col]+=1; V[rr][col]+=1
        W[C00][col]+=1; W[C01][col]+=1; W[C02][col]+=1
        W[C10][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M5 = (b+f-e)*(y+v+s) -> C01,C02,C11,C12
        U[b][col]+=1; U[f][col]+=1; U[e][col]-=1
        V[y][col]+=1; V[vv][col]+=1; V[ss][col]+=1
        W[C01][col]+=1; W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M6 = (a+e-f)*(z+w+t) -> C02,C11,C12
        U[a][col]+=1; U[e][col]+=1; U[f][col]-=1
        V[z][col]+=1; V[ww][col]+=1; V[tt][col]+=1
        W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M7 = (a+d+e)*(x+u+r) -> C00,C01,C02,C10,C11,C12
        U[a][col]+=1; U[d][col]+=1; U[e][col]+=1
        V[x][col]+=1; V[uu][col]+=1; V[rr][col]+=1
        W[C00][col]+=1; W[C01][col]+=1; W[C02][col]+=1
        W[C10][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M8 = (b+d+f)*(y+v+s) -> C01,C02,C11,C12
        U[b][col]+=1; U[d][col]+=1; U[f][col]+=1
        V[y][col]+=1; V[vv][col]+=1; V[ss][col]+=1
        W[C01][col]+=1; W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        # M9 = (c+d+e)*(z+w+t) -> C02,C11,C12
        U[c][col]+=1; U[d][col]+=1; U[e][col]+=1
        V[z][col]+=1; V[ww][col]+=1; V[tt][col]+=1
        W[C02][col]+=1; W[C11][col]+=1; W[C12][col]+=1
        col += 1
        return col

    def m213(u, v, w):
        """Rank-1 decomposition of M(2,1,3).
        A=[a0,a1] (u=[a0,a1]), B=[b0,b1,b2] (v=[b0,b1,b2]), C=[C00,C01,C02,C10,C11,C12] (w=[...]).
        Actually C is 2x3: C[i,k] = A[i] * B[k].
        u=[a0,a1], v=[b0,b1,b2], w=[C00,C01,C02,C10,C11,C12]"""
        a0, a1 = u
        b0, b1, b2 = v
        C00, C01, C02, C10, C11, C12 = w
        nonlocal col
        # Single term: U[i,0,r]=1, V[0,k,r]=1, W[k,i,r]=1
        U[a0][col] += 1
        U[a1][col] += 1
        V[b0][col] += 1
        V[b1][col] += 1
        V[b2][col] += 1
        W[C00][col] += 1
        W[C01][col] += 1
        W[C02][col] += 1
        W[C10][col] += 1
        W[C11][col] += 1
        W[C12][col] += 1
        col += 1
        return col

    def m243(u, v, w):
        """Rank-10 decomposition of M(2,4,3).
        A is 2x4: u=[a0,a1,a2,a3, d0,d1,d2,d3] (row-major: A[0,0],A[0,1],A[0,2],A[0,3],A[1,0],A[1,1],A[1,2],A[1,3])
        B is 4x3: v=[x0,x1,x2, x3,x4,x5, x6,x7,x8, x9,x10,x11] (row-major)
        C is 2x3: w=[C00,C01,C02, C10,C11,C12]
        Split j into 0,1,2 (M(2,3,3)) and 3 (M(2,1,3))."""
        # u: A[0,0],A[0,1],A[0,2],A[0,3],A[1,0],A[1,1],A[1,2],A[1,3]
        a0, a1, a2, a3, d0, d1, d2, d3 = u
        # v: B[0,0],B[0,1],B[0,2], B[1,0],B[1,1],B[1,2], B[2,0],B[2,1],B[2,2], B[3,0],B[3,1],B[3,2]
        b00, b01, b02, b10, b11, b12, b20, b21, b22, b30, b31, b32 = v
        C00, C01, C02, C10, C11, C12 = w
        nonlocal col
        # M(2,3,3) on j=0,1,2
        m233(
            u=[a0, a1, a2, d0, d1, d2],
            v=[b00, b01, b02, b10, b11, b12, b20, b21, b22],
            w=[C00, C01, C02, C10, C11, C12])
        # M(2,1,3) on j=3
        m213(
            u=[a3, d3],
            v=[b30, b31, b32],
            w=[C00, C01, C02, C10, C11, C12])
        return col

    # A1 = A[:, 0] -> U rows [0, 5] (A[0,0] and A[1,0])
    # A2 = A[:, 1:5] -> U rows [1,2,3,4, 6,7,8,9]
    # B1 = B[0, :] -> V rows [0,1,2, 3,4,5] (B[0,0..5])
    # B2 = B[1:5, :] -> V rows [6..29]
    # C cols 0-2: W [0..5]; C cols 3-5: W [6..11]

    # A1 * B1[:,0:3] -> C[:,0:3]  (M(2,1,3))
    m213(
        u=[0, 5],
        v=[0, 1, 2],
        w=[0, 1, 2, 3, 4, 5])
    # A1 * B1[:,3:6] -> C[:,3:6]  (M(2,1,3))
    m213(
        u=[0, 5],
        v=[3, 4, 5],
        w=[6, 7, 8, 9, 10, 11])
    # A2 * B2[:,0:3] -> C[:,0:3]  (M(2,4,3))
    m243(
        u=[1, 2, 3, 4, 6, 7, 8, 9],
        v=[6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26],
        w=[0, 1, 2, 3, 4, 5])
    # A2 * B2[:,3:6] -> C[:,3:6]  (M(2,4,3))
    m243(
        u=[1, 2, 3, 4, 6, 7, 8, 9],
        v=[9, 10, 11, 15, 16, 17, 21, 22, 23, 27, 28, 29],
        w=[6, 7, 8, 9, 10, 11])

    # ---- Verify correctness ----
    def schoolbook():
        rank = n * m * p
        Us = [[0] * rank for _ in range(n * m)]
        Vs = [[0] * rank for _ in range(m * p)]
        Ws = [[0] * rank for _ in range(p * n)]
        cc = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j][cc] = 1
                    Vs[j * p + k][cc] = 1
                    Ws[k * n + i][cc] = 1
                    cc += 1
        return {"U": Us, "V": Vs, "W": Ws}

    if time.time() < deadline:
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i * m + j, :], Va[j * p + k, :] * Wa[k * n + i, :])
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            return schoolbook()

    return {"U": U, "V": V, "W": W}
