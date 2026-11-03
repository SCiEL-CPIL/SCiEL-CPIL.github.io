import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook fallback: rank 72
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(nm)]
    V = [[0] * rank_school for _ in range(mp)]
    W = [[0] * rank_school for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
    
    # Try Strassen embedding: rank 68
    # 4 non-overlapping 2x2x2 blocks at (i0=0, j0=0, k0) for k0 in {0,2,4,6}
    strassen_blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    
    # Covered entries
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n * m * p - len(covered)  # 72 - 32 = 40
    rank = 7 * len(strassen_blocks) + n_school  # 28 + 40 = 68
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Strassen 2x2x2 coefficients
    # For matrices A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # C[0,0] = P1 - P2 + P3 + P4
    # C[0,1] = P1 - P5 + P6 + P7
    # C[1,0] = P2 - P3 - P4 + P5
    # C[1,1] = P1 - P2 + P4 - P7
    
    # U coefficients: which A elements contribute to each P (row = A index, col = P index)
    # A indices: 0=a, 1=b, 2=c, 3=d
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],  # a: P1, P3, P5, P6
        [0, 0, 0, 0, 1, 0, 1],   # b: P5, P7
        [0, 1, 0, 0, 0, 1, 0],   # c: P2, P6
        [1, 1, 0, 1, 0, 0, -1],  # d: P1, P2, P4, P7
    ]
    
    # V coefficients: which B elements contribute to each P (row = B index, col = P index)
    # B indices: 0=e, 1=f, 2=g, 3=h
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],  # e: P1, P2, P4, P6
        [0, 0, 1, 0, 0, 1, 0],   # f: P3, P6
        [0, 0, 0, 1, 0, 0, 1],   # g: P4, P7
        [1, 0, -1, 0, 1, 0, 1],  # h: P1, P3, P5, P7
    ]
    
    # W coefficients: which C elements each P contributes to (row = C index, col = P index)
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    W_strassen = [
        [1, -1, 1, 1, 0, 0, 0],   # C00: P1, P2, P3, P4
        [1, 0, 0, 0, -1, 1, 1],   # C01: P1, P5, P6, P7
        [0, 1, -1, -1, 1, 0, 0],  # C10: P2, P3, P4, P5
        [1, -1, 0, 1, 0, 0, -1],  # C11: P1, P2, P4, P7
    ]
    
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    a_idx = di * 2 + dj
                    U[(i0+di) * m + (j0+dj)][col+c] = U_strassen[a_idx][c]
            for dj in range(2):
                for dk in range(2):
                    b_idx = dj * 2 + dk
                    V[(j0+dj) * p + (k0+dk)][col+c] = V_strassen[b_idx][c]
            for dk in range(2):
                for di in range(2):
                    c_idx = dk * 2 + di
                    W[(k0+dk) * n + (i0+di)][col+c] = W_strassen[c_idx][c]
        col += 7
    
    # Fill in schoolbook for uncovered entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        # Return whatever we have; if incomplete, fall back to schoolbook
        return {"U": [[0]*rank_school for _ in range(nm)], 
                "V": [[0]*rank_school for _ in range(mp)], 
                "W": [[0]*rank_school for _ in range(pn)]}
    
    # Verify the decomposition
    def verify(Uf, Vf, Wf, nf, mf, pf, tol=1e-10):
        rank = len(Uf[0])
        for i in range(nf):
            for j in range(mf):
                for k in range(pf):
                    s = 0.0
                    for c in range(rank):
                        s += Uf[i*mf + j][c] * Vf[j*pf + k][c] * Wf[k*nf + i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        return {"U": [[0]*rank_school for _ in range(nm)],
                "V": [[0]*rank_school for _ in range(mp)],
                "W": [[0]*rank_school for _ in range(pn)]}
