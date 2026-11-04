DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    # Try rank-36 decomposition
    rank = 36
    U = [[0]*rank for _ in range(n*m)]  # 8 x 36
    V = [[0]*rank for _ in range(m*p)]  # 20 x 36
    W = [[0]*rank for _ in range(p*n)]  # 10 x 36
    
    # Strassen 2x2x2 decomposition (7 components)
    # For block (i0,i1) x (j0,j1) x (k0,k1)
    # Returns list of 7 dicts: each with U, V, W entries
    def strassen_222(i_off, j_off, k_off):
        # i_off, j_off, k_off are the starting indices
        # U indexed by [i, j], V by [j, k], W by [i, k] (with W[k*n+i] convention)
        comps = []
        # c1: U[i0,j0]=1,U[i1,j1]=1; V[j0,k0]=1,V[j1,k1]=1; W[i0,k0]=1,W[i1,k1]=1
        comps.append({'U': {(i_off,j_off):1,(i_off+1,j_off+1):1}, 'V': {(j_off,k_off):1,(j_off+1,k_off+1):1}, 'W': {(i_off,k_off):1,(i_off+1,k_off+1):1}})
        # c2: U[i1,j0]=1,U[i1,j1]=1; V[j0,k0]=1; W[i1,k0]=1
        comps.append({'U': {(i_off+1,j_off):1,(i_off+1,j_off+1):1}, 'V': {(j_off,k_off):1}, 'W': {(i_off+1,k_off):1}})
        # c3: U[i0,j0]=1; V[j1,k0]=1,V[j1,k1]=-1; W[i0,k1]=1
        comps.append({'U': {(i_off,j_off):1}, 'V': {(j_off+1,k_off):1,(j_off+1,k_off+1):-1}, 'W': {(i_off,k_off+1):1}})
        # c4: U[i1,j1]=1; V[j0,k0]=1,V[j1,k0]=-1; W[i0,k0]=-1,W[i1,k1]=1
        comps.append({'U': {(i_off+1,j_off+1):1}, 'V': {(j_off,k_off):1,(j_off+1,k_off):-1}, 'W': {(i_off,k_off):-1,(i_off+1,k_off+1):1}})
        # c5: U[i0,j0]=1,U[i0,j1]=1; V[j1,k1]=1; W[i0,k1]=1
        comps.append({'U': {(i_off,j_off):1,(i_off,j_off+1):1}, 'V': {(j_off+1,k_off+1):1}, 'W': {(i_off,k_off+1):1}})
        # c6: U[i1,j0]=1,U[i0,j0]=-1; V[j0,k0]=1,V[j0,k1]=1; W[i1,k1]=1
        comps.append({'U': {(i_off+1,j_off):1,(i_off,j_off):-1}, 'V': {(j_off,k_off):1,(j_off,k_off+1):1}, 'W': {(i_off+1,k_off+1):1}})
        # c7: U[i0,j1]=1,U[i1,j1]=-1; V[j1,k0]=1,V[j1,k1]=1; W[i0,k0]=1,W[i1,k1]=-1
        comps.append({'U': {(i_off,j_off+1):1,(i_off+1,j_off+1):-1}, 'V': {(j_off+1,k_off):1,(j_off+1,k_off+1):1}, 'W': {(i_off,k_off):1,(i_off+1,k_off+1):-1}})
        return comps
    
    col = 0
    # Block A1 (j=0,1) with k=0,1
    for comp in strassen_222(0, 0, 0):
        for (i,j), val in comp['U'].items():
            U[i*m+j][col] += val
        for (j,k), val in comp['V'].items():
            V[j*p+k][col] += val
        for (i,k), val in comp['W'].items():
            W[k*n+i][col] += val
        col += 1
    
    # Block A1 (j=0,1) with k=2,3
    for comp in strassen_222(0, 0, 2):
        for (i,j), val in comp['U'].items():
            U[i*m+j][col] += val
        for (j,k), val in comp['V'].items():
            V[j*p+k][col] += val
        for (i,k), val in comp['W'].items():
            W[k*n+i][col] += val
        col += 1
    
    # A1 (j=0,1) with k=4: schoolbook, 4 components
    for i in range(2):
        for j in range(2):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    
    # Block A2 (j=2,3) with k=0,1
    for comp in strassen_222(0, 2, 0):
        for (i,j), val in comp['U'].items():
            U[i*m+j][col] += val
        for (j,k), val in comp['V'].items():
            V[j*p+k][col] += val
        for (i,k), val in comp['W'].items():
            W[k*n+i][col] += val
        col += 1
    
    # Block A2 (j=2,3) with k=2,3
    for comp in strassen_222(0, 2, 2):
        for (i,j), val in comp['U'].items():
            U[i*m+j][col] += val
        for (j,k), val in comp['V'].items():
            V[j*p+k][col] += val
        for (i,k), val in comp['W'].items():
            W[k*n+i][col] += val
        col += 1
    
    # A2 (j=2,3) with k=4: schoolbook, 4 components
    for i in range(2):
        for j in range(2, 4):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    # Fall back to schoolbook
                    rank = n*m*p
                    U = [[0]*rank for _ in range(n*m)]
                    V = [[0]*rank for _ in range(m*p)]
                    W = [[0]*rank for _ in range(p*n)]
                    c2 = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U[i2*m+j2][c2] = 1
                                V[j2*p+k2][c2] = 1
                                W[k2*n+i2][c2] = 1
                                c2 += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
