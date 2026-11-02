import time

DIMENSIONS = [3, 3, 6]

def _true_verify(U, V, W, n, m, p, rank, tol=1e-9):
    # For all i,k,j,j': sum_c U[i*m+j,c] V[j'*p+k,c] W[k*n+i,c] == delta_{j,j'}
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[jp*p+k][c] * W[k*n+i][c]
                    target = 1.0 if j == jp else 0.0
                    if abs(s - target) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _block_3x3x2():
    n, m, p = 3, 3, 2
    rank = 9
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    # Strassen core (i,j in {0,1}, k in {0,1})
    U[0][0]=1; U[4][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[4][0]=1
    U[3][1]=1; U[4][1]=1; V[0][1]=1; V[1][1]=1; W[3][1]=1; W[4][1]=1
    U[0][2]=1; V[3][2]=1; V[2][2]=-1; W[0][2]=1; W[3][2]=-1
    U[4][3]=1; V[0][3]=1; V[1][3]=-1; W[1][3]=-1; W[4][3]=1
    U[0][4]=1; U[1][4]=1; V[2][4]=1; W[0][4]=1; W[3][4]=1
    U[3][5]=1; U[4][5]=1; V[1][5]=1; W[1][5]=1; W[4][5]=1
    U[1][6]=1; U[3][6]=1; V[1][6]=1; V[2][6]=-1; W[1][6]=1; W[3][6]=-1
    # term7: i==2 part
    for j in range(m):
        U[2*m+j][7] = 1
    for j in range(m):
        V[j][7] = 1
    for k in range(p):
        for i in range(n):
            W[k*n+i][7] = 1
    # term8: j==2, i in {0,1}
    U[0*m+2][8] = 1
    U[1*m+2][8] = 1
    V[2][8] = 1
    for k in range(p):
        for i in range(n):
            W[k*n+i][8] = 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110.0
    try:
        blk = _block_3x3x2()
        per = 9
        U = [[0]*(3*per) for _ in range(n*m)]
        V = [[0]*(3*per) for _ in range(m*p)]
        W = [[0]*(3*per) for _ in range(p*n)]
        for i in range(n*m):
            for c in range(per):
                U[i][c] = blk["U"][i][c]
        for blk_i in range(3):
            base = blk_i*per
            for j in range(m):
                for t in range(2):
                    k = 2*blk_i + t
                    for c in range(per):
                        V[j*p+k][base+c] = blk["V"][j*2+t][c]
            for t in range(2):
                k = 2*blk_i + t
                for i in range(n):
                    for c in range(per):
                        W[k*n+i][base+c] = blk["W"][t*3+i][c]
        if _true_verify(U, V, W, n, m, p, 3*per):
            if time.time() < deadline:
                return {"U": U, "V": V, "W": W}
        return _schoolbook(n, m, p)
    except Exception:
        return _schoolbook(n, m, p)
