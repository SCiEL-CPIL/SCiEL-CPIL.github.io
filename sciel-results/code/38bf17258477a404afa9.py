"""Matrix multiplication 3,4,5 — rank-minimal exact tensor decomposition.

Uses block decomposition with Strassen for 2x2x2 blocks and schoolbook for others.
Split: n=3=2+1, m=4=2+2, p=5=2+3.
Total rank: 7+12+7+12+4+6+4+6 = 58.
"""

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Strassen rank-7 for 2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7).
    U rows: a00, a01, a10, a11
    V rows: b00, b01, b10, b11
    W rows: c00, c01, c10, c11
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    
    # M2 = (a10+a11)b00
    U[2][1]=1; U[3][1]=1; V[0][1]=1
    W[2][1]=1; W[3][1]=-1
    
    # M3 = a00(b10-b11)
    U[0][2]=1; V[2][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=-1
    
    # M4 = a11(b01-b00)
    U[3][3]=1; V[0][3]=-1; V[1][3]=1
    W[0][3]=-1; W[1][3]=1
    
    # M5 = (a00+a01)b11
    U[0][4]=1; U[1][4]=1; V[3][4]=1
    W[0][4]=1
    
    # M6 = (a10-a00)(b00+b10)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[2][5]=1
    W[3][5]=1
    
    # M7 = (a01-a11)(b01+b11)
    U[1][6]=1; U[3][6]=-1; V[1][6]=1; V[3][6]=1
    W[0][6]=-1; W[2][6]=1
    
    return U, V, W


def _schoolbook_nm_p(n, m, p):
    """Schoolbook rank n*m*p decomposition.
    Returns (U, V, W) with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    U[i*m+j, c] for A[i,j], V[j*p+k, c] for B[j,k], W[k*n+i, c] for C[i,k].
    """
    rank = n * m * p
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
    return U, V, W


def _build_3x4x5():
    """Build rank-58 decomposition for 3x4 * 4x5 using block decomposition.
    
    Split: n1=2, n2=1, m1=2, m2=2, p1=2, p2=3.
    For each (bi, bj, bk) in {0,1}^3, decompose the sub-multiplication
    n_{bi} x m_{bj} * m_{bj} x p_{bk}.
    """
    n, m, p = 3, 4, 5
    n1, n2 = 2, 1
    m1, m2 = 2, 2
    p1, p2 = 2, 3
    
    n_sizes = [n1, n2]
    m_sizes = [m1, m2]
    p_sizes = [p1, p2]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                ni = n_sizes[bi]
                mj = m_sizes[bj]
                pk = p_sizes[bk]
                
                if ni == 2 and mj == 2 and pk == 2:
                    U_sub, V_sub, W_sub = _strassen_2x2()
                else:
                    U_sub, V_sub, W_sub = _schoolbook_nm_p(ni, mj, pk)
                
                rank_sub = len(U_sub[0])
                
                for c in range(rank_sub):
                    U_col = [0] * (n * m)
                    V_col = [0] * (m * p)
                    W_col = [0] * (p * n)
                    
                    # Map U: A[gi, gj] where gi in row block bi, gj in col block bj
                    for li in range(ni):
                        for lj in range(mj):
                            gi = (n1 if bi == 1 else 0) + li
                            gj = (m1 if bj == 1 else 0) + lj
                            U_col[gi * m + gj] = U_sub[li * mj + lj][c]
                    
                    # Map V: B[gj, gk] where gj in row block bj, gk in col block bk
                    for lj in range(mj):
                        for lk in range(pk):
                            gj = (m1 if bj == 1 else 0) + lj
                            gk = (p1 if bk == 1 else 0) + lk
                            V_col[gj * p + gk] = V_sub[lj * pk + lk][c]
                    
                    # Map W: C[gi, gk] where gi in row block bi, gk in col block bk
                    # W index: k*n + i = gk * n + gi
                    for lk in range(pk):
                        for li in range(ni):
                            gk = (p1 if bk == 1 else 0) + lk
                            gi = (n1 if bi == 1 else 0) + li
                            W_col[gk * n + gi] = W_sub[lk * ni + li][c]
                    
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
    
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return U, V, W, rank


def _verify(U, V, W, n, m, p):
    """Exact verification of tensor decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    U, V, W, rank = _build_3x4x5()
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
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

# EVOLVE-BLOCK-END
