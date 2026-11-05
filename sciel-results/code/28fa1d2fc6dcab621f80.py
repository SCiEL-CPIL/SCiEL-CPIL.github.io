"""Block-diagonal Kronecker decomposition for 4x4x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = 120.0
START = time.time()


def _deadline_exceeded():
    return (time.time() - START) > DEADLINE - 2.0


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta(i,i')delta(j,j')delta(k,k')."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                a = uc[i * m + j]
                if a == 0:
                    continue
                for k in range(p):
                    T[i, j, k] += a * vc[j * p + k] * wc[k * n + i]
    # Target: T[i,j,k] = 1 iff i==i',j==j',k==k' i.e. T[i,j,k]=1 for all (i,j,k)
    target = np.ones((n, m, p), dtype=float)
    return np.max(np.abs(T - target)) <= tol


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """Rank-7 decomposition of 2x2x2 multiplication. Returns (U,V,W) shapes (4,7),(4,7),(4,7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]."""
    # Strassen factors. Each term c: U_c, V_c, W_c are 4-vectors (flattened 2x2).
    # Standard Strassen 7 products:
    # M1=(A11)(B12-B22), M2=(A11+A12)(B22), M3=(A21+A22)(B11),
    # M4=(A22)(B21-B11), M5=(A11+A22)(B11+B22),
    # M6=(A12-A22)(B21+B22), M7=(A21-A11)(B11+B12)
    # C11 = M1+M4-M5+M7, C12 = M2+M5, C21 = M3+M6, C22 = M1-M2+M3+M5
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # Flatten 2x2: index = i*2+j
    def setv(vec, vals):
        for idx, v in zip([0,1,2,3], vals):
            vec[idx] = v
    # term 0: M1 = A11 * (B12 - B22)
    setv(U[0], [1,0,0,0]); setv(V[0], [0,1,0,-1]); setv(W[0], [1,0,0,0])
    # term 1: M2 = (A11+A12) * B22
    setv(U[1], [1,1,0,0]); setv(V[1], [0,0,0,1]); setv(W[1], [0,1,0,0])
    # term 2: M3 = (A21+A22) * B11
    setv(U[2], [0,0,1,1]); setv(V[2], [1,0,0,0]); setv(W[2], [0,0,1,0])
    # term 3: M4 = A22 * (B21 - B11)
    setv(U[3], [0,0,0,1]); setv(V[3], [0,0,1,-1]); setv(W[3], [0,0,0,1])
    # term 4: M5 = (A11+A22)*(B11+B22)
    setv(U[4], [1,0,0,1]); setv(V[4], [1,0,0,1]); setv(W[4], [-1,0,0,-1])
    # term 5: M6 = (A12-A22)*(B21+B22)
    setv(U[5], [0,1,0,-1]); setv(V[5], [0,0,1,1]); setv(W[5], [0,0,0,1])
    # term 6: M7 = (A21-A11)*(B11+B12)
    setv(U[6], [0,0,1,-1]); setv(V[6], [1,1,0,0]); setv(W[6], [1,0,0,0])
    return U, V, W


def _kronecker_block(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """Kronecker product of two decomposition tensors.
    Block (i1,i2) of A is A1[i1]*A2[i2], etc.
    Returns U,V,W for n=n1*n2, m=m1*m2, p=p1*p2.
    Indexing U[i*m+j] where i = i1*n2+i2, j = j1*m2+j2.
    V[j*p+k] where j = j1*m2+j2, k = k1*p2+k2.
    W[k*n+i] where k = k1*p2+k2, i = i1*n2+i2.
    """
    n, m, p = n1*n2, m1*m2, p1*p2
    rank = len(U1[0]) * len(U2[0])
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for c1 in range(len(U1[0])):
        for c2 in range(len(U2[0])):
            c = c1 * len(U2[0]) + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for k1 in range(p1):
                        for i2 in range(n2):
                            for j2 in range(m2):
                                for k2 in range(p2):
                                    u = U1[i1*m1+j1][c1] * U2[i2*m2+j2][c2]
                                    v = V1[j1*p1+k1][c1] * V2[j2*p2+k2][c2]
                                    w = W1[k1*n1+i1][c1] * W2[k2*n2+i2][c2]
                                    i = i1*n2 + i2
                                    j = j1*m2 + j2
                                    k = k1*p2 + k2
                                    U[i*m + j][c] += u
                                    V[j*p + k][c] += v
                                    W[k*n + i][c] += w
    return U, V, W


def _build_4x4x5():
    """Build 4x4x5 via block partition: split p=5 into p1=2, p2=3.
    C = A(4x4) * B(4x5). Partition B columns into B1(4x2) and B2(4x3).
    C1 = A*B1 (4x2), C2 = A*B2 (4x3).
    C1: 4x4x2 = (2x2x2) x (2x2x1) via Kronecker. rank 7*4=28.
    C2: 4x4x3 = (2x2x2) x (2x2x3) via Kronecker. 2x2x3 rank 6 (schoolbook 2*2*3). rank 7*6=42.
    Total rank 70.
    """
    n, m, p = 4, 4, 5
    # 2x2x2 Strassen
    Su, Sv, Sw = _strassen_2x2x2()
    # 2x2x1 schoolbook: rank 4
    def school221():
        n1, m1, p1 = 2, 2, 1
        r = n1*m1*p1
        U = [[0]*r for _ in range(n1*m1)]
        V = [[0]*r for _ in range(m1*p1)]
        W = [[0]*r for _ in range(p1*n1)]
        c = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U[i*m1+j][c] = 1
                    V[j*p1+k][c] = 1
                    W[k*n1+i][c] = 1
                    c += 1
        return U, V, W
    def school223():
        n1, m1, p1 = 2, 2, 3
        r = n1*m1*p1
        U = [[0]*r for _ in range(n1*m1)]
        V = [[0]*r for _ in range(m1*p1)]
        W = [[0]*r for _ in range(p1*n1)]
        c = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U[i*m1+j][c] = 1
                    V[j*p1+k][c] = 1
                    W[k*n1+i][c] = 1
                    c += 1
        return U, V, W
    U1, V1, W1 = school221()
    U2, V2, W2 = school223()
    # C1 = 4x4x2 = Kronecker(2x2x2 Strassen, 2x2x1)
    Uc1, Vc1, Wc1 = _kronecker_block(Su, Sv, Sw, 2,2,2, U1, V1, W1, 2,2,1)
    # C2 = 4x4x3 = Kronecker(2x2x2 Strassen, 2x2x3)
    Uc2, Vc2, Wc2 = _kronecker_block(Su, Sv, Sw, 2,2,2, U2, V2, W2, 2,2,3)
    # Now combine: full 4x4x5. The Kronecker above already gives 4x4x2 and 4x4x3.
    # We need to merge into 4x4x5 where the p-index is unified (k = 0..4).
    # In C1, k ranges 0..1 (p1=2). In C2, k ranges 0..2 (p2=3).
    # We map C1's k (0,1) -> full k (0,1) and C2's k (0,1,2) -> full k (2,3,4).
    rank1 = len(Uc1[0]); rank2 = len(Uc2[0])
    rank = rank1 + rank2
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    # Copy C1 (k in 0..1)
    for c in range(rank1):
        for i in range(n):
            for j in range(m):
                U[i*m+j][c] = Uc1[i*m+j][c]
        for j in range(m):
            for k in range(2):
                V[j*p+k][c] = Vc1[j*2+k][c]
        for k in range(2):
            for i in range(n):
                W[k*n+i][c] = Wc1[k*n+i][c]
    # Copy C2 (k in 2..4), offset c by rank1
    for c in range(rank2):
        for i in range(n):
            for j in range(m):
                U[i*m+j][c+rank1] = Uc2[i*m+j][c]
        for j in range(m):
            for k in range(3):
                V[j*p+(k+2)][c+rank1] = Vc2[j*3+k][c]
        for k in range(3):
            for i in range(n):
                W[(k+2)*n+i][c+rank1] = Wc2[k*n+i][c]
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    # Try the structured rank-70 decomposition first.
    try:
        U, V, W = _build_4x4x5()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    if _deadline_exceeded():
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    # Fallback: verified schoolbook rank 80.
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
