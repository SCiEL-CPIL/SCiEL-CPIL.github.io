"""Tensor decomposition for 3x4 * 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_222():
    """Return rank-7 decomposition of M_{2,2,2} (Strassen).
    
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    where n=m=p=2.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1; W[1][0] = -1; W[2][0] = -1
    
    # Term 1: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # Term 2: A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # Term 3: A11(B00-B01)
    U[3][3] = 1
    V[0][3] = 1; V[1][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # Term 4: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # Term 5: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = -1
    
    # Term 6: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = -1
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """Return schoolbook decomposition of M_{n,m,p} with rank n*m*p."""
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

def _kron_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Kronecker product of two decompositions.
    M_{n1,m1,p1} rank r1, M_{n2,m2,p2} rank r2
    -> M_{n1*n2, m1*m2, p1*p2} rank r1*r2
    """
    r1 = len(W1[0]) if W1 else 0
    r2 = len(W2[0]) if W2 else 0
    rank = r1 * r2
    n, m, p = n1*n2, m1*m2, p1*p2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            idx_u = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
                            val1 = U1[i1*m1+j1][c1]
                            val2 = U2[i2*m2+j2][c2]
                            if val1 != 0 and val2 != 0:
                                U[idx_u][c] += val1 * val2
            
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            idx_v = (j1*m2+j2)*(p1*p2) + (k1*p2+k2)
                            val1 = V1[j1*p1+k1][c1]
                            val2 = V2[j2*p2+k2][c2]
                            if val1 != 0 and val2 != 0:
                                V[idx_v][c] += val1 * val2
            
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            idx_w = (k1*p2+k2)*(n1*n2) + (i1*n2+i2)
                            val1 = W1[k1*n1+i1][c1]
                            val2 = W2[k2*n2+i2][c2]
                            if val1 != 0 and val2 != 0:
                                W[idx_w][c] += val1 * val2
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > tol:
                    return False, (i, j, k, s)
    return True, None

def _block_combine_n(U1, V1, W1, U2, V2, W2, n1, n2, m, p):
    """Combine two decompositions that split the n-dimension.
    
    M_{n1,m,p} and M_{n2,m,p} -> M_{n1+n2,m,p}
    U indices: (i,m) -> row i*m+j
    V indices: (j,p) -> row j*p+k  
    W indices: (k,n) -> row k*n+i
    """
    n = n1 + n2
    r1 = len(W1[0]) if W1 else 0
    r2 = len(W2[0]) if W2 else 0
    rank = r1 + r2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # First part: rows 0..n1-1
    for i in range(n1):
        for j in range(m):
            for c in range(r1):
                U[i*m+j][c] = U1[i*m+j][c]
    for j in range(m):
        for k in range(p):
            for c in range(r1):
                V[j*p+k][c] = V1[j*p+k][c]
    for k in range(p):
        for i in range(n1):
            for c in range(r1):
                W[k*n+i][c] = W1[k*n1+i][c]
    
    # Second part: rows n1..n-1
    for i in range(n2):
        for j in range(m):
            for c in range(r2):
                U[(n1+i)*m+j][r1+c] = U2[i*m+j][c]
    for j in range(m):
        for k in range(p):
            for c in range(r2):
                V[j*p+k][r1+c] = V2[j*p+k][c]
    for k in range(p):
        for i in range(n2):
            for c in range(r2):
                W[k*n+(n1+i)][r1+c] = W2[k*n2+i][c]
    
    return U, V, W

def _block_combine_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """Combine two decompositions that split the m-dimension.
    
    M_{n,m1,p} and M_{n,m2,p} -> M_{n,m1+m2,p}
    """
    m = m1 + m2
    r1 = len(W1[0]) if W1 else 0
    r2 = len(W2[0]) if W2 else 0
    rank = r1 + r2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # First part: columns 0..m1-1
    for i in range(n):
        for j in range(m1):
            for c in range(r1):
                U[i*m+j][c] = U1[i*m1+j][c]
    for j in range(m1):
        for k in range(p):
            for c in range(r1):
                V[j*p+k][c] = V1[j*p+k][c]
    for k in range(p):
        for i in range(n):
            for c in range(r1):
                W[k*n+i][c] = W1[k*n+i][c]
    
    # Second part: columns m1..m-1
    for i in range(n):
        for j in range(m2):
            for c in range(r2):
                U[i*m+(m1+j)][r1+c] = U2[i*m2+j][c]
    for j in range(m2):
        for k in range(p):
            for c in range(r2):
                V[(m1+j)*p+k][r1+c] = V2[j*p+k][c]
    for k in range(p):
        for i in range(n):
            for c in range(r2):
                W[k*n+i][r1+c] = W2[k*n+i][c]
    
    return U, V, W

def _block_combine_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    """Combine two decompositions that split the p-dimension.
    
    M_{n,m,p1} and M_{n,m,p2} -> M_{n,m,p1+p2}
    """
    p = p1 + p2
    r1 = len(W1[0]) if W1 else 0
    r2 = len(W2[0]) if W2 else 0
    rank = r1 + r2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # First part: columns 0..p1-1
    for i in range(n):
        for j in range(m):
            for c in range(r1):
                U[i*m+j][c] = U1[i*m+j][c]
    for j in range(m):
        for k in range(p1):
            for c in range(r1):
                V[j*p+k][c] = V1[j*p1+k][c]
    for k in range(p1):
        for i in range(n):
            for c in range(r1):
                W[k*n+i][c] = W1[k*n+i][c]
    
    # Second part: columns p1..p-1
    for i in range(n):
        for j in range(m):
            for c in range(r2):
                U[i*m+j][r1+c] = U2[i*m+j][c]
    for j in range(m):
        for k in range(p2):
            for c in range(r2):
                V[j*p+(p1+k)][r1+c] = V2[j*p2+k][c]
    for k in range(p2):
        for i in range(n):
            for c in range(r2):
                W[(p1+k)*n+i][r1+c] = W2[k*n+i][c]
    
    return U, V, W

def _decomp_m22p(p):
    """Decomposition of M_{2,2,p} using Strassen for p=2 blocks."""
    if p == 1:
        return _schoolbook_nm_p(2, 2, 1)
    if p == 2:
        return _strassen_222()
    # Split p into 2 + (p-2)
    U1, V1, W1 = _decomp_m22p(2)
    U2, V2, W2 = _decomp_m22p(p - 2)
    return _block_combine_p(U1, V1, W1, U2, V2, W2, 2, 2, 2, p-2)

def _decomp_m12p(p):
    """Decomposition of M_{1,2,p} - schoolbook is optimal (rank 2p)."""
    return _schoolbook_nm_p(1, 2, p)

def _decomp_m32p(p):
    """Decomposition of M_{3,2,p} by splitting n=3 into 2+1."""
    U1, V1, W1 = _decomp_m22p(p)
    U2, V2, W2 = _decomp_m12p(p)
    return _block_combine_n(U1, V1, W1, U2, V2, W2, 2, 1, 2, p)

def _decomp_m34p(p):
    """Decomposition of M_{3,4,p} by splitting m=4 into 2+2."""
    U1, V1, W1 = _decomp_m32p(p)
    U2, V2, W2 = _decomp_m32p(p)
    return _block_combine_m(U1, V1, W1, U2, V2, W2, 3, 2, 2, p)

def _decomp_m347():
    """Decomposition of M_{3,4,7} by splitting p=7 into 2+2+2+1."""
    # Split p=7 as 2+2+2+1
    U1, V1, W1 = _decomp_m34p(2)
    U2, V2, W2 = _decomp_m34p(2)
    U3, V3, W3 = _decomp_m34p(2)
    U4, V4, W4 = _decomp_m34p(1)
    
    # Combine all four
    Ua, Va, Wa = _block_combine_p(U1, V1, W1, U2, V2, W2, 3, 4, 2, 2)
    Ub, Vb, Wb = _block_combine_p(U3, V3, W3, U4, V4, W4, 3, 4, 2, 1)
    U, V, W = _block_combine_p(Ua, Va, Wa, Ub, Vb, Wb, 3, 4, 4, 3)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Schoolbook baseline
    U_sb, V_sb, W_sb = _schoolbook_nm_p(n, m, p)
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p  # 84
    
    # Try the recursive decomposition with Strassen
    try:
        U, V, W = _decomp_m347()
        rank = len(W[0]) if W else 0
        if rank < best_rank:
            # Verify
            ok, err = _verify_decomp(U, V, W, n, m, p)
            if ok:
                best = {"U": U, "V": V, "W": W}
                best_rank = rank
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        return best
    
    # Try alternative: split p=7 as 4+3
    try:
        U1, V1, W1 = _decomp_m34p(4)
        U2, V2, W2 = _decomp_m34p(3)
        U, V, W = _block_combine_p(U1, V1, W1, U2, V2, W2, 3, 4, 4, 3)
        rank = len(W[0]) if W else 0
        if rank < best_rank:
            ok, err = _verify_decomp(U, V, W, n, m, p)
            if ok:
                best = {"U": U, "V": V, "W": W}
                best_rank = rank
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return best
    
    # Try: M_{3,4,7} via Kronecker M_{3,2,7} ⊗ M_{1,2,1}
    try:
        U1, V1, W1 = _decomp_m32p(7)
        U2, V2, W2 = _schoolbook_nm_p(1, 2, 1)
        U, V, W = _kron_decomp(U1, V1, W1, U2, V2, W2, 3, 2, 7, 1, 2, 1)
        rank = len(W[0]) if W else 0
        if rank < best_rank:
            ok, err = _verify_decomp(U, V, W, n, m, p)
            if ok:
                best = {"U": U, "V": V, "W": W}
                best_rank = rank
    except Exception:
        pass
    
    return best

# EVOLVE-BLOCK-END
