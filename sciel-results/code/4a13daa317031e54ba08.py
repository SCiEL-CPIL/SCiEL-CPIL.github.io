"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor using NumPy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(result, 1.0, atol=1e-9):
        return False
    return True

def _strassen_2x2():
    """Return Strassen's 7-term decomposition for 2x2 x 2x2."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _schoolbook_1x1x2():
    """Schoolbook 1x1x2: rank 2."""
    n, m, p = 1, 1, 2
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

def _kronecker(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Kronecker product of two matrix multiplication decompositions.
    (n1,m1,p1) x (n2,m2,p2) -> (n1*n2, m1*m2, p1*p2)
    Index convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2
    n, m, p = n1*n2, m1*m2, p1*p2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # U[i*m+j, r1*r2+r2] = U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2]
    # where i = i1*n2 + i2, j = j1*m2 + j2
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    row_U = (i1*n2 + i2) * m + (j1*m2 + j2)
                    for r1i in range(r1):
                        v1 = U1[i1*m1 + j1][r1i]
                        if v1 == 0: continue
                        for r2i in range(r2):
                            v2 = U2[i2*m2 + j2][r2i]
                            if v2 == 0: continue
                            col = r1i * r2 + r2i
                            U[row_U][col] = v1 * v2
    
    # V[j*p+k, r] = V1[j1*p1+k1, r1] * V2[j2*p2+k2, r2]
    # where j = j1*m2 + j2, k = k1*p2 + k2
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    row_V = (j1*m2 + j2) * p + (k1*p2 + k2)
                    for r1i in range(r1):
                        v1 = V1[j1*p1 + k1][r1i]
                        if v1 == 0: continue
                        for r2i in range(r2):
                            v2 = V2[j2*p2 + k2][r2i]
                            if v2 == 0: continue
                            col = r1i * r2 + r2i
                            V[row_V][col] = v1 * v2
    
    # W[k*n+i, r] = W1[k1*n1+i1, r1] * W2[k2*n2+i2, r2]
    # where k = k1*p2 + k2, i = i1*n2 + i2
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    row_W = (k1*p2 + k2) * n + (i1*n2 + i2)
                    for r1i in range(r1):
                        v1 = W1[k1*n1 + i1][r1i]
                        if v1 == 0: continue
                        for r2i in range(r2):
                            v2 = W2[k2*n2 + i2][r2i]
                            if v2 == 0: continue
                            col = r1i * r2 + r2i
                            W[row_W][col] = v1 * v2
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Build 2x2x4 via Kronecker(2x2x2 Strassen, 1x1x2 schoolbook)
    # 2x2x2 has rank 7, 1x1x2 has rank 2, so 2x2x4 has rank 14
    U_s, V_s, W_s = _strassen_2x2()
    U_112, V_112, W_112 = _schoolbook_1x1x2()
    d_224 = _kronecker(U_s, V_s, W_s, U_112, V_112, W_112, 2, 2, 2, 1, 1, 2)
    
    # Verify 2x2x4
    if _verify(d_224["U"], d_224["V"], d_224["W"], 2, 2, 4):
        # Build 4x4x8 via Kronecker(2x2x4, 2x2x2 Strassen)
        # 2x2x4 has rank 14, 2x2x2 has rank 7, so 4x4x8 has rank 98
        d_448 = _kronecker(d_224["U"], d_224["V"], d_224["W"], U_s, V_s, W_s, 2, 2, 4, 2, 2, 2)
        
        kron_rank = len(d_448["U"][0])
        if kron_rank < best_rank:
            if _verify(d_448["U"], d_448["V"], d_448["W"], n, m, p):
                best = d_448
                best_rank = kron_rank
    else:
        # Fallback: try the other Kronecker order
        d_224_alt = _kronecker(U_112, V_112, W_112, U_s, V_s, W_s, 1, 1, 2, 2, 2, 2)
        if _verify(d_224_alt["U"], d_224_alt["V"], d_224_alt["W"], 2, 2, 4):
            d_448_alt = _kronecker(U_s, V_s, W_s, d_224_alt["U"], d_224_alt["V"], d_224_alt["W"], 2, 2, 2, 2, 2, 4)
            kron_rank = len(d_448_alt["U"][0])
            if kron_rank < best_rank:
                if _verify(d_448_alt["U"], d_448_alt["V"], d_448_alt["W"], n, m, p):
                    best = d_448_alt
                    best_rank = kron_rank
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best
    
    return best
# EVOLVE-BLOCK-END
