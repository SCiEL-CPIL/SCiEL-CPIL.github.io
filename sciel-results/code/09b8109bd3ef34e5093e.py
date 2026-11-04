import time
import sys

DIMENSIONS = [6, 6, 6]

def _verify(U, V, W, n, m, p):
    """Verify that U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] sums to 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            total = 0
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _kronecker_decomp(n, m, p, n1, m1, p1, n2, m2, p2, U1, V1, W1, U2, V2, W2):
    """
    Combine two tensor decompositions via Kronecker product.
    U1 has shape (n1*m1, r1), V1 has shape (m1*p1, r1), W1 has shape (p1*n1, r1)
    U2 has shape (n2*m2, r2), V2 has shape (m2*p2, r2), W2 has shape (p2*n2, r2)
    
    Result: U has shape (n*m, r1*r2), V has shape (m*p, r1*r2), W has shape (p*n, r1*r2)
    where n = n1*n2, m = m1*m2, p = p1*p2.
    
    Index mapping:
    For the combined tensor, the index for U is (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
    The rank index is r1 * r2 + r2 where r1 from first, r2 from second.
    
    U[(i1*n2+i2)*(m1*m2) + (j1*m2+j2), r1*r2+r2] = U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2]
    V[(j1*m2+j2)*(p1*p2) + (k1*p2+k2), r1*r2+r2] = V1[j1*p1+k1, r1] * V2[j2*p2+k2, r2]
    W[(k1*p2+k2)*(n1*n2) + (i1*n2+i2), r1*r2+r2] = W1[k1*n1+i1, r1] * W2[k2*n2+i2, r2]
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                for r1_idx in range(r1):
                    u1 = U1[i1 * m1 + j1][r1_idx]
                    v1 = V1[j1 * p1 + k1][r1_idx]
                    w1 = W1[k1 * n1 + i1][r1_idx]
                    
                    if u1 == 0 and v1 == 0 and w1 == 0:
                        continue
                    
                    for i2 in range(n2):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                for r2_idx in range(r2):
                                    u2 = U2[i2 * m2 + j2][r2_idx]
                                    v2 = V2[j2 * p2 + k2][r2_idx]
                                    w2 = W2[k2 * n2 + i2][r2_idx]
                                    
                                    if u2 == 0 and v2 == 0 and w2 == 0:
                                        continue
                                    
                                    r_combined = r1_idx * r2 + r2_idx
                                    
                                    # U index: (i1*n2+i2) * (m1*m2) + (j1*m2+j2)
                                    ui = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2)
                                    U[ui][r_combined] += u1 * u2
                                    
                                    # V index: (j1*m2+j2) * (p1*p2) + (k1*p2+k2)
                                    vi = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
                                    V[vi][r_combined] += v1 * v2
                                    
                                    # W index: (k1*p2+k2) * (n1*n2) + (i1*n2+i2)
                                    wi = (k1 * p2 + k2) * (n1 * n2) + (i1 * n2 + i2)
                                    W[wi][r_combined] += w1 * w2
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin for the 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build 3x3 schoolbook decomposition (rank 27)
    n1, m1, p1 = 3, 3, 3
    r1 = 27
    U1 = [[0] * r1 for _ in range(n1 * m1)]
    V1 = [[0] * r1 for _ in range(m1 * p1)]
    W1 = [[0] * r1 for _ in range(p1 * n1)]
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j][col] = 1
                V1[j * p1 + k][col] = 1
                W1[k * n1 + i][col] = 1
                col += 1
    
    # Build 2x2 Strassen decomposition (rank 7)
    n2, m2, p2 = 2, 2, 2
    r2 = 7
    U2 = [[0] * r2 for _ in range(n2 * m2)]
    V2 = [[0] * r2 for _ in range(m2 * p2)]
    W2 = [[0] * r2 for _ in range(p2 * n2)]
    
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Indices: A is 2x2, so A[i,j] with i=0,1; j=0,1
    # U[i*m+j, r] where m=2: U[0*2+0]=U[0] is A00, U[1] is A01, U[2] is A10, U[3] is A11
    # V[j*p+k, r] where p=2: V[0*2+0]=V[0] is B00, V[1] is B01, V[2] is B10, V[3] is B11
    # W[k*n+i, r] where n=2: W[0*2+0]=W[0] is C00, W[1] is C01, W[2] is C10, W[3] is C11
    
    # M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # M3 = A00(B10-B11)
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # M4 = A11(B01-B00)
    U2[3][3] = 1
    V2[0][3] = -1; V2[1][3] = 1
    W2[0][3] = 1; W2[2][3] = 1
    
    # M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[0][5] = -1; U2[2][5] = 1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Verify 2x2 Strassen
    if not _verify(U2, V2, W2, 2, 2, 2):
        # Fallback: return schoolbook 6x6
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Combine via Kronecker product
    U, V, W = _kronecker_decomp(
        n, m, p, n1, m1, p1, n2, m2, p2,
        U1, V1, W1, U2, V2, W2
    )
    
    # Verify the result
    if not _verify(U, V, W, 6, 6, 6):
        # Fallback: return schoolbook 6x6
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
