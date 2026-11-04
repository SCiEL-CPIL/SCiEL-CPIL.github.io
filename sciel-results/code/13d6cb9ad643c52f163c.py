"""Matrix multiplication 6x6x6 via Strassen(2x2) ⊗ Schoolbook(3x3)."""
import time

DIMENSIONS = [6, 6, 6]

def _strassen_2x2():
    """Return U, V, W for 2x2 matrix multiplication with rank 7 (Strassen).
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] where n=m=p=2.
    """
    n = m = p = 2
    rank = 7
    # Strassen's 7 multiplications for 2x2 matrix product
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # Result: [[P1+P4+P5-P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
    
    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    U = [[0]*rank for _ in range(4)]  # U[i*2+j, c]
    V = [[0]*rank for _ in range(4)]  # V[j*2+k, c]
    W = [[0]*rank for _ in range(4)]  # W[k*2+i, c]
    
    # c=0: P1 = (a+d)(e+h)
    # (a+d): U[0][0]=1, U[3][0]=1 (a at (0,0), d at (1,1))
    # (e+h): V[0][0]=1, V[3][0]=1 (e at (0,0), h at (1,1))
    # P1 contributes to C[0,0] and C[1,1]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C[0,0] gets P1
    W[3][0] = 1  # C[1,1] gets P1
    
    # c=1: P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1  # c at (1,0), d at (1,1)
    V[0][1] = 1  # e at (0,0)
    W[2][1] = 1  # C[1,0] gets P2
    W[3][1] = -1 # C[1,1] gets -P2
    
    # c=2: P3 = a(f-h)
    U[0][2] = 1  # a at (0,0)
    V[1][2] = 1; V[3][2] = -1  # f at (0,1), -h at (1,1)
    W[1][2] = 1  # C[0,1] gets P3
    W[3][2] = 1  # C[1,1] gets P3
    
    # c=3: P4 = d(g-e)
    U[3][3] = 1  # d at (1,1)
    V[2][3] = 1; V[0][3] = -1  # g at (1,0), -e at (0,0)
    W[0][3] = 1  # C[0,0] gets P4
    W[2][3] = 1  # C[1,0] gets P4
    
    # c=4: P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1  # a at (0,0), b at (0,1)
    V[3][4] = 1  # h at (1,1)
    W[0][4] = 1  # C[0,0] gets P5
    W[1][4] = 1  # C[0,1] gets P5
    
    # c=5: P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1  # c at (1,0), -a at (0,0)
    V[0][5] = 1; V[1][5] = 1  # e at (0,0), f at (0,1)
    W[3][5] = 1  # C[1,1] gets P6
    
    # c=6: P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1  # b at (0,1), -d at (1,1)
    V[2][6] = 1; V[3][6] = 1  # g at (1,0), h at (1,1)
    W[0][6] = -1  # C[0,0] gets -P7
    
    return U, V, W

def _schoolbook_3x3():
    """Return U, V, W for 3x3 matrix multiplication with rank 27 (schoolbook).
    U has shape (9, 27), V has shape (9, 27), W has shape (9, 27).
    Convention: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c] where n=m=p=3.
    """
    n = m = p = 3
    rank = 27
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3 + j][col] = 1
                V[j*3 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
    return U, V, W

def _kron_product(U1, V1, W1, U2, V2, W2):
    """Compute Kronecker product of two tensor decompositions.
    If (U1,V1,W1) decomposes multiplication of size n1 x m1 x p1 with rank r1,
    and (U2,V2,W2) decomposes multiplication of size n2 x m2 x p2 with rank r2,
    then the Kronecker product decomposes n1*n2 x m1*m2 x p1*p2 with rank r1*r2.
    
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    """
    n1, m1, p1 = U1[0].__len__() and (len(U1)//m1, m1, len(W1)//n1) if False else None
    # Let me extract dimensions properly
    r1 = len(U1[0])
    n1 = len(U1) // m1  # need to figure out m1
    # Actually, let me pass dimensions explicitly or compute from shapes
    # U1 shape: (n1*m1, r1), V1 shape: (m1*p1, r1), W1 shape: (p1*n1, r1)
    # We need to infer n1, m1, p1 from the shapes
    
    # From U1: rows = n1*m1
    # From V1: rows = m1*p1  
    # From W1: rows = p1*n1
    # r1 = len(U1[0])
    
    rows_u1 = len(U1)
    rows_v1 = len(V1)
    rows_w1 = len(W1)
    r1 = len(U1[0])
    
    rows_u2 = len(U2)
    rows_v2 = len(V2)
    rows_w2 = len(W2)
    r2 = len(U2[0])
    
    # n1*m1 = rows_u1, m1*p1 = rows_v1, p1*n1 = rows_w1
    # For our case: n1=2,m1=2,p1=2 so all are 4
    # n2=3,m2=3,p2=3 so all are 9
    
    # We can solve: m1 = rows_u1 / n1, but we know the specific cases
    # Let's just use the known dimensions
    # Actually, for correctness, let me infer:
    # m1^2 = rows_u1 * rows_v1 / rows_w1 ... this gets complicated
    # Since we know n1=m1=p1=2 and n2=m2=p2=3, let me just handle that
    
    # General approach: n1 = rows_w1 // p1, but we need p1 first
    # p1 = rows_v1 // m1, m1 = rows_u1 // n1, n1 = rows_w1 // p1
    # This is circular. For our specific problem, let me hardcode the inference:
    # n1*m1=4, m1*p1=4, p1*n1=4 => n1=m1=p1=2
    # n2*m2=9, m2*p2=9, p2*n2=9 => n2=m2=p2=3
    
    # Let me just compute it numerically for the general case using the fact
    # that for square matrices n=m=p, so n1 = sqrt(rows_u1) if n1==m1
    import math
    n1 = int(round(math.sqrt(rows_u1)))
    n2 = int(round(math.sqrt(rows_u2)))
    m1 = n1
    p1 = n1
    m2 = n2
    p2 = n2
    
    # Verify
    assert n1 * m1 == rows_u1
    assert m1 * p1 == rows_v1
    assert p1 * n1 == rows_w1
    assert n2 * m2 == rows_u2
    assert m2 * p2 == rows_v2
    assert p2 * n2 == rows_w2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    rank = r1 * r2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            # For U: U[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
            # maps to U[(i1*n2+i2)*m + (j1*m2+j2), c]
            # = U[(i1*n2+i2)*m1*m2 + j1*m2 + j2, c]
            for i1 in range(n1):
                for j1 in range(m1):
                    val1 = U1[i1*m1 + j1][c1]
                    if val1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            val2 = U2[i2*m2 + j2][c2]
                            if val2 == 0:
                                continue
                            idx = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                            U[idx][c] += val1 * val2
            
            # For V: V[j1*p1+k1, c1] * V2[j2*p2+k2, c2]
            # maps to V[(j1*p2+j2)*p + (k1*n2+k2), c]
            # Wait, let me think about this more carefully.
            # V has shape (m*p, rank). V[j*p+k, c]
            # V1[j1*p1+k1, c1], V2[j2*p2+k2, c2]
            # Combined: j = j1*p2 + j2, k = k1*n2 + k2? No...
            # Actually, the Kronecker product structure:
            # If A = A1 ⊗ A2, B = B1 ⊗ B2, then AB = (A1B1) ⊗ (A2B2)
            # A1 is n1 x m1, A2 is n2 x m2
            # A = A1 ⊗ A2 is (n1*n2) x (m1*m2)
            # Index: A[i1*n2+i2, j1*m2+j2] = A1[i1,j1] * A2[i2,j2]
            # 
            # B1 is m1 x p1, B2 is m2 x p2
            # B = B1 ⊗ B2 is (m1*m2) x (p1*p2)
            # Index: B[j1*p2+j2, k1*p2... wait
            # B[j1*m2+j2? No.
            # B is (m1*m2) x (p1*p2)
            # B[i, j] where i ranges over m1*m2 and j ranges over p1*p2
            # B = B1 ⊗ B2: B[(j1*m2+j2), (k1*p2+k2)]? 
            # Actually, Kronecker product: (B1 ⊗ B2)[j1*m2+j2, k1*p2+k2] = B1[j1,k1] * B2[j2,k2]
            # Hmm, but the row index of B goes from 0 to m1*m2-1, and it's indexed by (j1, j2)
            # where j1 is from B1's row (m1 values) and j2 is from B2's row (m2 values)
            # So row index = j1*m2 + j2
            # Column index = k1*p2 + k2
            # 
            # So V_combined[(j1*m2+j2)*p + (k1*p2+k2), c] = V1[j1*p1+k1, c1] * V2[j2*p2+k2, c2]
            # where p = p1*p2
            for j1 in range(m1):
                for k1 in range(p1):
                    val1 = V1[j1*p1 + k1][c1]
                    if val1 == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            val2 = V2[j2*p2 + k2][c2]
                            if val2 == 0:
                                continue
                            idx = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                            V[idx][c] += val1 * val2
            
            # For W: W[k1*n1+i1, c1] * W2[k2*n2+i2, c2]
            # W has shape (p*n, rank). W[k*n+i, c]
            # W1[k1*n1+i1, c1], W2[k2*n2+i2, c2]
            # Combined: k = k1*n2+k2, i = i1*p2+i2? 
            # W is (p1*p2) x (n1*n2)
            # W[(k1*n2+k2), (i1*n2+i2)]? 
            # Actually W[k*n+i, c] where k ranges over p and i ranges over n
            # W_combined[k*n+i, c] where k = k1*p2+k2? No...
            # p = p1*p2, n = n1*n2
            # The index k goes from 0 to p-1 = p1*p2-1
            # k = k1*p2 + k2 where k1 in [0,p1), k2 in [0,p2)
            # i = i1*n2 + i2 where i1 in [0,n1), i2 in [0,n2)
            # W_combined[(k1*p2+k2)*n + (i1*n2+i2), c] = W1[k1*n1+i1, c1] * W2[k2*n2+i2, c2]
            for k1 in range(p1):
                for i1 in range(n1):
                    val1 = W1[k1*n1 + i1][c1]
                    if val1 == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            val2 = W2[k2*n2 + i2][c2]
                            if val2 == 0:
                                continue
                            idx = (k1 * p2 + k2) * n + (i1 * n2 + i2)
                            W[idx][c] += val1 * val2
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6
    
    # Get 2x2 Strassen decomposition (rank 7)
    U1, V1, W1 = _strassen_2x2()
    
    # Get 3x3 schoolbook decomposition (rank 27)
    U2, V2, W2 = _schoolbook_3x3()
    
    # Kronecker product gives 6x6x6 with rank 7*27 = 189
    U, V, W = _kron_product(U1, V1, W1, U2, V2, W2)
    
    # Verify dimensions
    assert len(U) == 36
    assert len(U[0]) == 189
    assert len(V) == 36
    assert len(V[0]) == 189
    assert len(W) == 36
    assert len(W[0]) == 189
    
    return {"U": U, "V": V, "W": W}
