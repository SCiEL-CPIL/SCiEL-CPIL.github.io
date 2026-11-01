import time

DIMENSIONS = [2, 2, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes C = A*B."""
    rank = len(U[0])
    # Test with random matrices
    import random
    random.seed(42)
    for _ in range(5):
        A = [[random.randint(-5,5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5,5) for _ in range(p)] for _ in range(m)]
        # Compute C = A*B
        C = [[sum(A[i][j]*B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        # Compute using decomposition
        for i in range(n):
            for k in range(p):
                total = 0
                for r in range(rank):
                    L = sum(U[i*m+j][r] * A[i][j] for j in range(m))
                    M = sum(V[j*p+k][r] * B[j][k] for j in range(m))
                    total += W[k*n+i][r] * L * M
                if total != C[i][k]:
                    return False
    return True

def _strassen_21():
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for block in range(3):
        base = block * 7
        c0 = 2 * block
        c1 = 2 * block + 1
        
        # m1 = (A00+A11)(B00+B11)
        U[0][base+0] += 1; U[3][base+0] += 1
        V[c0][base+0] += 1; V[2+c1][base+0] += 1
        W[c0*2+0][base+0] += 1; W[c1*2+1][base+0] += 1
        
        # m2 = (A10+A11)B00
        U[2][base+1] += 1; U[3][base+1] += 1
        V[c0][base+1] += 1
        W[c0*2+1][base+1] += 1; W[c1*2+1][base+1] += -1
        
        # m3 = A00(B10-B11)
        U[0][base+2] += 1
        V[2+c0][base+2] += 1; V[2+c1][base+2] += -1
        W[c1*2+0][base+2] += 1; W[c1*2+1][base+2] += 1
        
        # m4 = A11(B01-B10)
        U[3][base+3] += 1
        V[c1][base+3] += 1; V[2+c0][base+3] += -1
        W[c0*2+0][base+3] += 1; W[c0*2+1][base+3] += 1
        
        # m5 = (A00+A01)B11
        U[0][base+4] += 1; U[1][base+4] += 1
        V[2+c1][base+4] += 1
        W[c0*2+0][base+4] += -1; W[c1*2+0][base+4] += 1
        
        # m6 = (A10-A00)(B00+B10)
        U[2][base+5] += 1; U[0][base+5] += -1
        V[c0][base+5] += 1; V[2+c0][base+5] += 1
        W[c1*2+1][base+5] += 1
        
        # m7 = (A01-A11)(B01+B11)
        U[1][base+6] += 1; U[3][base+6] += -1
        V[c1][base+6] += 1; V[2+c1][base+6] += 1
        W[c0*2+0][base+6] += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = 2, 2, 6
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
    start = time.time()
    deadline = start + 140  # 10s margin
    
    # Try Strassen 21
    result = _strassen_21()
    if _verify(result["U"], result["V"], result["W"], 2, 2, 6):
        return result
    
    # Fall back to schoolbook
    return _schoolbook()
