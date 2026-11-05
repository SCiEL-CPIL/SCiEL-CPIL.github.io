import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    
    def check_validity(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def generate_strassen(n, m, p):
        if n == 2 and m == 2 and p == 2:
            rank = 7
            U = [[0]*rank for _ in range(4)]
            V = [[0]*rank for _ in range(4)]
            W = [[0]*rank for _ in range(4)]
            
            # M1 = (A00+A11)(B00+B11)
            U[0][0] += 1; U[3][0] += 1
            V[0][0] += 1; V[3][0] += 1
            W[0][0] += 1; W[3][0] += 1
            
            # M2 = (A10+A11)B00
            U[2][1] += 1; U[3][1] += 1
            V[0][1] += 1
            W[1][1] += 1; W[3][1] += -1
            
            # M3 = A00(B01-B11)
            U[0][2] += 1
            V[1][2] += 1; V[3][2] += -1
            W[2][2] += 1; W[3][2] += 1
            
            # M4 = A11(B10-B00)
            U[3][3] += 1
            V[2][3] += 1; V[0][3] += -1
            W[0][3] += 1; W[1][3] += 1
            
            # M5 = (A00+A01)B11
            U[0][4] += 1; U[1][4] += 1
            V[3][4] += 1
            W[0][4] += -1; W[2][4] += 1
            
            # M6 = (A10-A00)(B00+B01)
            U[2][5] += 1; U[0][5] += -1
            V[0][5] += 1; V[1][5] += 1
            W[3][5] += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[1][6] += 1; U[3][6] += -1
            V[2][6] += 1; V[3][6] += 1
            W[0][6] += 1
            
            return U, V, W, rank
        
        if m > 2:
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1, r1 = generate_strassen(n, m1, p)
            U2, V2, W2, r2 = generate_strassen(n, m2, p)
            rank = r1 + r2
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            for i in range(n):
                for j in range(m1):
                    U[i*m+j][0:r1] = U1[i*m1+j]
                for j in range(m2):
                    U[i*m+m1+j][r1:r1+r2] = U2[i*m2+j]
            for j in range(m1):
                for k in range(p):
                    V[j*p+k][0:r1] = V1[j*p+k]
            for j in range(m2):
                for k in range(p):
                    V[(m1+j)*p+k][r1:r1+r2] = V2[j*p+k]
            for k in range(p):
                for i in range(n):
                    W[k*n+i][0:r1] = W1[k*n+i]
                    W[k*n+i][r1:r1+r2] = W2[k*n+i]
            return U, V, W, rank
        elif p > 2:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1, r1 = generate_strassen(n, m, p1)
            U2, V2, W2, r2 = generate_strassen(n, m, p2)
            rank = r1 + r2
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            for i in range(n):
                for j in range(m):
                    U[i*m+j][0:r1] = U1[i*m+j]
                    U[i*m+j][r1:r1+r2] = U2[i*m+j]
            for j in range(m):
                for k in range(p1):
                    V[j*p+k][0:r1] = V1[j*p1+k]
                for k in range(p2):
                    V[j*p+p1+k][r1:r1+r2] = V2[j*p2+k]
            for k in range(p1):
                for i in range(n):
                    W[k*n+i][0:r1] = W1[k*n+i]
            for k in range(p2):
                for i in range(n):
                    W[(p1+k)*n+i][r1:r1+r2] = W2[k*n+i]
            return U, V, W, rank
        elif n > 2:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1, r1 = generate_strassen(n1, m, p)
            U2, V2, W2, r2 = generate_strassen(n2, m, p)
            rank = r1 + r2
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            for i in range(n1):
                for j in range(m):
                    U[i*m+j][0:r1] = U1[i*m+j]
            for i in range(n2):
                for j in range(m):
                    U[(n1+i)*m+j][r1:r1+r2] = U2[i*m+j]
            for j in range(m):
                for k in range(p):
                    V[j*p+k][0:r1] = V1[j*p+k]
                    V[j*p+k][r1:r1+r2] = V2[j*p+k]
            for k in range(p):
                for i in range(n1):
                    W[k*n+i][0:r1] = W1[k*n+i]
                for i in range(n2):
                    W[k*n+n1+i][r1:r1+r2] = W2[k*n+i]
            return U, V, W, rank
        else:
            raise ValueError("Base case not reached")
    
    U, V, W, rank = generate_strassen(n, m, p)
    
    # Verify
    if not check_validity(U, V, W, n, m, p):
        # Fallback: schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
