import time

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block for columns k0, k1
    def add_strassen(k0, k1):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        U[0][col] += 1
        U[3][col] += 1
        V[k0][col] += 1
        V[7+k1][col] += 1
        W[2*k0][col] += 1
        W[2*k1+1][col] += 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[2][col] += 1
        U[3][col] += 1
        V[k0][col] += 1
        W[2*k0+1][col] += 1
        col += 1
        
        # M3 = A11(B12-B22)
        U[0][col] += 1
        V[k1][col] += 1
        V[7+k1][col] -= 1
        W[2*k1][col] += 1
        W[2*k1+1][col] += 1
        col += 1
        
        # M4 = A22(B21-B11)
        U[3][col] += 1
        V[7+k0][col] += 1
        V[k0][col] -= 1
        W[2*k0][col] += 1
        W[2*k0+1][col] += 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[0][col] += 1
        U[1][col] += 1
        V[7+k1][col] += 1
        W[2*k0][col] -= 1
        W[2*k1][col] += 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[0][col] -= 1
        U[2][col] += 1
        V[k0][col] += 1
        V[k1][col] += 1
        W[2*k1+1][col] += 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1][col] += 1
        U[3][col] -= 1
        V[7+k0][col] += 1
        V[7+k1][col] += 1
        W[2*k0][col] += 1
        W[2*k1+1][col] -= 1
        col += 1

    # Helper to add standard block for column k
    def add_standard(k):
        nonlocal col
        # A11 * B1k -> C1k
        U[0][col] += 1
        V[k][col] += 1
        W[2*k][col] += 1
        col += 1
        
        # A12 * B2k -> C1k
        U[1][col] += 1
        V[7+k][col] += 1
        W[2*k][col] += 1
        col += 1
        
        # A21 * B1k -> C2k
        U[2][col] += 1
        V[k][col] += 1
        W[2*k+1][col] += 1
        col += 1
        
        # A22 * B2k -> C2k
        U[3][col] += 1
        V[7+k][col] += 1
        W[2*k+1][col] += 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_standard(6)
    
    return {"U": U, "V": V, "W": W}
