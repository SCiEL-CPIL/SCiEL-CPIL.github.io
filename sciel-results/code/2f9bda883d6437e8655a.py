import time
import sys

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition: 5 pairs of columns (Strassen rank 7) + 1 single column (Schoolbook rank 4)
    rank = 5 * 7 + 1 * 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # 5 pairs of columns
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # V indices for B: B[0, k], B[1, k]
        # V index convention: j * p + k
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = p + k0   # B[1, k0]
        v_h = p + k1   # B[1, k1]
        
        # W indices for C: C[i, k]
        # W index convention: k * n + i
        w_c00 = k0 * n     # C[0, k0]
        w_c10 = k0 * n + 1 # C[1, k0]
        w_c01 = k1 * n     # C[0, k1]
        w_c11 = k1 * n + 1 # C[1, k1]
        
        # U indices for A: A[i, j]
        # U index convention: i * m + j
        # U[0] = A[0,0], U[1] = A[0,1], U[2] = A[1,0], U[3] = A[1,1]
        
        # Strassen's 7 terms for 2x2 multiplication
        # m1 = (a00 + a11) * (b00 + b11)
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m2 = (a10 + a11) * b00
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        # m3 = a00 * (b01 - b11)
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m4 = a11 * (b10 - b00)
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        # m5 = (a00 + a01) * b11
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        # m6 = (a10 - a00) * (b00 + b01)
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        # m7 = (a01 - a11) * (b10 + b11)
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1

    # Single column k2 = 10
    k2 = 10
    v_e2 = k2       # B[0, k2]
    v_g2 = p + k2   # B[1, k2]
    w_c02 = k2 * n     # C[0, k2]
    w_c12 = k2 * n + 1 # C[1, k2]
    
    # Schoolbook 4 terms
    # a00 * b00
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # a01 * b10
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # a10 * b00
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    # a11 * b10
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}
