import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use 5 blocks of size 2 (Strassen) and 1 block of size 1 (Schoolbook)
    # Total rank = 5 * 7 + 1 * 4 = 35 + 4 = 39
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    k_start = 0
    
    # 5 blocks of 2x2
    for _ in range(5):
        # Strassen decomposition for A (2x2) * B_block (2x2)
        # B_block columns are k_start, k_start+1
        # A indices: (0,0), (0,1), (1,0), (1,1) -> U[0], U[1], U[2], U[3]
        # B indices: 
        # j=0, k=k_start -> V[0*p + k_start]
        # j=0, k=k_start+1 -> V[0*p + k_start+1]
        # j=1, k=k_start -> V[1*p + k_start]
        # j=1, k=k_start+1 -> V[1*p + k_start+1]
        
        # W indices:
        # k=k_start, i=0 -> W[k_start*n + 0]
        # k=k_start, i=1 -> W[k_start*n + 1]
        # k=k_start+1, i=0 -> W[(k_start+1)*n + 0]
        # k=k_start+1, i=1 -> W[(k_start+1)*n + 1]
        
        # m1 = (a00 + a11) * (b00 + b11)
        U[0][col] += 1
        U[3][col] += 1
        V[0*p + k_start][col] += 1
        V[1*p + k_start+1][col] += 1
        # C00 += m1, C11 += m1
        W[k_start*n + 0][col] += 1
        W[(k_start+1)*n + 1][col] += 1
        col += 1
        
        # m2 = (a10 + a11) * b00
        U[2][col] += 1
        U[3][col] += 1
        V[0*p + k_start][col] += 1
        # C10 += m2
        W[k_start*n + 1][col] += 1
        col += 1
        
        # m3 = a00 * (b01 - b11)
        U[0][col] += 1
        V[0*p + k_start+1][col] += 1
        V[1*p + k_start+1][col] += -1
        # C01 += m3
        W[(k_start+1)*n + 0][col] += 1
        col += 1
        
        # m4 = a11 * (b10 - b00)
        U[3][col] += 1
        V[1*p + k_start][col] += 1
        V[0*p + k_start][col] += -1
        # C00 += m4, C10 += m4
        W[k_start*n + 0][col] += 1
        W[k_start*n + 1][col] += 1
        col += 1
        
        # m5 = (a00 + a01) * b11
        U[0][col] += 1
        U[1][col] += 1
        V[1*p + k_start+1][col] += 1
        # C00 -= m5, C01 += m5
        W[k_start*n + 0][col] += -1
        W[(k_start+1)*n + 0][col] += 1
        col += 1
        
        # m6 = (a10 - a00) * (b00 + b01)
        U[2][col] += 1
        U[0][col] += -1
        V[0*p + k_start][col] += 1
        V[0*p + k_start+1][col] += 1
        # C11 += m6
        W[(k_start+1)*n + 1][col] += 1
        col += 1
        
        # m7 = (a01 - a11) * (b10 + b11)
        U[1][col] += 1
        U[3][col] += -1
        V[1*p + k_start][col] += 1
        V[1*p + k_start+1][col] += 1
        # C00 += m7, C11 -= m7
        W[k_start*n + 0][col] += 1
        W[(k_start+1)*n + 1][col] += -1
        col += 1
        
        k_start += 2
        
    # 1 block of 2x1 (last column k_start=10)
    # Schoolbook for A (2x2) * v (2x1)
    # m1 = a00 * v0
    U[0][col] += 1
    V[0*p + k_start][col] += 1
    W[k_start*n + 0][col] += 1
    col += 1
    
    # m2 = a01 * v1
    U[1][col] += 1
    V[1*p + k_start][col] += 1
    W[k_start*n + 0][col] += 1
    col += 1
    
    # m3 = a10 * v0
    U[2][col] += 1
    V[0*p + k_start][col] += 1
    W[k_start*n + 1][col] += 1
    col += 1
    
    # m4 = a11 * v1
    U[3][col] += 1
    V[1*p + k_start][col] += 1
    W[k_start*n + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
