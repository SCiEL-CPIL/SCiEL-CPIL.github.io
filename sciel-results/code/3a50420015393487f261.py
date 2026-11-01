import time

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=7
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen blocks for k in {0,1}, {2,3}, {4,5}
    # 3 blocks of size 2x2
    for b in range(3):
        base_r = b * 7
        k_offset = 2 * b
        
        # Term 1: P1 = a00 * (b00 + b11)
        r = base_r + 0
        U[0][r] = 1  # a00
        V[0 * p + k_offset + 0][r] = 1  # b00
        V[1 * p + k_offset + 1][r] = 1  # b11
        W[(k_offset + 0) * n + 0][r] = 1  # c00
        W[(k_offset + 1) * n + 1][r] = 1  # c11
        
        # Term 2: P2 = (a00 + a10) * b11
        r = base_r + 1
        U[0][r] = 1  # a00
        U[2][r] = 1  # a10
        V[1 * p + k_offset + 1][r] = 1  # b11
        W[(k_offset + 0) * n + 0][r] = -1  # c00
        W[(k_offset + 0) * n + 1][r] = 1  # c10
        
        # Term 3: P3 = (a01 - a11) * (b10 + b11)
        r = base_r + 2
        U[1][r] = 1  # a01
        U[3][r] = -1  # a11
        V[1 * p + k_offset + 0][r] = 1  # b10
        V[1 * p + k_offset + 1][r] = 1  # b11
        W[(k_offset + 1) * n + 0][r] = 1  # c01
        W[(k_offset + 1) * n + 1][r] = -1  # c11
        
        # Term 4: P4 = a11 * (b00 - b10)
        r = base_r + 3
        U[3][r] = 1  # a11
        V[0 * p + k_offset + 0][r] = 1  # b00
        V[1 * p + k_offset + 0][r] = -1  # b10
        W[(k_offset + 0) * n + 0][r] = 1  # c00
        W[(k_offset + 0) * n + 1][r] = 1  # c10
        
        # Term 5: P5 = (a00 + a01) * b00
        r = base_r + 4
        U[0][r] = 1  # a00
        U[1][r] = 1  # a01
        V[0 * p + k_offset + 0][r] = 1  # b00
        W[(k_offset + 1) * n + 0][r] = 1  # c01
        W[(k_offset + 1) * n + 1][r] = 1  # c11
        
        # Term 6: P6 = (a10 - a00) * (b00 + b01)
        r = base_r + 5
        U[2][r] = 1  # a10
        U[0][r] = -1  # a00
        V[0 * p + k_offset + 0][r] = 1  # b00
        V[0 * p + k_offset + 1][r] = 1  # b01
        W[(k_offset + 0) * n + 0][r] = 1  # c00
        
        # Term 7: P7 = (a01 + a11) * (b10 - b01)
        r = base_r + 6
        U[1][r] = 1  # a01
        U[3][r] = 1  # a11
        V[1 * p + k_offset + 0][r] = 1  # b10
        V[0 * p + k_offset + 1][r] = -1  # b01
        W[(k_offset + 1) * n + 1][r] = 1  # c11

    # Last column k=6
    base_r = 21
    k = 6
    
    # Term 8: a00 * b06 -> c06
    r = base_r + 0
    U[0][r] = 1
    V[0 * p + k][r] = 1
    W[k * n + 0][r] = 1
    
    # Term 9: a01 * b16 -> c06
    r = base_r + 1
    U[1][r] = 1
    V[1 * p + k][r] = 1
    W[k * n + 0][r] = 1
    
    # Term 10: a10 * b06 -> c16
    r = base_r + 2
    U[2][r] = 1
    V[0 * p + k][r] = 1
    W[k * n + 1][r] = 1
    
    # Term 11: a11 * b16 -> c16
    r = base_r + 3
    U[3][r] = 1
    V[1 * p + k][r] = 1
    W[k * n + 1][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
