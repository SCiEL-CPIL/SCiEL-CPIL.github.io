import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    # rank = 68
    rank = 68
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for q in range(4):
        # Local to global k mapping
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # 17 terms for this block
        
        # Term 0: M1 = (a00+a11)(b00+b11)
        U[0][col] += 1.0
        U[4][col] += 1.0
        V[k0][col] += 1.0
        V[8+k1][col] += 1.0 # b11 is j=1, k=k1 -> 1*8+k1
        W[k0*n + 0][col] += 1.0
        W[k1*n + 1][col] += 1.0
        col += 1
        
        # Term 1: M2 = (a10+a11)b00
        U[2][col] += 1.0 # a10 is i=1, j=0 -> 1*3+0=3? No.
        # A indices: i=row, j=col.
        # a00: i=0, j=0 -> idx 0
        # a01: i=0, j=1 -> idx 1
        # a10: i=1, j=0 -> idx 3
        # a11: i=1, j=1 -> idx 4
        # a02: i=0, j=2 -> idx 2
        # a12: i=1, j=2 -> idx 5
        # a20: i=2, j=0 -> idx 6
        # a21: i=2, j=1 -> idx 7
        # a22: i=2, j=2 -> idx 8
        
        # Correction on U indices:
        # U index = i * m + j = i * 3 + j
        
        # M2 = (a10+a11)b00
        U[3][col] += 1.0 # a10
        U[4][col] += 1.0 # a11
        V[k0][col] += 1.0 # b00 (j=0, k=k0)
        # W: C10 += M2
        # C10 is i=1, k=k0. W index = k0*3 + 1
        W[k0*n + 1][col] += 1.0
        col += 1
        
        # Term 2: M3 = a00(b10-b11)
        U[0][col] += 1.0
        V[8+k0][col] += 1.0 # b10 (j=1, k=k0)
        V[8+k1][col] -= 1.0 # b11 (j=1, k=k1)
        # W: C01 += M3
        # C01 is i=0, k=k1. W index = k1*3 + 0
        W[k1*n + 0][col] += 1.0
        col += 1
        
        # Term 3: M4 = a11(b01-b00)
        U[4][col] += 1.0
        V[k1][col] += 1.0 # b01 (j=0, k=k1)
        V[k0][col] -= 1.0 # b00 (j=0, k=k0)
        # W: C00 += M4, C11 += M4
        W[k0*n + 0][col] += 1.0
        W[k1*n + 1][col] += 1.0
        col += 1
        
        # Term 4: M5 = (a00+a01)b11
        U[0][col] += 1.0
        U[1][col] += 1.0 # a01
        V[8+k1][col] += 1.0 # b11
        # W: C00 -= M5, C01 += M5
        W[k0*n + 0][col] -= 1.0
        W[k1*n + 0][col] += 1.0
        col += 1
        
        # Term 5: M6 = (a10-a00)(b00+b01)
        U[3][col] += 1.0
        U[0][col] -= 1.0
        V[k0][col] += 1.0
        V[k1][col] += 1.0
        # W: C11 += M6
        W[k1*n + 1][col] += 1.0
        col += 1
        
        # Term 6: M7 = (a01-a11)(b10+b11)
        U[1][col] += 1.0
        U[4][col] -= 1.0
        V[8+k0][col] += 1.0
        V[8+k1][col] += 1.0
        # W: C00 += M7
        W[k0*n + 0][col] += 1.0
        col += 1
        
        # Term 7: m8 = a02*b20
        U[2][col] += 1.0 # a02
        V[16+k0][col] += 1.0 # b20 (j=2, k=k0) -> 2*8+k0
        # W: C00 += m8
        W[k0*n + 0][col] += 1.0
        col += 1
        
        # Term 8: m9 = a02*b21
        U[2][col] += 1.0
        V[16+k1][col] += 1.0
        # W: C01 += m9
        W[k1*n + 0][col] += 1.0
        col += 1
        
        # Term 9: m10 = a12*b20
        U[5][col] += 1.0 # a12
        V[16+k0][col] += 1.0
        # W: C10 += m10
        W[k0*n + 1][col] += 1.0
        col += 1
        
        # Term 10: m11 = a12*b21
        U[5][col] += 1.0
        V[16+k1][col] += 1.0
        # W: C11 += m11
        W[k1*n + 1][col] += 1.0
        col += 1
        
        # Term 11: m12 = a20*b00
        U[6][col] += 1.0 # a20
        V[k0][col] += 1.0
        # W: C20 += m12
        W[k0*n + 2][col] += 1.0
        col += 1
        
        # Term 12: m13 = a20*b01
        U[6][col] += 1.0
        V[k1][col] += 1.0
        # W: C21 += m13
        W[k1*n + 2][col] += 1.0
        col += 1
        
        # Term 13: m14 = a21*b10
        U[7][col] += 1.0 # a21
        V[8+k0][col] += 1.0
        # W: C20 += m14
        W[k0*n + 2][col] += 1.0
        col += 1
        
        # Term 14: m15 = a21*b11
        U[7][col] += 1.0
        V[8+k1][col] += 1.0
        # W: C21 += m15
        W[k1*n + 2][col] += 1.0
        col += 1
        
        # Term 15: m16 = a22*b20
        U[8][col] += 1.0 # a22
        V[16+k0][col] += 1.0
        # W: C20 += m16
        W[k0*n + 2][col] += 1.0
        col += 1
        
        # Term 16: m17 = a22*b21
        U[8][col] += 1.0
        V[16+k1][col] += 1.0
        # W: C21 += m17
        W[k1*n + 2][col] += 1.0
        col += 1
        
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
