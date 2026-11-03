import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # l corresponds to the block index of C (0..3)
    # C columns 2l, 2l+1
    for l in range(4):
        k_base = 2 * l
        
        # --- A_L * B_top_l ---
        # A_L: i in {0,1}, j in {0,1}
        # B_top_l: j in {0,1}, k in {2l, 2l+1}
        
        # M1: (a00 + a11) * (b00 + b11)
        # a00 -> U[0*4+0], a11 -> U[1*4+1]
        # b00 -> V[0*8+2l], b11 -> V[1*8+2l+1]
        # W: C00 (i=0, k=2l) += 1, C11 (i=1, k=2l+1) += 1
        U[0*4+0][col] = 1
        U[1*4+1][col] = 1
        V[0*8+k_base][col] = 1
        V[1*8+k_base+1][col] = 1
        W[k_base*2+0][col] = 1
        W[(k_base+1)*2+1][col] = 1
        col += 1
        
        # M2: (a10 + a11) * b00
        # a10 -> U[1*4+0], a11 -> U[1*4+1]
        # b00 -> V[0*8+2l]
        # W: C10 (i=1, k=2l) += 1
        U[1*4+0][col] = 1
        U[1*4+1][col] = 1
        V[0*8+k_base][col] = 1
        W[k_base*2+1][col] = 1
        col += 1
        
        # M3: a00 * (b01 - b11)
        # a00 -> U[0*4+0]
        # b01 -> V[0*8+2l+1], b11 -> V[1*8+2l+1]
        # W: C01 (i=0, k=2l+1) += 1
        U[0*4+0][col] = 1
        V[0*8+k_base+1][col] = 1
        V[1*8+k_base+1][col] = -1
        W[(k_base+1)*2+0][col] = 1
        col += 1
        
        # M4: a11 * (b10 - b00)
        # a11 -> U[1*4+1]
        # b10 -> V[1*8+2l], b00 -> V[0*8+2l]
        # W: C00 (i=0, k=2l) += 1, C10 (i=1, k=2l) += 1
        U[1*4+1][col] = 1
        V[1*8+k_base][col] = 1
        V[0*8+k_base][col] = -1
        W[k_base*2+0][col] = 1
        W[k_base*2+1][col] = 1
        col += 1
        
        # M5: (a00 + a01) * b11
        # a00 -> U[0*4+0], a01 -> U[0*4+1]
        # b11 -> V[1*8+2l+1]
        # W: C00 (i=0, k=2l) -= 1, C01 (i=0, k=2l+1) += 1
        U[0*4+0][col] = 1
        U[0*4+1][col] = 1
        V[1*8+k_base+1][col] = 1
        W[k_base*2+0][col] = -1
        W[(k_base+1)*2+0][col] = 1
        col += 1
        
        # M6: (a10 - a00) * (b00 + b01)
        # a10 -> U[1*4+0], a00 -> U[0*4+0]
        # b00 -> V[0*8+2l], b01 -> V[0*8+2l+1]
        # W: C11 (i=1, k=2l+1) += 1
        U[1*4+0][col] = 1
        U[0*4+0][col] = -1
        V[0*8+k_base][col] = 1
        V[0*8+k_base+1][col] = 1
        W[(k_base+1)*2+1][col] = 1
        col += 1
        
        # M7: (a01 - a11) * (b10 + b11)
        # a01 -> U[0*4+1], a11 -> U[1*4+1]
        # b10 -> V[1*8+2l], b11 -> V[1*8+2l+1]
        # W: C00 (i=0, k=2l) += 1
        U[0*4+1][col] = 1
        U[1*4+1][col] = -1
        V[1*8+k_base][col] = 1
        V[1*8+k_base+1][col] = 1
        W[k_base*2+0][col] = 1
        col += 1
        
        # --- A_R * B_bot_l ---
        # A_R: i in {0,1}, j in {2,3}
        # B_bot_l: j in {2,3}, k in {2l, 2l+1}
        
        # M1: (a00 + a11) * (b00 + b11)
        # a00 -> A[0,2], a11 -> A[1,3]
        # b00 -> B[2,2l], b11 -> B[3,2l+1]
        U[0*4+2][col] = 1
        U[1*4+3][col] = 1
        V[2*8+k_base][col] = 1
        V[3*8+k_base+1][col] = 1
        W[k_base*2+0][col] = 1
        W[(k_base+1)*2+1][col] = 1
        col += 1
        
        # M2: (a10 + a11) * b00
        # a10 -> A[1,2], a11 -> A[1,3]
        # b00 -> B[2,2l]
        U[1*4+2][col] = 1
        U[1*4+3][col] = 1
        V[2*8+k_base][col] = 1
        W[k_base*2+1][col] = 1
        col += 1
        
        # M3: a00 * (b01 - b11)
        # a00 -> A[0,2]
        # b01 -> B[2,2l+1], b11 -> B[3,2l+1]
        U[0*4+2][col] = 1
        V[2*8+k_base+1][col] = 1
        V[3*8+k_base+1][col] = -1
        W[(k_base+1)*2+0][col] = 1
        col += 1
        
        # M4: a11 * (b10 - b00)
        # a11 -> A[1,3]
        # b10 -> B[3,2l], b00 -> B[2,2l]
        U[1*4+3][col] = 1
        V[3*8+k_base][col] = 1
        V[2*8+k_base][col] = -1
        W[k_base*2+0][col] = 1
        W[k_base*2+1][col] = 1
        col += 1
        
        # M5: (a00 + a01) * b11
        # a00 -> A[0,2], a01 -> A[0,3]
        # b11 -> B[3,2l+1]
        U[0*4+2][col] = 1
        U[0*4+3][col] = 1
        V[3*8+k_base+1][col] = 1
        W[k_base*2+0][col] = -1
        W[(k_base+1)*2+0][col] = 1
        col += 1
        
        # M6: (a10 - a00) * (b00 + b01)
        # a10 -> A[1,2], a00 -> A[0,2]
        # b00 -> B[2,2l], b01 -> B[2,2l+1]
        U[1*4+2][col] = 1
        U[0*4+2][col] = -1
        V[2*8+k_base][col] = 1
        V[2*8+k_base+1][col] = 1
        W[(k_base+1)*2+1][col] = 1
        col += 1
        
        # M7: (a01 - a11) * (b10 + b11)
        # a01 -> A[0,3], a11 -> A[1,3]
        # b10 -> B[3,2l], b11 -> B[3,2l+1]
        U[0*4+3][col] = 1
        U[1*4+3][col] = -1
        V[3*8+k_base][col] = 1
        V[3*8+k_base+1][col] = 1
        W[k_base*2+0][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
