def add_mult(U, V, W, u_idx, u_val, v_idx, v_val, w_idx, w_val):
    U[u_idx] += u_val
    V[v_idx] += v_val
    W[w_idx] += w_val

def construct():
    n, m, p = 2, 4, 7
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: A1 (cols 0,1), B1 (rows 0,1)
    # A indices: 0,1,4,5
    # B indices: 0..6, 7..13
    # W indices: same as block 2
    
    a00, a01, a10, a11 = 0, 1, 4, 5
    
    for alpha in range(3):
        col0 = 2 * alpha
        col1 = 2 * alpha + 1
        # B indices
        b00 = col0
        b01 = col1
        b10 = 7 + col0
        b11 = 7 + col1
        
        # W indices
        w00 = 2 * col0
        w01 = 2 * col1
        w10 = 2 * col0 + 1
        w11 = 2 * col1 + 1
        
        # 1. (a00 + a11)(b00 + b11)
        U[a00][0] += 1; U[a11][0] += 1
        V[b00][0] += 1; V[b11][0] += 1
        W[w00][0] += 1; W[w01][0] += 1; W[w10][0] += 1; W[w11][0] -= 1
        
        # 2. (a10 + a11)b00
        U[a10][1] += 1; U[a11][1] += 1
        V[b00][1] += 1
        W[w10][1] += 1; W[w11][1] -= 1
        
        # 3. a00(b10 - b11)
        U[a00][2] += 1
        V[b10][2] += 1; V[b11][2] -= 1
        W[w01][2] += 1; W[w11][2] += 1
        
        # 4. a11(b01 - b00)
        U[a11][3] += 1
        V[b01][3] += 1; V[b00][3] -= 1
        W[w00][3] += 1; W[w10][3] += 1
        
        # 5. (a00 + a01)b11
        U[a00][4] += 1; U[a01][4] += 1
        V[b11][4] += 1
        W[w00][4] -= 1; W[w01][4] += 1
        
        # 6. (a10 - a00)(b00 + b01)
        U[a10][5] += 1; U[a00][5] -= 1
        V[b00][5] += 1; V[b01][5] += 1
        W[w11][5] += 1
        
        # 7. (a01 - a11)(b10 + b11)
        U[a01][6] += 1; U[a11][6] -= 1
        V[b10][6] += 1; V[b11][6] += 1
        W[w01][6] += 1

    # Last column of Block 1 (col 6)
    col = 6
    b0 = col
    b1 = 7 + col
    w0 = 2 * col
    w1 = 2 * col + 1
    
    # 8. a00 b0
    U[a00][7] += 1
    V[b0][7] += 1
    W[w0][7] += 1
    
    # 9. a01 b1
    U[a01][8] += 1
    V[b1][8] += 1
    W[w0][8] += 1
    
    # 10. a10 b0
    U[a10][9] += 1
    V[b0][9] += 1
    W[w1][9] += 1
    
    # 11. a11 b1
    U[a11][10] += 1
    V[b1][10] += 1
    W[w1][10] += 1

    # Block 2: A2 (cols 2,3), B2 (rows 2,3)
    a00, a01, a10, a11 = 2, 3, 6, 7
    
    for alpha in range(3):
        col0 = 2 * alpha
        col1 = 2 * alpha + 1
        # B indices
        b00 = 14 + col0
        b01 = 14 + col1
        b10 = 21 + col0
        b11 = 21 + col1
        
        # W indices
        w00 = 2 * col0
        w01 = 2 * col1
        w10 = 2 * col0 + 1
        w11 = 2 * col1 + 1
        
        r = 11 + alpha * 7
        
        # 1. (a00 + a11)(b00 + b11)
        U[a00][r] += 1; U[a11][r] += 1
        V[b00][r] += 1; V[b11][r] += 1
        W[w00][r] += 1; W[w01][r] += 1; W[w10][r] += 1; W[w11][r] -= 1
        
        # 2. (a10 + a11)b00
        U[a10][r+1] += 1; U[a11][r+1] += 1
        V[b00][r+1] += 1
        W[w10][r+1] += 1; W[w11][r+1] -= 1
        
        # 3. a00(b10 - b11)
        U[a00][r+2] += 1
        V[b10][r+2] += 1; V[b11][r+2] -= 1
        W[w01][r+2] += 1; W[w11][r+2] += 1
        
        # 4. a11(b01 - b00)
        U[a11][r+3] += 1
        V[b01][r+3] += 1; V[b00][r+3] -= 1
        W[w00][r+3] += 1; W[w10][r+3] += 1
        
        # 5. (a00 + a01)b11
        U[a00][r+4] += 1; U[a01][r+4] += 1
        V[b11][r+4] += 1
        W[w00][r+4] -= 1; W[w01][r+4] += 1
        
        # 6. (a10 - a00)(b00 + b01)
        U[a10][r+5] += 1; U[a00][r+5] -= 1
        V[b00][r+5] += 1; V[b01][r+5] += 1
        W[w11][r+5] += 1
        
        # 7. (a01 - a11)(b10 + b11)
        U[a01][r+6] += 1; U[a11][r+6] -= 1
        V[b10][r+6] += 1; V[b11][r+6] += 1
        W[w01][r+6] += 1

    # Last column of Block 2 (col 6)
    col = 6
    b0 = 14 + col
    b1 = 21 + col
    w0 = 2 * col
    w1 = 2 * col + 1
    
    r = 32
    
    # 8. a00 b0
    U[a00][r] += 1
    V[b0][r] += 1
    W[w0][r] += 1
    
    # 9. a01 b1
    U[a01][r+1] += 1
    V[b1][r+1] += 1
    W[w0][r+1] += 1
    
    # 10. a10 b0
    U[a10][r+2] += 1
    V[b0][r+2] += 1
    W[w1][r+2] += 1
    
    # 11. a11 b1
    U[a11][r+3] += 1
    V[b1][r+3] += 1
    W[w1][r+3] += 1

    return {"U": U, "V": V, "W": W}
