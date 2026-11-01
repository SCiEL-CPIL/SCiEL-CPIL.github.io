import time

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U: 4x7 (A00, A01, A10, A11)
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ]
    
    # V: 4x7 (B00, B01, B10, B11)
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ]
    
    # W: 4x7 (C00, C01, C10, C11)
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0,  1, 0, 0],
        [0, 1, 0, 1,  0, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ]
    
    # Partition p=7 into 2+2+2+1
    # 3 Strassen blocks (rank 7 each) + 1 Schoolbook block (rank 4)
    # Total rank = 25
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 3 Strassen blocks
    for block in range(3):
        k_base = block * 2
        for r in range(7):
            # U
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = U_s[i * m + j][r]
            # V
            for j in range(m):
                for kk in range(2):
                    V[j * p + (k_base + kk)][col] = V_s[j * 2 + kk][r]
            # W
            for i in range(n):
                for kk in range(2):
                    W[(k_base + kk) * n + i][col] = W_s[i * 2 + kk][r]
            col += 1
            
    # 1 Schoolbook block for k=6
    for k in range(6, p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i * m + j][r] * V[jp * p + kp][r] * W[kp * n + ip][r]
                            if abs(lhs - rhs) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
        if not valid: break
        
    if valid:
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook rank 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
