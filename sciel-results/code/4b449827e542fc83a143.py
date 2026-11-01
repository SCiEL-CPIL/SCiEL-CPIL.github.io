import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, safe margin
    
    n, m, p = 2, 2, 14
    rank = 49
    
    # Strassen's 7 linear forms for 2x2 matrix multiplication
    A_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    
    B_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    
    C_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(7):
        if time.time() > deadline:
            break
        for s in range(7):
            # U: A coefficients
            for i in range(4):
                U[i][col] = A_coeffs[s][i]
            
            # V: B coefficients
            for r in range(4):
                coeff = B_coeffs[s][r]
                if coeff != 0:
                    if r == 0:
                        V[2 * b][col] += coeff
                    elif r == 1:
                        V[2 * b + 1][col] += coeff
                    elif r == 2:
                        V[14 + 2 * b][col] += coeff
                    elif r == 3:
                        V[15 + 2 * b][col] += coeff
            
            # W: C coefficients
            for r in range(4):
                coeff = C_coeffs[s][r]
                if coeff != 0:
                    if r == 0:
                        W[4 * b][col] += coeff
                    elif r == 1:
                        W[4 * b + 2][col] += coeff
                    elif r == 2:
                        W[4 * b + 1][col] += coeff
                    elif r == 3:
                        W[4 * b + 3][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
