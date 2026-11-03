import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    X_blocks = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1}
    ]
    Y_blocks = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1}
    ]
    C_coeffs = [
        {(0,0):1, (0,1):1, (1,0):1, (1,1):1},
        {(1,0):1, (1,1):-1},
        {(0,1):1, (1,1):1},
        {(0,0):1, (1,0):1},
        {(0,0):-1, (0,1):1},
        {(1,1):1},
        {(0,0):1, (1,1):1}
    ]
    
    col = 0
    for s in range(7):
        for a in range(3):
            for t in range(3):
                for b in range(3):
                    if time.time() > deadline:
                        break
                    for (br, bc), coeff in X_blocks[s].items():
                        i = br * 3 + a
                        j = bc * 3 + t
                        U[i * 6 + j][col] = coeff
                    
                    for (br, bc), coeff in Y_blocks[s].items():
                        j = br * 3 + t
                        k = bc * 3 + b
                        V[j * 6 + k][col] = coeff
                    
                    for (br, bc), coeff in C_coeffs[s].items():
                        k = br * 3 + b
                        i = bc * 3 + a
                        W[k * 6 + i][col] = coeff
                    
                    col += 1
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
