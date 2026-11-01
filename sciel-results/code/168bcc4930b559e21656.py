import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    rank = 7 * (p // 2)
    
    nU = n * m
    nV = m * p
    nW = p * n

    U = [[0] * rank for _ in range(nU)]
    V = [[0] * rank for _ in range(nV)]
    W = [[0] * rank for _ in range(nW)]

    # Strassen's algorithm terms for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a10)(b10 + b11)
    #
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M6
    # c11 = M1 - M2 + M3 + M7
    
    # Indices for A (n*m = 4): 0: a00, 1: a01, 2: a10, 3: a11
    # Indices for B block (2x2): 0: b00, 1: b01, 2: b10, 3: b11
    # Indices for C block (2x2) in W (k*n+i):
    # W index for C[i,k] is k*n + i.
    # For block k:
    # C[0, 2k] -> W[2k*n + 0] -> c00
    # C[1, 2k] -> W[2k*n + 1] -> c10
    # C[0, 2k+1] -> W[(2k+1)*n + 0] -> c01
    # C[1, 2k+1] -> W[(2k+1)*n + 1] -> c11
    
    # So c_vec order is [c00, c10, c01, c11]
    
    terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0]),
        # M7
        ([0, 1, -1, 0], [0, 0, 1, 1], [-1, 0, 0, 1]),
    ]
    
    col = 0
    for block in range(p // 2):
        for a_vec, b_vec, c_vec in terms:
            for idx in range(nU):
                U[idx][col] = a_vec[idx]
            
            V[0 * p + 2 * block][col] = b_vec[0]
            V[0 * p + 2 * block + 1][col] = b_vec[1]
            V[1 * p + 2 * block][col] = b_vec[2]
            V[1 * p + 2 * block + 1][col] = b_vec[3]
            
            W[2 * block * n + 0][col] = c_vec[0]
            W[2 * block * n + 1][col] = c_vec[1]
            W[(2 * block + 1) * n + 0][col] = c_vec[2]
            W[(2 * block + 1) * n + 1][col] = c_vec[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
