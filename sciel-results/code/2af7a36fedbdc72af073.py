import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    rank = 7
    
    nU = n * m  # 4
    nV = m * p  # 20
    nW = p * n  # 20
    
    # Initialize matrices
    U = [[0] * rank for _ in range(nU)]
    V = [[0] * rank for _ in range(nV)]
    W = [[0] * rank for _ in range(nW)]
    
    # Strassen's algorithm terms for 2x2 matrix multiplication
    # U_vec: [a00, a01, a10, a11]
    # V_vec: [b00, b01, b10, b11]
    # W_vec: [c00, c10, c01, c11]
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Fill U
    for r in range(rank):
        for idx in range(nU):
            U[idx][r] = terms[r][0][idx]
            
    # Fill V and W by repeating the pattern for each block
    for block in range(p // 2):
        for r in range(rank):
            b_vec = terms[r][1]
            c_vec = terms[r][2]
            
            # V indices for block b
            # Row 0, Col 2b
            V[0 * p + 2 * block][r] = b_vec[0]
            # Row 0, Col 2b+1
            V[0 * p + 2 * block + 1][r] = b_vec[1]
            # Row 1, Col 2b
            V[1 * p + 2 * block][r] = b_vec[2]
            # Row 1, Col 2b+1
            V[1 * p + 2 * block + 1][r] = b_vec[3]
            
            # W indices for block b
            # Col 2b, Row 0
            W[2 * block * n + 0][r] = c_vec[0]
            # Col 2b, Row 1
            W[2 * block * n + 1][r] = c_vec[1]
            # Col 2b+1, Row 0
            W[(2 * block + 1) * n + 0][r] = c_vec[2]
            # Col 2b+1, Row 1
            W[(2 * block + 1) * n + 1][r] = c_vec[3]
            
    return {"U": U, "V": V, "W": W}
