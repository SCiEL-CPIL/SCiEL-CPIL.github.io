import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]   # (16, 44)
    V = [[0] * rank for _ in range(m * p)]   # (20, 44)
    W = [[0] * rank for _ in range(p * n)]   # (20, 44)
    
    col = 0
    
    def add_2x2x2(a_blocks, b_blocks, c_blocks):
        nonlocal col
        for j in range(2):
            u_sparse = []
            for (ar, ac, a_sign) in a_blocks:
                for i in range(2):
                    u_sparse.append(((ar*2+i)*4 + (ac*2+j), a_sign))
            v_sparse = []
            for (br, bc, b_sign) in b_blocks:
                for k in range(2):
                    v_sparse.append(((br*2+j)*5 + (bc*2+k), b_sign))
            w_sparse = []
            for (cr, cc, coeff) in c_blocks:
                for i in range(2):
                    for k in range(2):
                        w_sparse.append(((cc*2+k)*4 + (cr*2+i), coeff))
            for idx, coeff in u_sparse:
                U[idx][col] += coeff
            for idx, coeff in v_sparse:
                V[idx][col] += coeff
            for idx, coeff in w_sparse:
                W[idx][col] += coeff
            col += 1
    
    # M1 = (A00+A11)(B00+B11) -> C00 (+1), C11 (+1)
    add_2x2x2([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
    
    # M2 = (A10+A11)B00 -> C10 (+1), C11 (-1)
    add_2x2x2([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
    
    # M3 = A00(B01-B11) -> C01 (+1), C11 (+1)
    add_2x2x2([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
    
    # M4 = A11(B10-B00) -> C00 (+1), C10 (+1)
    add_2x2x2([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
    
    # M5 = (A00+A01)B11 -> C00 (-1), C01 (+1)
    add_2x2x2([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
    
    # M6 = (A10-A00)(B00+B01) -> C11 (+1)
    add_2x2x2([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
    
    # M7 = (A01-A11)(B10+B11) -> C00 (+1)
    add_2x2x2([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    
    # Part 2: 4x4x1 schoolbook (rank 16)
    if time.time() < deadline:
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][col] = 1
                V[j * 5 + 4][col] = 1
                W[4 * 4 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
