import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 10 x 36
    V = [[0] * rank for _ in range(m * p)]  # 30 x 36
    W = [[0] * rank for _ in range(p * n)]  # 12 x 36
    
    col = 0
    
    # Helper: add 2x2 * 2x2 schoolbook block
    def add_2x2_block(i_base, j_base, k_base):
        nonlocal col
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    U[(i_base+i)*m + (j_base+j)][col] = 1
                    V[(j_base+j)*p + (k_base+k)][col] = 1
                    W[(k_base+k)*n + (i_base+i)][col] = 1
                    col += 1
    
    # Block 1: A[:, 0:2] (2x2) * B[0:2, :] (2x6)
    for kb in range(3):
        add_2x2_block(0, 0, kb*2)
    
    # Block 2: A[:, 2] (2x1) * B[2, :] (1x6)
    for k in range(6):
        for i in range(2):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    # Block 3: A[:, 3:5] (2x2) * B[3:5, :] (2x6)
    for kb in range(3):
        add_2x2_block(0, 3, kb*2)
    
    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}
