import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    rank = 11
    
    # Initialize factor matrices
    # U: (n*m, rank) = (4, 11)
    # V: (m*p, rank) = (6, 11)
    # W: (p*n, rank) = (6, 11)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_u(i, j, r, val):
        idx = i * m + j
        U[idx][r] += val
        
    def add_v(j, k, r, val):
        idx = j * p + k
        V[idx][r] += val
        
    def add_w(k, i, r, val):
        idx = k * n + i
        W[idx][r] += val

    # Strassen-based decomposition for C[:, 0:2] plus standard mult for C[:, 2]
    # Total rank: 7 (Strassen) + 4 (standard) = 11, which is optimal for 2x2x3
    
    # r=0: M1 = (A00+A11)(B00+B11)
    add_u(0, 0, 0, 1); add_u(1, 1, 0, 1)
    add_v(0, 0, 0, 1); add_v(1, 1, 0, 1)
    add_w(0, 0, 0, 1)   # C00 += M1
    add_w(1, 1, 0, 1)   # C11 += M1

    # r=1: M2 = (A10+A11)B00
    add_u(1, 0, 1, 1); add_u(1, 1, 1, 1)
    add_v(0, 0, 1, 1)
    add_w(0, 1, 1, 1)   # C10 += M2
    add_w(1, 1, 1, -1)  # C11 -= M2

    # r=2: M3 = A00(B01-B11)
    add_u(0, 0, 2, 1)
    add_v(0, 1, 2, 1); add_v(1, 1, 2, -1)
    add_w(1, 0, 2, 1)   # C01 += M3
    add_w(1, 1, 2, 1)   # C11 += M3

    # r=3: M4 = A11(B10-B00)
    add_u(1, 1, 3, 1)
    add_v(1, 0, 3, 1); add_v(0, 0, 3, -1)
    add_w(0, 0, 3, 1)   # C00 += M4
    add_w(0, 1, 3, 1)   # C10 += M4

    # r=4: M5 = (A00+A01)B11
    add_u(0, 0, 4, 1); add_u(0, 1, 4, 1)
    add_v(1, 1, 4, 1)
    add_w(0, 0, 4, -1)  # C00 -= M5
    add_w(1, 0, 4, 1)   # C01 += M5

    # r=5: M6 = (A10-A00)(B00+B01)
    add_u(1, 0, 5, 1); add_u(0, 0, 5, -1)
    add_v(0, 0, 5, 1); add_v(0, 1, 5, 1)
    add_w(1, 1, 5, 1)   # C11 += M6

    # r=6: M7 = (A01-A11)(B10+B11)
    add_u(0, 1, 6, 1); add_u(1, 1, 6, -1)
    add_v(1, 0, 6, 1); add_v(1, 1, 6, 1)
    add_w(0, 0, 6, 1)   # C00 += M7

    # r=7: A00 * B02
    add_u(0, 0, 7, 1)
    add_v(0, 2, 7, 1)
    add_w(2, 0, 7, 1)   # C02 += A00*B02

    # r=8: A01 * B12
    add_u(0, 1, 8, 1)
    add_v(1, 2, 8, 1)
    add_w(2, 0, 8, 1)   # C02 += A01*B12

    # r=9: A10 * B02
    add_u(1, 0, 9, 1)
    add_v(0, 2, 9, 1)
    add_w(2, 1, 9, 1)   # C12 += A10*B02

    # r=10: A11 * B12
    add_u(1, 1, 10, 1)
    add_v(1, 2, 10, 1)
    add_w(2, 1, 10, 1)  # C12 += A11*B12

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
