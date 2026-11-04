import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for 2x2x2 blocks
    # 5 blocks, each rank 7
    for r in range(5):
        # V indices for this block
        v_00 = 0 * p + 2 * r
        v_01 = 0 * p + 2 * r + 1
        v_10 = 1 * p + 2 * r
        v_11 = 1 * p + 2 * r + 1
        
        # W indices for this block
        w_00 = 2 * r * n + 0
        w_01 = (2 * r + 1) * n + 0
        w_10 = 2 * r * n + 1
        w_11 = (2 * r + 1) * n + 1
        
        # U indices (same for all blocks)
        u_00 = 0 * m + 0
        u_01 = 0 * m + 1
        u_10 = 1 * m + 0
        u_11 = 1 * m + 1
        
        for s in range(7):
            c = 7 * r + s
            if s == 0:
                # M1 = (x00 + x11)(y00 + y11)
                U[u_00][c] = 1
                U[u_11][c] = 1
                V[v_00][c] = 1
                V[v_11][c] = 1
                W[w_00][c] = 1
                W[w_11][c] = 1
            elif s == 1:
                # M2 = (x10 + x11)y00
                U[u_10][c] = 1
                U[u_11][c] = 1
                V[v_00][c] = 1
                W[w_10][c] = 1
                W[w_11][c] = -1
            elif s == 2:
                # M3 = x00(y01 - y11)
                U[u_00][c] = 1
                V[v_01][c] = 1
                V[v_11][c] = -1
                W[w_01][c] = 1
                W[w_11][c] = 1
            elif s == 3:
                # M4 = x11(y10 - y00)
                U[u_11][c] = 1
                V[v_10][c] = 1
                V[v_00][c] = -1
                W[w_00][c] = 1
                W[w_10][c] = 1
            elif s == 4:
                # M5 = (x00 + x01)y11
                U[u_00][c] = 1
                U[u_01][c] = 1
                V[v_11][c] = 1
                W[w_00][c] = -1
                W[w_01][c] = 1
            elif s == 5:
                # M6 = (x10 - x00)(y00 + y01)
                U[u_10][c] = 1
                U[u_00][c] = -1
                V[v_00][c] = 1
                V[v_01][c] = 1
                W[w_11][c] = 1
            elif s == 6:
                # M7 = (x01 - x11)(y10 + y11)
                U[u_01][c] = 1
                U[u_11][c] = -1
                V[v_10][c] = 1
                V[v_11][c] = 1
                W[w_00][c] = 1
                W[w_11][c] = 1

    # Schoolbook part for 2x1x10
    # i in {0, 1}, j = 2, k in {0, ..., 9}
    for i in range(2):
        for k in range(10):
            c = 35 + i * 10 + k
            U[i * m + 2][c] = 1
            V[2 * p + k][c] = 1
            W[k * n + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
