import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    
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

    # --- Strassen Decomposition for C[:, 0:2] ---
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Recombination for C_left:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # r=0: M1
    add_u(0, 0, 0, 1); add_u(1, 1, 0, 1)
    add_v(0, 0, 0, 1); add_v(1, 1, 0, 1)
    add_w(0, 0, 0, 1)
    add_w(1, 1, 0, 1)

    # r=1: M2
    add_u(1, 0, 1, 1); add_u(1, 1, 1, 1)
    add_v(0, 0, 1, 1)
    add_w(0, 1, 1, 1)
    add_w(1, 1, 1, -1)

    # r=2: M3
    add_u(0, 0, 2, 1)
    add_v(0, 1, 2, 1); add_v(1, 1, 2, -1)
    add_w(1, 0, 2, 1)
    add_w(1, 1, 2, 1)

    # r=3: M4
    add_u(1, 1, 3, 1)
    add_v(1, 0, 3, 1); add_v(0, 0, 3, -1)
    add_w(0, 0, 3, 1)
    add_w(0, 1, 3, 1)

    # r=4: M5
    add_u(0, 0, 4, 1); add_u(0, 1, 4, 1)
    add_v(1, 1, 4, 1)
    add_w(0, 0, 4, -1)
    add_w(1, 0, 4, 1)

    # r=5: M6
    add_u(1, 0, 5, 1); add_u(0, 0, 5, -1)
    add_v(0, 0, 5, 1); add_v(0, 1, 5, 1)
    add_w(1, 1, 5, 1)

    # r=6: M7
    add_u(0, 1, 6, 1); add_u(1, 1, 6, -1)
    add_v(1, 0, 6, 1); add_v(1, 1, 6, 1)
    add_w(0, 0, 6, 1)

    # --- Standard Multiplication for C[:, 2] ---
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # r=7: A00 * B02
    add_u(0, 0, 7, 1)
    add_v(0, 2, 7, 1)
    add_w(2, 0, 7, 1)

    # r=8: A01 * B12
    add_u(0, 1, 8, 1)
    add_v(1, 2, 8, 1)
    add_w(2, 0, 8, 1)

    # r=9: A10 * B02
    add_u(1, 0, 9, 1)
    add_v(0, 2, 9, 1)
    add_w(2, 1, 9, 1)

    # r=10: A11 * B12
    add_u(1, 1, 10, 1)
    add_v(1, 2, 10, 1)
    add_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
