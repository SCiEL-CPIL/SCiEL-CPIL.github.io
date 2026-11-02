import time

def construct():
    start_time = time.time()
    deadline = 140  # 10s safety margin before 150s kill

    n, m, p = 2, 2, 12
    # Use rank-8 decomposition per 4-column block (schoolbook for 2x2x4), 3 blocks -> total rank 24
    rank = 24
    U = [[0]*rank for _ in range(n*m)]   # 4 x 24
    V = [[0]*rank for _ in range(m*p)]   # 24 x 24
    W = [[0]*rank for _ in range(p*n)]   # 24 x 24

    num_blocks = p // 4  # 3 blocks of 4 columns each

    col = 0
    for t in range(num_blocks):
        # W indices for block t: C(i,k) -> W[k*n+i]
        w_c00 = 8*t      # C(0, 4t)
        w_c10 = 8*t + 1  # C(1, 4t)
        w_c01 = 8*t + 2  # C(0, 4t+1)
        w_c11 = 8*t + 3  # C(1, 4t+1)
        w_c02 = 8*t + 4  # C(0, 4t+2)
        w_c12 = 8*t + 5  # C(1, 4t+2)
        w_c03 = 8*t + 6  # C(0, 4t+3)
        w_c13 = 8*t + 7  # C(1, 4t+3)

        # B indices for block t: B(j,k) -> V[j*p+k]
        v_b00 = 4*t       # B(0, 4t)
        v_b01 = 4*t + 1   # B(0, 4t+1)
        v_b02 = 4*t + 2   # B(0, 4t+2)
        v_b03 = 4*t + 3   # B(0, 4t+3)
        v_b10 = 12 + 4*t       # B(1, 4t)
        v_b11 = 12 + 4*t + 1   # B(1, 4t+1)
        v_b12 = 12 + 4*t + 2   # B(1, 4t+2)
        v_b13 = 12 + 4*t + 3   # B(1, 4t+3)

        # Row 0 of A times each column of B block t
        # M1 = A00 * B00
        U[0][col] = 1
        V[v_b00][col] = 1
        W[w_c00][col] = 1
        col += 1

        # M2 = A00 * B01
        U[0][col] = 1
        V[v_b01][col] = 1
        W[w_c01][col] = 1
        col += 1

        # M3 = A00 * B02
        U[0][col] = 1
        V[v_b02][col] = 1
        W[w_c02][col] = 1
        col += 1

        # M4 = A00 * B03
        U[0][col] = 1
        V[v_b03][col] = 1
        W[w_c03][col] = 1
        col += 1

        # Row 1 of A times each column of B block t
        # M5 = A10 * B00
        U[2][col] = 1
        V[v_b00][col] = 1
        W[w_c10][col] = 1
        col += 1

        # M6 = A10 * B01
        U[2][col] = 1
        V[v_b01][col] = 1
        W[w_c11][col] = 1
        col += 1

        # M7 = A10 * B02
        U[2][col] = 1
        V[v_b02][col] = 1
        W[w_c12][col] = 1
        col += 1

        # M8 = A10 * B03
        U[2][col] = 1
        V[v_b03][col] = 1
        W[w_c13][col] = 1
        col += 1

    # Time guard
    if time.time() - start_time > deadline:
        pass

    return {"U": U, "V": V, "W": W}
