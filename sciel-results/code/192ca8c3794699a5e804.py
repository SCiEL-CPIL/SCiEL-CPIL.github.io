import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_list, v_list, w_list):
        nonlocal col
        if col >= rank:
            return
        for (i, j), c in u_list:
            U[i * m + j][col] += c
        for (j, k), c in v_list:
            V[j * p + k][col] += c
        for (k, i), c in w_list:
            W[k * n + i][col] += c
        col += 1

    for k_off in [0, 3]:
        # 1. Strassen 2x2x2
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term(
            [((0,0),1), ((1,1),1)],
            [((0,0),1), ((1,1),1)],
            [((0+k_off,0),1), ((1+k_off,1),1)]
        )
        # M2: (A10+A11)B00 -> C10, -C11
        add_term(
            [((1,0),1), ((1,1),1)],
            [((0,0),1)],
            [((0+k_off,1),1), ((1+k_off,1),-1)]
        )
        # M3: A00(B10-B11) -> C01, C11
        add_term(
            [((0,0),1)],
            [((1,0),1), ((1,1),-1)],
            [((1+k_off,0),1), ((1+k_off,1),1)]
        )
        # M4: A11(B01-B00) -> C00, C10
        add_term(
            [((1,1),1)],
            [((0,1),1), ((0,0),-1)],
            [((0+k_off,0),1), ((0+k_off,1),1)]
        )
        # M5: (A00+A01)B11 -> C01, -C00
        add_term(
            [((0,0),1), ((0,1),1)],
            [((1,1),1)],
            [((1+k_off,0),1), ((0+k_off,0),-1)]
        )
        # M6: (A10-A00)(B00+B01) -> C11
        add_term(
            [((1,0),1), ((0,0),-1)],
            [((0,0),1), ((0,1),1)],
            [((1+k_off,1),1)]
        )
        # M7: (A01-A11)(B10+B11) -> C00
        add_term(
            [((0,1),1), ((1,1),-1)],
            [((1,0),1), ((1,1),1)],
            [((0+k_off,0),1)]
        )

        # 2. Schoolbook 2x2x1 (A[0:2, 0:2] * B[0:2, 2])
        # C_i,2 = A_i0 B_02 + A_i1 B_12
        add_term(
            [((0,0),1)],
            [((0,2),1)],
            [((2+k_off,0),1)]
        )
        add_term(
            [((1,0),1)],
            [((0,2),1)],
            [((2+k_off,1),1)]
        )
        add_term(
            [((0,1),1)],
            [((1,2),1)],
            [((2+k_off,0),1)]
        )
        add_term(
            [((1,1),1)],
            [((1,2),1)],
            [((2+k_off,1),1)]
        )

        # 3. Schoolbook 2x1x3 (A[0:2, 2] * B[2, 0:3])
        # C_i,k = A_i2 B_2k
        for i in range(2):
            for k in range(3):
                add_term(
                    [((i,2),1)],
                    [((2,k),1)],
                    [((k+k_off,i),1)]
                )

    return {"U": U, "V": V, "W": W}
