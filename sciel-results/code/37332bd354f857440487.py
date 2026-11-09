DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7

    def strassen_2x2_2x2():
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]

        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[3][0] += 1

        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[2][1] += 1

        U[0][2] += 1
        V[1][2] += 1; V[3][2] += -1
        W[1][2] += 1; W[3][2] += 1

        U[3][3] += 1
        V[2][3] += 1; V[0][3] += -1
        W[0][3] += 1; W[2][3] += 1

        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] += -1; W[3][4] += 1

        U[2][5] += 1; U[0][5] += -1
        V[0][5] += 1; V[2][5] += 1
        W[3][5] += 1

        U[1][6] += 1; U[3][6] += -1
        V[1][6] += 1; V[3][6] += 1
        W[0][6] += 1

        return U, V, W

    def build_strassen_4x4_4x4():
        U2, V2, W2 = strassen_2x2_2x2()
        rank_total = 49
        U = [[0]*rank_total for _ in range(16)]
        V = [[0]*rank_total for _ in range(16)]
        W = [[0]*rank_total for _ in range(16)]

        a_coeffs = [
            {'A00': 1, 'A11': 1},
            {'A10': 1, 'A11': 1},
            {'A00': 1},
            {'A11': 1},
            {'A00': 1, 'A01': 1},
            {'A10': 1, 'A00': -1},
            {'A01': 1, 'A11': -1},
        ]
        b_coeffs = [
            {'B00': 1, 'B11': 1},
            {'B00': 1},
            {'B01': 1, 'B11': -1},
            {'B10': 1, 'B00': -1},
            {'B11': 1},
            {'B00': 1, 'B10': 1},
            {'B01': 1, 'B11': 1},
        ]
        a_block_ranges = {
            'A00': (0, 2, 0, 2),
            'A01': (0, 2, 2, 4),
            'A10': (2, 4, 0, 2),
            'A11': (2, 4, 2, 4),
        }
        b_block_ranges = {
            'B00': (0, 2, 0, 2),
            'B01': (0, 2, 2, 4),
            'B10': (2, 4, 0, 2),
            'B11': (2, 4, 2, 4),
        }
        c_block_coeffs = {
            'C00': {'M1': 1, 'M4': 1, 'M5': -1, 'M7': 1},
            'C01': {'M3': 1, 'M5': 1},
            'C10': {'M2': 1, 'M4': 1},
            'C11': {'M1': 1, 'M2': -1, 'M3': 1, 'M6': 1},
        }
        c_block_ranges = {
            'C00': (0, 2, 0, 2),
            'C01': (0, 2, 2, 4),
            'C10': (2, 4, 0, 2),
            'C11': (2, 4, 2, 4),
        }

        for h in range(7):
            for l in range(7):
                col = h * 7 + l
                for block_name, coeff in a_coeffs[h].items():
                    r0, r1, c0, c1 = a_block_ranges[block_name]
                    for i in range(r0, r1):
                        for j in range(c0, c1):
                            li, lj = i - r0, j - c0
                            uv = U2[li*2+lj][l]
                            if uv:
                                U[i*4+j][col] += coeff * uv
                for block_name, coeff in b_coeffs[h].items():
                    r0, r1, c0, c1 = b_block_ranges[block_name]
                    for j in range(r0, r1):
                        for k in range(c0, c1):
                            lj, lk = j - r0, k - c0
                            vv = V2[lj*2+lk][l]
                            if vv:
                                V[j*4+k][col] += coeff * vv
                for cb, mc in c_block_coeffs.items():
                    key = f'M{h+1}'
                    if key in mc:
                        wc = mc[key]
                        r0, r1, c0, c1 = c_block_ranges[cb]
                        for k in range(c0, c1):
                            for i in range(r0, r1):
                                lk, li = k - c0, i - r0
                                wv = W2[lk*2+li][l]
                                if wv:
                                    W[k*4+i][col] += wc * wv
        return U, V, W

    def build_schoolbook_4x4_4x3():
        rank = 48
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(12)]
        W = [[0]*rank for _ in range(12)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i*4+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return U, V, W

    U4, V4, W4 = build_strassen_4x4_4x4()
    U3, V3, W3 = build_schoolbook_4x4_4x3()

    rank_total = 97
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(28)]
    W = [[0]*rank_total for _ in range(28)]

    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
    for j in range(16):
        for c in range(49):
            V[j][c] = V4[j][c]
    for k in range(16):
        for c in range(49):
            W[k][c] = W4[k][c]

    for i in range(16):
        for c in range(48):
            U[i][49+c] = U3[i][c]
    for j in range(12):
        for c in range(48):
            V[16+j][49+c] = V3[j][c]
    for k in range(12):
        for c in range(48):
            W[16+k][49+c] = W3[k][c]

    return {"U": U, "V": V, "W": W}
