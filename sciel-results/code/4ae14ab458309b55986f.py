import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS  # 4, 4, 7

    def build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_strassen_2x2():
        rank_s = 7
        U_s = [[0] * rank_s for _ in range(4)]
        V_s = [[0] * rank_s for _ in range(4)]
        W_s = [[0] * rank_s for _ in range(4)]

        # P1 = (a11+a22)(b11+b22)
        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[3][0] = 1
        W_s[0][0] = 1; W_s[3][0] = 1

        # P2 = (a21+a22)b11
        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        W_s[2][1] = 1; W_s[3][1] = -1

        # P3 = a11(b12-b22)
        U_s[0][2] = 1
        V_s[1][2] = 1; V_s[3][2] = -1
        W_s[1][2] = 1; W_s[3][2] = 1

        # P4 = a22(b21-b11)
        U_s[3][3] = 1
        V_s[2][3] = 1; V_s[0][3] = -1
        W_s[0][3] = 1; W_s[2][3] = 1

        # P5 = (a11+a12)b22
        U_s[0][4] = 1; U_s[1][4] = 1
        V_s[3][4] = 1
        W_s[0][4] = -1; W_s[1][4] = 1

        # P6 = (a21-a11)(b11+b12)
        U_s[2][5] = 1; U_s[0][5] = -1
        V_s[0][5] = 1; V_s[1][5] = 1
        W_s[3][5] = 1

        # P7 = (a12-a22)(b21+b22)
        U_s[1][6] = 1; U_s[3][6] = -1
        V_s[2][6] = 1; V_s[3][6] = 1
        W_s[0][6] = 1

        return U_s, V_s, W_s

    def build_4x4x4_strassen():
        U_s, V_s, W_s = build_strassen_2x2()
        rank_44 = 49
        U_44 = [[0] * rank_44 for _ in range(16)]
        V_44 = [[0] * rank_44 for _ in range(16)]
        W_44 = [[0] * rank_44 for _ in range(16)]

        A_coef = [
            [1, 0, 0, 1],   # P1: A00 + A11
            [0, 0, 1, 1],   # P2: A10 + A11
            [1, 0, 0, 0],   # P3: A00
            [0, 0, 0, 1],   # P4: A11
            [1, 1, 0, 0],   # P5: A00 + A01
            [-1, 0, 1, 0],  # P6: -A00 + A10
            [0, 1, 0, -1],  # P7: A01 - A11
        ]
        B_coef = [
            [1, 0, 0, 1],   # P1: B00 + B11
            [1, 0, 0, 0],   # P2: B00
            [0, 1, 0, -1],  # P3: B01 - B11
            [-1, 0, 1, 0],  # P4: -B00 + B10
            [0, 0, 0, 1],   # P5: B11
            [1, 1, 0, 0],   # P6: B00 + B01
            [0, 0, 1, 1],   # P7: B10 + B11
        ]
        C_coef = [
            [1, 0, 0, 1],   # P1: C00 += 1, C11 += 1
            [0, 0, 1, -1],  # P2: C10 += 1, C11 -= 1
            [0, 1, 0, 1],   # P3: C01 += 1, C11 += 1
            [1, 0, 1, 0],   # P4: C00 += 1, C10 += 1
            [-1, 1, 0, 0],  # P5: C00 -= 1, C01 += 1
            [0, 0, 0, 1],   # P6: C11 += 1
            [1, 0, 0, 0],   # P7: C00 += 1
        ]

        for s in range(7):
            for t in range(7):
                col = s * 7 + t
                for bi in range(2):
                    for bj in range(2):
                        a_mult = A_coef[s][bi*2+bj]
                        if a_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                u_val = U_s[ii*2+jj][t]
                                if u_val == 0:
                                    continue
                                gi = bi*2 + ii
                                gj = bj*2 + jj
                                U_44[gi*4+gj][col] += a_mult * u_val
                
                for bj in range(2):
                    for bk in range(2):
                        b_mult = B_coef[s][bj*2+bk]
                        if b_mult == 0:
                            continue
                        for jj in range(2):
                            for kk in range(2):
                                v_val = V_s[jj*2+kk][t]
                                if v_val == 0:
                                    continue
                                gj = bj*2 + jj
                                gk = bk*2 + kk
                                V_44[gj*4+gk][col] += b_mult * v_val
                
                for ci in range(2):
                    for cj in range(2):
                        c_mult = C_coef[s][ci*2+cj]
                        if c_mult == 0:
                            continue
                        for kk in range(2):
                            for ii in range(2):
                                w_val = W_s[kk*2+ii][t]
                                if w_val == 0:
                                    continue
                                gk = ci*2 + kk
                                gi = cj*2 + ii
                                W_44[gk*4+gi][col] += c_mult * w_val

        return U_44, V_44, W_44

    def build_4x4x7_optimized():
        U_44, V_44, W_44 = build_4x4x4_strassen()
        rank_44 = 49
        rank_43 = 48
        total_rank = rank_44 + rank_43
        
        U_out = [[0] * total_rank for _ in range(16)]
        V_out = [[0] * total_rank for _ in range(28)]
        W_out = [[0] * total_rank for _ in range(28)]

        for r in range(16):
            for c in range(rank_44):
                U_out[r][c] = U_44[r][c]
                V_out[r][c] = V_44[r][c]
                W_out[r][c] = W_44[r][c]

        col_offset = rank_44
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    kk = k + 4
                    U_out[i * 4 + j][col_offset] = 1
                    V_out[j * 7 + kk][col_offset] = 1
                    W_out[kk * 4 + i][col_offset] = 1
                    col_offset += 1

        return {"U": U_out, "V": V_out, "W": W_out}

    if time.time() - start_time > DEADLINE:
        return build_schoolbook()

    try:
        result = build_4x4x7_optimized()
        
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        rank = U.shape[1]
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        tensor_decomp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        target = np.ones((n, m, p), dtype=float)
        if not np.allclose(tensor_decomp, target, atol=1e-9):
            return build_schoolbook()

        return result
    except Exception:
        return build_schoolbook()
