import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_dict, v_dict, w_dict):
        U_row = np.zeros(nm)
        V_row = np.zeros(mp)
        W_row = np.zeros(pn)
        for (i, j), val in u_dict.items():
            U_row[i * m + j] += val
        for (j, k), val in v_dict.items():
            V_row[j * p + k] += val
        for (k, i), val in w_dict.items():
            W_row[k * n + i] += val
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)

    def strassen_4x4(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
        """Add 28 terms for a 4x4*4x4 Strassen decomposition."""
        
        def A_block(block, sign=1):
            bi, bj = divmod(block, 2)
            result = {}
            for si in range(2):
                for sj in range(2):
                    gi = A_row_off + bi * 2 + si
                    gj = A_col_off + bj * 2 + sj
                    result[(si, sj)] = [(gi, gj, sign)]
            return result

        def B_block(block, sign=1):
            bi, bj = divmod(block, 2)
            result = {}
            for sj in range(2):
                for sk in range(2):
                    gj = B_row_off + bi * 2 + sj
                    gk = B_col_off + bj * 2 + sk
                    result[(sj, sk)] = [(gj, gk, sign)]
            return result

        def add_block_product(A_coeffs, B_coeffs, C_block_coeffs):
            for b in range(2):
                u_dict = {}
                v_dict = {}
                w_dict = {}
                for (si, sj), entries in A_coeffs.items():
                    if sj == b:
                        for (gi, gj, c) in entries:
                            u_dict[(gi, gj)] = u_dict.get((gi, gj), 0) + c
                for (sj, sk), entries in B_coeffs.items():
                    if sj == b:
                        for (gj, gk, c) in entries:
                            v_dict[(gj, gk)] = v_dict.get((gj, gk), 0) + c
                for (si, sk), c in C_block_coeffs.items():
                    if c != 0:
                        for di in range(2):
                            for dk in range(2):
                                gi = C_row_off + si * 2 + di
                                gk = C_col_off + sk * 2 + dk
                                w_dict[(gk, gi)] = w_dict.get((gk, gi), 0) + c
                if u_dict and v_dict and w_dict:
                    add_term(u_dict, v_dict, w_dict)

        # M1 = (A00+A11)*(B00+B11), C00+=M1, C11+=M1
        A1 = {**A_block(0, 1), **A_block(3, 1)}
        B1 = {**B_block(0, 1), **B_block(3, 1)}
        C1 = {(0, 0): 1, (1, 1): 1}
        add_block_product(A1, B1, C1)

        # M2 = (A10+A11)*B00, C10+=M2, C11-=M2
        A2 = {**A_block(2, 1), **A_block(3, 1)}
        B2 = B_block(0, 1)
        C2 = {(0, 1): 1, (1, 1): -1}
        add_block_product(A2, B2, C2)

        # M3 = A00*(B01-B11), C01+=M3, C11+=M3
        A3 = A_block(0, 1)
        B3 = {**B_block(1, 1), **B_block(3, -1)}
        C3 = {(0, 1): 1, (1, 1): 1}
        add_block_product(A3, B3, C3)

        # M4 = A11*(B10-B00), C00-=M4, C10+=M4
        A4 = A_block(3, 1)
        B4 = {**B_block(2, 1), **B_block(0, -1)}
        C4 = {(0, 0): -1, (0, 1): 1}
        add_block_product(A4, B4, C4)

        # M5 = (A00+A10)*B11, C00+=M5, C01+=M5
        A5 = {**A_block(0, 1), **A_block(2, 1)}
        B5 = B_block(3, 1)
        C5 = {(0, 0): 1, (0, 1): 1}
        add_block_product(A5, B5, C5)

        # M6 = (A01+A11)*(B00-B10), C11+=M6
        A6 = {**A_block(1, 1), **A_block(3, 1)}
        B6 = {**B_block(0, 1), **B_block(2, -1)}
        C6 = {(1, 1): 1}
        add_block_product(A6, B6, C6)

        # M7 = (A00+A01)*(B11-B01), C00-=M7
        A7 = {**A_block(0, 1), **A_block(1, 1)}
        B7 = {**B_block(3, 1), **B_block(1, -1)}
        C7 = {(0, 0): -1}
        add_block_product(A7, B7, C7)

    # Apply Strassen for B_left (cols 0:4)
    strassen_4x4(0, 0, 0, 0, 0, 0)
    # Apply Strassen for B_right (cols 4:8)
    strassen_4x4(0, 0, 0, 4, 0, 4)

    rank = len(U_list)
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
