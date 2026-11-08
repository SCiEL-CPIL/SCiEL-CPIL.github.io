import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = 4, 4, 8
    rank = 98

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Strassen 2x2x2 decomposition
    U_2 = np.zeros((4, 7), dtype=np.float64)
    V_2 = np.zeros((4, 7), dtype=np.float64)
    W_2 = np.zeros((4, 7), dtype=np.float64)

    # c=0: (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    # c=1: (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    # c=2: A00(B01-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    # c=3: A11(B10-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    # c=4: (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    # c=5: (A10-A00)(B00+B01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    # c=6: (A01-A11)(B10+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Block Strassen coefficients for 4x4x4
    # A_specs[k_p]: list of (coeff, block_idx) for effective A
    # B_specs[k_p]: list of (coeff, block_idx) for effective B  
    # C_specs[k_p]: list of (coeff, block_idx) for how P_k contributes to C blocks
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)

    A_specs = [
        [(1, 0), (1, 3)],    # P0: A00 + A11
        [(1, 2), (1, 3)],    # P1: A10 + A11
        [(1, 0)],            # P2: A00
        [(1, 3)],            # P3: A11
        [(1, 0), (1, 1)],    # P4: A00 + A01
        [(1, 2), (-1, 0)],   # P5: A10 - A00
        [(1, 1), (-1, 3)],   # P6: A01 - A11
    ]

    B_specs = [
        [(1, 0), (1, 3)],    # P0: B00 + B11
        [(1, 0)],            # P1: B00
        [(1, 1), (-1, 3)],   # P2: B01 - B11
        [(1, 2), (-1, 0)],   # P3: B10 - B00
        [(1, 3)],            # P4: B11
        [(1, 0), (1, 1)],    # P5: B00 + B01
        [(1, 2), (1, 3)],    # P6: B10 + B11
    ]

    C_specs = [
        [(1, 0), (1, 3)],    # P0: C00 + C11
        [(1, 2), (-1, 3)],   # P1: C10 - C11
        [(1, 1), (1, 3)],    # P2: C01 + C11
        [(1, 0), (1, 2)],    # P3: C00 + C10
        [(-1, 0), (1, 1)],   # P4: -C00 + C01
        [(1, 3)],            # P5: C11
        [(1, 0)],            # P6: C00
    ]

    # Precompute mappings
    # For each block_idx (0-3), map to (bi, bj)
    block_to_bi_bj = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

    for half in range(2):
        col_base = half * 49

        for k_p in range(7):
            for m_s in range(7):
                col = col_base + k_p * 7 + m_s

                # U contribution: A is 4x4, indexed as (row)*4 + col
                # U_2 has shape (4, 7), indexed as a_eff_idx = r*2+c where r,c in {0,1}
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = block_to_bi_bj[block_idx]
                    for r in range(2):
                        for c in range(2):
                            global_a_idx = (bi * 2 + r) * 4 + (bj * 2 + c)
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]

                # V contribution: B is 4x8, indexed as (row)*8 + col
                # For half=0, cols 0-3; for half=1, cols 4-7
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = block_to_bi_bj[block_idx]
                    for r in range(2):
                        for c in range(2):
                            global_b_idx = (bi * 2 + r) * 8 + (bj * 2 + c + half * 4)
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]

                # W contribution: W is 8x4, indexed as (row)*4 + col
                # W[k*n+i, r] where k is output col index (0-7), i is output row index (0-3)
                # C block (bi, bj) in 4x4 output: bi is row block, bj is col block
                # Global W index: (bj*2+c + half*4)*4 + (bi*2+r)
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = block_to_bi_bj[c_block_idx]
                    for r in range(2):
                        for c in range(2):
                            global_w_idx = (bj * 2 + c + half * 4) * 4 + (bi * 2 + r)
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]

                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
