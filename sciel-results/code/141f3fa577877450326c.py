"""Matrix multiplication 6x6x6 tensor decomposition via Strassen on 2x2 blocks of 3x3."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 100.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank = 63  # 7 (Strassen) * 9 (schoolbook 3x3)

    U = np.zeros((36, rank), dtype=np.float64)
    V = np.zeros((36, rank), dtype=np.float64)
    W = np.zeros((36, rank), dtype=np.float64)

    # Strassen terms for 2x2 block matrix multiplication
    # Each term: (A_coeff, B_coeff, C_coeff) where coeff maps (block_row, block_col) -> scalar
    strassen_terms = [
        # M0 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M1 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M2 = A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M3 = A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M4 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M5 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M6 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    for t, (a_coeff, b_coeff, c_coeff) in enumerate(strassen_terms):
        for s in range(9):  # 3x3 schoolbook terms
            r = s // 3  # local row in 3x3 block
            c = s % 3   # local col in 3x3 block
            idx = t * 9 + s

            # Fill U: row = global_row * 6 + global_col
            for (br, bc), val in a_coeff.items():
                global_row = br * 3 + r
                global_col = bc * 3 + c
                U[global_row * 6 + global_col, idx] += val

            # Fill V: row = global_row * 6 + global_col (same block structure for B)
            for (br, bc), val in b_coeff.items():
                global_row = br * 3 + r
                global_col = bc * 3 + c
                V[global_row * 6 + global_col, idx] += val

            # Fill W: trace convention W[k*n + i] where k=col of C, i=row of C
            for (br, bc), val in c_coeff.items():
                global_row = br * 3 + r  # row of C
                global_col = bc * 3 + c  # col of C
                W[global_col * 6 + global_row, idx] += val

    # Verify with random matrices
    if time.time() < deadline - 5:
        np.random.seed(42)
        A = np.random.randn(6, 6)
        B = np.random.randn(6, 6)
        C_expected = A @ B

        C_computed = np.zeros((6, 6))
        for i in range(6):
            for k in range(6):
                s_val = 0.0
                for r in range(rank):
                    u_val = 0.0
                    for j in range(6):
                        u_val += A[i, j] * U[i * 6 + j, r]
                    v_val = 0.0
                    for j in range(6):
                        v_val += B[j, k] * V[j * 6 + k, r]
                    s_val += u_val * v_val * W[k * 6 + i, r]
                C_computed[i, k] = s_val

        err = np.max(np.abs(C_computed - C_expected))
        if err > 1e-9:
            # Fallback to schoolbook rank 216
            U_fb = np.zeros((36, 216))
            V_fb = np.zeros((36, 216))
            W_fb = np.zeros((36, 216))
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        idx = i * 36 + j * 6 + k
                        U_fb[i * 6 + j, idx] = 1
                        V_fb[j * 6 + k, idx] = 1
                        W_fb[k * 6 + i, idx] = 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
