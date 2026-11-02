import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = 120.0  # safety margin before 150s kill

    n, m, p = 2, 2, 12
    # Rank 42: 6 blocks of 2 columns, each using Strassen rank 7
    rank = 42

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    num_blocks = p // 2  # 6 blocks of 2 columns each

    col = 0
    for b in range(num_blocks):
        # B block indices: B(j, 2b) -> V[j*p + 2b], B(j, 2b+1) -> V[j*p + 2b+1]
        v_b00 = 0 * p + 2 * b      # B[0][2b]
        v_b01 = 0 * p + 2 * b + 1  # B[0][2b+1]
        v_b10 = 1 * p + 2 * b      # B[1][2b]
        v_b11 = 1 * p + 2 * b + 1  # B[1][2b+1]

        # C block indices: C(i, 2b) -> W[(2b)*n + i] = W[4b + i]
        w_c00 = 4 * b + 0   # C[0][2b]
        w_c01 = 4 * b + 1   # C[0][2b+1]
        w_c10 = 4 * b + 2   # C[1][2b]
        w_c11 = 4 * b + 3   # C[1][2b+1]

        # Strassen rank-7 decomposition for 2x2 x 2x2:
        # M1 = (A00+A11)(B00+B11)
        U[0, col] = 1
        U[3, col] = 1
        V[v_b00, col] = 1
        V[v_b11, col] = 1
        W[w_c00, col] = 1
        W[w_c11, col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U[2, col] = 1
        U[3, col] = 1
        V[v_b00, col] = 1
        W[w_c10, col] = 1
        W[w_c11, col] = -1
        col += 1

        # M3 = A00(B01-B11)
        U[0, col] = 1
        V[v_b01, col] = 1
        V[v_b11, col] = -1
        W[w_c01, col] = 1
        W[w_c11, col] = 1
        col += 1

        # M4 = A11(B10-B00)
        U[3, col] = 1
        V[v_b10, col] = 1
        V[v_b00, col] = -1
        W[w_c00, col] = 1
        W[w_c10, col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U[0, col] = 1
        U[1, col] = 1
        V[v_b11, col] = 1
        W[w_c00, col] = -1
        W[w_c01, col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[2, col] = 1
        U[0, col] = -1
        V[v_b00, col] = 1
        V[v_b01, col] = 1
        W[w_c11, col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[1, col] = 1
        U[3, col] = -1
        V[v_b10, col] = 1
        V[v_b11, col] = 1
        W[w_c00, col] = 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"

    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
