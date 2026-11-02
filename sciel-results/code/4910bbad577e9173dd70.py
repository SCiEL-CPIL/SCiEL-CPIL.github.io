import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = 4, 4, 6

    # Strategy: Recursive Strassen for 4x4 * 4x6.
    # Split A (4x4) into four 2x2 blocks, B (4x6) into two 2x6 halves.
    # C[0:2,:] = A[0:2,0:2]*B[0:2,:] + A[0:2,2:4]*B[2:4,:]
    # C[2:4,:] = A[2:4,0:2]*B[0:2,:] + A[2:4,2:4]*B[2:4,:]
    # Each 2x2*2x6 uses Strassen on three 2x2 column-blocks: 3*7 = 21 terms.
    # Total: 4 * 21 = 84 terms.

    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84

    col = 0

    def a_idx(r, c):
        return r * m + c

    def b_idx(r, c):
        return r * p + c

    def c_idx(r, c):
        return c * n + r  # W[k*n+i] where i=row, k=col

    def add_term(u_entries, v_entries, w_entries, c):
        for idx, val in u_entries:
            U[idx][c] += val
        for idx, val in v_entries:
            V[idx][c] += val
        for idx, val in w_entries:
            W[idx][c] += val

    # Strassen for 2x2 * 2x2 at A[r0:r0+2, c0:c0+2] * B[rb0:rb0+2, cb0:cb0+2] -> C[rc0:rc0+2, cc0:cc0+2]
    def add_strassen_2x2x2(r0, c0, rb0, cb0, rc0, cc0, c):
        A11 = a_idx(r0, c0)
        A12 = a_idx(r0, c0 + 1)
        A21 = a_idx(r0 + 1, c0)
        A22 = a_idx(r0 + 1, c0 + 1)

        B11 = b_idx(rb0, cb0)
        B12 = b_idx(rb0, cb0 + 1)
        B21 = b_idx(rb0 + 1, cb0)
        B22 = b_idx(rb0 + 1, cb0 + 1)

        C11 = c_idx(rc0, cc0)
        C12 = c_idx(rc0, cc0 + 1)
        C21 = c_idx(rc0 + 1, cc0)
        C22 = c_idx(rc0 + 1, cc0 + 1)

        # M1 = (A11+A22)(B11+B22)
        add_term([(A11, 1), (A22, 1)], [(B11, 1), (B22, 1)],
                 [(C11, 1), (C22, 1)], c)
        c += 1

        # M2 = (A21+A22)B11
        add_term([(A21, 1), (A22, 1)], [(B11, 1)],
                 [(C21, 1)], c)
        c += 1

        # M3 = A11(B12-B22)
        add_term([(A11, 1)], [(B12, 1), (B22, -1)],
                 [(C12, 1)], c)
        c += 1

        # M4 = A22(B21-B11)
        add_term([(A22, 1)], [(B21, 1), (B11, -1)],
                 [(C11, -1), (C21, 1)], c)
        c += 1

        # M5 = (A11+A12)B22
        add_term([(A11, 1), (A12, 1)], [(B22, 1)],
                 [(C11, -1), (C12, 1)], c)
        c += 1

        # M6 = (A21-A11)(B11+B12)
        add_term([(A21, 1), (A11, -1)], [(B11, 1), (B12, 1)],
                 [(C22, 1)], c)
        c += 1

        # M7 = (A12-A22)(B21+B22)
        add_term([(A12, 1), (A22, -1)], [(B21, 1), (B22, 1)],
                 [(C11, 1), (C22, -1)], c)
        c += 1

        return c

    # For 2x2 * 2x6: split B into three 2x2 column-blocks
    def add_strassen_2x2x6(r0, c0, rb0, cb0, rc0, cc0, c):
        c = add_strassen_2x2x2(r0, c0, rb0, cb0, rc0, cc0, c)
        c = add_strassen_2x2x2(r0, c0, rb0, cb0 + 2, rc0, cc0 + 2, c)
        c = add_strassen_2x2x2(r0, c0, rb0, cb0 + 4, rc0, cc0 + 4, c)
        return c

    # Build the four 2x2*2x6 products
    col = add_strassen_2x2x6(0, 0, 0, 0, 0, 0, col)
    col = add_strassen_2x2x6(0, 2, 2, 0, 0, 0, col)
    col = add_strassen_2x2x6(2, 0, 0, 0, 2, 0, col)
    col = add_strassen_2x2x6(2, 2, 2, 0, 2, 0, col)

    assert col == rank, f"Expected {rank} columns, got {col}"

    # Verify with random test
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 6)
    C_expected = A_test @ B_test

    C_computed = np.zeros((4, 6))
    for r in range(rank):
        alpha_r = np.dot(U_np[:, r], A_test.flatten())
        beta_r = np.dot(V_np[:, r], B_test.flatten())
        C_computed += alpha_r * beta_r * W_np[:, r].reshape(6, 4).T

    error = np.max(np.abs(C_computed - C_expected))

    if error > 1e-10:
        # Fall back to naive if verification fails
        rank_naive = n * m * p
        U = [[0] * rank_naive for _ in range(n * m)]
        V = [[0] * rank_naive for _ in range(m * p)]
        W = [[0] * rank_naive for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
