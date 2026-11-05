import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14

    start_time = time.time()
    deadline = start_time + 110.0

    def get_schoolbook():
        rank = n * m * p  # 42
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def build_rank32():
        """Rank-32 decomposition:
        - 3 Strassen blocks (2x2x2) for k-pairs (0,1), (2,3), (4,5) -> 3*7 = 21
        - 1 schoolbook block (2x2x1) for k=6, j=0,1 -> 4
        - 1 factored block (2x1x7) for j=2 -> 7 (outer product factorization)
        Total: 21 + 4 + 7 = 32
        """
        rank = 32
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]

        col = 0

        def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
            nonlocal col
            for idx, c in zip(u_indices, u_coeffs):
                U[idx][col] += c
            for idx, c in zip(v_indices, v_coeffs):
                V[idx][col] += c
            for idx, c in zip(w_indices, w_coeffs):
                W[idx][col] += c
            col += 1

        def add_strassen_block(k_start):
            """Add a 2x2x2 Strassen block for output columns k_start, k_start+1."""
            a00 = 0 * 3 + 0
            a01 = 0 * 3 + 1
            a10 = 1 * 3 + 0
            a11 = 1 * 3 + 1

            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1

            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1

            # M1: (A00+A11)(B00+B11) -> +C00, +C11
            add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])

            # M2: (A10+A11)B00 -> +C10, -C11
            add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])

            # M3: A00(B01-B11) -> +C01, +C11
            add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])

            # M4: A11(B10-B00) -> +C00, +C10
            add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])

            # M5: (A00+A01)B11 -> -C00, +C01
            add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])

            # M6: (A10-A00)(B00+B01) -> +C11
            add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])

            # M7: (A01-A11)(B10+B11) -> +C00
            add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])

        # Add 3 Strassen blocks for k-pairs (0,1), (2,3), (4,5)
        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)

        # Block for k=6, j=0,1: 2x2x1 schoolbook (4 terms)
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1

        add_term([0], [1], [b06], [1], [c06], [1])
        add_term([1], [1], [b16], [1], [c06], [1])
        add_term([3], [1], [b06], [1], [c16], [1])
        add_term([4], [1], [b16], [1], [c16], [1])

        # Block for j=2: 2x1x7 factored (7 terms)
        # T[i,2,k] = 1 for all i in {0,1}, k in {0..6}
        # Factor: U[i*3+2, r] = 1 for both i, V[2*7+k, r] = delta_{k,r}, W[k*2+i, r] = 1
        for r in range(7):
            k = r
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            # U[0,2] and U[1,2] both get 1, V[2,k] gets 1, W[k,0] and W[k,1] both get 1
            add_term([2, 5], [1, 1], [b2k], [1], [c0k, c1k], [1, 1])

        return {"U": U, "V": V, "W": W}

    def verify_decomposition(result):
        U = np.array(result["U"], dtype=np.float64)
        V = np.array(result["V"], dtype=np.float64)
        W = np.array(result["W"], dtype=np.float64)
        rank = U.shape[1]

        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1.0
        return np.allclose(T_calc, T_target, atol=1e-9)

    if time.time() > deadline - 10:
        return get_schoolbook()

    result = build_rank32()

    if verify_decomposition(result):
        return result

    return get_schoolbook()
