import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

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

    def build_strassen_rank39():
        """Build the rank-39 Strassen-based decomposition."""
        rank = 39
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

            add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
            add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
            add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])
            add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
            add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
            add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
            add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])

        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)

        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1

        add_term([0], [1], [b06], [1], [c06], [1])
        add_term([1], [1], [b16], [1], [c06], [1])
        add_term([3], [1], [b06], [1], [c16], [1])
        add_term([4], [1], [b16], [1], [c16], [1])

        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            add_term([2], [1], [b2k], [1], [c0k], [1])
            add_term([5], [1], [b2k], [1], [c1k], [1])

        return {"U": U, "V": V, "W": W}

    def verify_decomposition(result):
        """Verify that the decomposition is exact using integer arithmetic."""
        U = np.array(result["U"], dtype=np.int64)
        V = np.array(result["V"], dtype=np.int64)
        W = np.array(result["W"], dtype=np.int64)
        rank = U.shape[1]

        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1
        return np.array_equal(T_calc, T_target)

    def try_reduce_rank(result, max_reduction=3):
        """Try to reduce rank by checking if any terms can be removed or merged."""
        U = np.array(result["U"], dtype=np.int64)
        V = np.array(result["V"], dtype=np.int64)
        W = np.array(result["W"], dtype=np.int64)
        rank = U.shape[1]
        
        best_result = result
        best_rank = rank
        
        # Try removing each term and see if the decomposition still works
        # This is expensive, so we only try a limited number
        for term_idx in range(rank):
            if time.time() > deadline - 30:
                break
            
            U_test = U[:, :term_idx].tolist() + U[:, term_idx+1:].tolist()
            V_test = V[:, :term_idx].tolist() + V[:, term_idx+1:].tolist()
            W_test = W[:, :term_idx].tolist() + W[:, term_idx+1:].tolist()
            
            test_result = {"U": U_test, "V": V_test, "W": W_test}
            if verify_decomposition(test_result):
                best_result = test_result
                best_rank = rank - 1
                U = np.array(U_test, dtype=np.int64)
                V = np.array(V_test, dtype=np.int64)
                W = np.array(W_test, dtype=np.int64)
                rank = best_rank
        
        return best_result, best_rank

    # Check time budget
    if time.time() > deadline - 10:
        return get_schoolbook()

    # Build the rank-39 Strassen decomposition
    result = build_strassen_rank39()

    # Verify the decomposition with exact integer arithmetic
    if not verify_decomposition(result):
        return get_schoolbook()

    # Try to reduce rank
    if time.time() < deadline - 40:
        result, final_rank = try_reduce_rank(result)
        if verify_decomposition(result):
            return result

    return result

# EVOLVE-BLOCK-END
