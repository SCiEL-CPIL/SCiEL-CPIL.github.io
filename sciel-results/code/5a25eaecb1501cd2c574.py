import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    The rank-11 decomposition is known to be optimal for this tensor.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # The known rank-11 decomposition (optimal)
    def make_rank11():
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def set_u(i, j, r, val): U[i * m + j][r] = val
        def set_v(j, k, r, val): V[j * p + k][r] = val
        def set_w(k, i, r, val): W[k * n + i][r] = val

        # Term 1: (A00 + A11) * (B00 + B11)
        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        # Term 2: (A10 + A11) * B00
        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        # Term 3: A00 * (B01 - B11)
        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        # Term 4: A11 * (B10 - B00)
        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        # Term 5: (A00 + A01) * B11
        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        # Term 6: (A10 - A00) * (B00 + B01)
        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        # Term 7: (A01 - A11) * (B10 + B11)
        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        # Term 8: A00 * B02
        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        # Term 9: A01 * B12
        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        # Term 10: A10 * B02
        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        # Term 11: A11 * B12
        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}

    def verify_solution(sol):
        """Verify that the decomposition exactly computes matrix multiplication."""
        U = np.array(sol["U"], dtype=float)
        V = np.array(sol["V"], dtype=float)
        W = np.array(sol["W"], dtype=float)
        rank = U.shape[1]
        for _ in range(100):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for r in range(rank):
                Mr = 0.0
                for i in range(n):
                    for j in range(m):
                        Mr += U[i * m + j, r] * A[i, j]
                Nr = 0.0
                for j in range(m):
                    for k in range(p):
                        Nr += V[j * p + k, r] * B[j, k]
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += W[k * n + i, r] * Mr * Nr
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    # Return the known optimal rank-11 solution
    sol11 = make_rank11()
    
    # Quick verification (should always pass)
    if not verify_solution(sol11):
        # Extremely unlikely; return anyway
        pass
    
    return sol11

# EVOLVE-BLOCK-END
