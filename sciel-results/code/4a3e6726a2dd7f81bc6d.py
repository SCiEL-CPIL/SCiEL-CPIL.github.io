import time
import numpy as np

_DEADLINE = 120.0  # seconds

def construct():
    """
    Matrix multiplication 2x3 by 3x8.
    Rank-44 decomposition:
      - 16 scalar terms for the j=0 column of A times the j=0 row of B
      - 4 blocks of 2x2x2 Strassen-style (7 terms each) for j=1,2
    Total rank = 16 + 4*7 = 44
    """
    t_start = time.time()
    if time.time() - t_start > _DEADLINE - 5:
        # Fallback: return schoolbook 48-rank
        n, m, p = 2, 3, 8
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

    n, m, p = 2, 3, 8

    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    col = 0

    # Part 1: j=0, scalar terms: A[i,0] * B[0,k] -> C[i,k]
    for k in range(p):
        for i in range(n):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 16

    # Part 2: 4 blocks of 2x2x2 using 7-term Strassen decomposition
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1

        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1

        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1

        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    """Check that the decomposition is exact by testing basis matrices."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0

    # Convert to numpy for efficient checking
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                # Build the expected C matrix for A=E_{i0,j0}, B=E_{j0,k0}
                expected = np.zeros((n, p))
                expected[i0, k0] = 1.0

                # Compute actual C
                # C[i,k] = sum_r U[i*m+j0] * V[j0*p+k] * W[k*n+i]
                # Vectorized: for each r, outer product of U[:,r] (reshaped) and V[:,r] (reshaped), weighted by W[:,r]
                # Simpler: just compute directly
                actual = np.zeros((n, p))
                for r in range(R):
                    u_col = U_np[:, r]  # length n*m
                    v_col = V_np[:, r]  # length m*p
                    w_col = W_np[:, r]  # length p*n
                    # u_col[i*m + j0] for i in range(n)
                    u_vals = [u_col[i * m + j0] for i in range(n)]
                    # v_col[j0*p + k] for k in range(p)
                    v_vals = [v_col[j0 * p + k] for k in range(p)]
                    # w_col[k*n + i] for k in range(p), i in range(n)
                    for i in range(n):
                        for k in range(p):
                            actual[i, k] += u_vals[i] * v_vals[k] * w_col[k * n + i]

                if not np.allclose(actual, expected, atol=1e-9):
                    return False
    return True


if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
    print(f"Time: {time.time() - t0:.2f}s")
