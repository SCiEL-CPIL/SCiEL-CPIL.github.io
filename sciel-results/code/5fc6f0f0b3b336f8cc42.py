import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def get_rank46():
        """
        Rank 46 decomposition:
        - 6 Strassen 2x2x2 blocks: 6 * 7 = 42 terms
        - 4 rank-1 terms for the 2x4x1 portion (k=6): 4 terms
        Total: 46
        """
        rank = 46
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # 6 Strassen blocks for the 2x4x6 portion
        # Block 1: A(0:2,0:2), B(0:2,0:2) -> C(0:2,0:2)
        add_strassen(0, 0, 0)
        # Block 2: A(0:2,0:2), B(0:2,2:4) -> C(0:2,2:4)
        add_strassen(0, 0, 2)
        # Block 3: A(0:2,2:4), B(2:4,0:2) -> C(0:2,0:2)
        add_strassen(0, 2, 0)
        # Block 4: A(0:2,2:4), B(2:4,2:4) -> C(0:2,2:4)
        add_strassen(0, 2, 2)
        # Block 5: A(0:2,0:2), B(0:2,4:6) -> C(0:2,4:6)
        add_strassen(0, 0, 4)
        # Block 6: A(0:2,2:4), B(2:4,4:6) -> C(0:2,4:6)
        add_strassen(0, 2, 4)

        # 4 rank-1 terms for the 2x4x1 portion (k=6)
        # Each term handles one column j of A, contributing to both rows of C
        for j in range(m):
            U[j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank {rank}, got {col}"
        return U, V, W

    # Get the rank-46 decomposition
    U46, V46, W46 = get_rank46()

    # Verify it's correct
    if not verify_decomp(U46, V46, W46):
        # Fallback to schoolbook
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}
# EVOLVE-BLOCK-END
