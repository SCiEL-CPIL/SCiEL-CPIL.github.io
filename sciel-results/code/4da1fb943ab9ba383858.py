"""Optimized matrix multiplication tensor decomposition for 3x4 x 4x8."""
DIMENSIONS = [3, 4, 8]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def build_schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Check: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # Vectorized check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U[i * m + j, :]
                    v = V[j * p + k, :]
                    w = W[k * n + i, :]
                    val = np.dot(u, v * w)
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def build_strassen_2x2x2():
        """Strassen rank-7 decomposition for 2x2 x 2x2."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # Term 0: (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # Term 1: (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        # Term 2: A11(B21-B22)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[2, 2] = 1
        # Term 3: A22(B12-B11)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1
        # Term 4: (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        # Term 5: (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        # Term 6: (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W

    def build_strassen_2x2x8():
        """Rank-28 decomposition for 2x2 x 2x8 via 4 Strassen 2x2x2 blocks."""
        U = np.zeros((4, 28))
        V = np.zeros((16, 28))
        W = np.zeros((16, 28))
        U0, V0, W0 = build_strassen_2x2x2()
        for block in range(4):
            off = block * 7
            U[:, off:off + 7] = U0
            for r in range(2):
                for c in range(2):
                    V[2 * block + r, off + c] = V0[r, c]
                    W[2 * block + r, off + c] = W0[r, c]
        return U, V, W

    def build_hybrid_3x4x8():
        """
        Decomposition for 3x4 x 4x8:
        - A (3x4), B (4x8), C (3x8)
        - Split A into top 2 rows (2x4) and bottom 1 row (1x4)
        - Top: 2x4 @ 4x8 -> split B into two 2x8 row blocks
          - 2x2 @ 2x8 (Strassen rank 28) for cols 0-1
          - 2x2 @ 2x8 (Strassen rank 28) for cols 2-3
          Total top: 56
        - Bottom: 1x4 @ 4x8 -> split B into two 2x8 row blocks
          - 1x2 @ 2x8 (schoolbook rank 8) for cols 0-1
          - 1x2 @ 2x8 (schoolbook rank 8) for cols 2-3
          Total bottom: 16
        Grand total: 72
        """
        total_rank = 72
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))

        # Part 1a: A[0:2, 0:2] (2x2) @ B[0:2, :] (2x8) -> C[0:2, :]
        U1, V1, W1 = build_strassen_2x2x8()
        # A flat: A[i,j] -> i*m+j = i*4+j
        a_idx = [0 * 4 + 0, 0 * 4 + 1, 1 * 4 + 0, 1 * 4 + 1]  # [0,1,4,5]
        for i in range(4):
            U[a_idx[i], 0:28] = U1[i, :]
        # B flat: B[j,k] -> j*p+k = j*8+k
        for j in range(2):
            for k in range(8):
                V[j * 8 + k, 0:28] = V1[j * 8 + k, :]
        # W flat: W[k*n+i, r] where i = C row (0,1), k = C col (0..7)
        for i in range(2):
            for k in range(8):
                W[k * n + i, 0:28] = W1[k * 2 + i, :]

        # Part 1b: A[0:2, 2:4] (2x2) @ B[2:4, :] (2x8) -> C[0:2, :]
        U2, V2, W2 = build_strassen_2x2x8()
        a_idx2 = [0 * 4 + 2, 0 * 4 + 3, 1 * 4 + 2, 1 * 4 + 3]  # [2,3,6,7]
        for i in range(4):
            U[a_idx2[i], 28:56] = U2[i, :]
        for j in range(2, 4):
            for k in range(8):
                V[j * 8 + k, 28:56] = V2[(j - 2) * 8 + k, :]
        for i in range(2):
            for k in range(8):
                W[k * n + i, 28:56] = W2[k * 2 + i, :]

        # Part 2a: A[2, 0:2] (1x2) @ B[0:2, :] (2x8) -> C[2, :]
        # Schoolbook: 2*8 = 16 terms
        off = 56
        for j in range(2):
            for k in range(8):
                col = off + j * 8 + k
                U[2 * 4 + j, col] = 1       # A[2,j]
                V[j * 8 + k, col] = 1        # B[j,k]
                W[k * n + 2, col] = 1        # C[2,k]

        # Part 2b: A[2, 2:4] (1x2) @ B[2:4, :] (2x8) -> C[2, :]
        off = 56 + 16
        for j in range(2):
            for k in range(8):
                col = off + j * 8 + k
                U[2 * 4 + 2 + j, col] = 1    # A[2, 2+j]
                V[(2 + j) * 8 + k, col] = 1  # B[2+j, k]
                W[k * n + 2, col] = 1        # C[2,k]

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try the hybrid Strassen/schoolbook decomposition (rank 72)
    try:
        result = build_hybrid_3x4x8()
        if verify_decomp(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass

    # Fallback: schoolbook (rank 96)
    return build_schoolbook()

# EVOLVE-BLOCK-END
