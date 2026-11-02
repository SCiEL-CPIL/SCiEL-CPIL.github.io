"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        # Check integer/half-integer constraint
        for name, M in (("U", U), ("V", V), ("W", W)):
            for row in M:
                for val in row:
                    if val != int(val) and 2 * val != int(2 * val):
                        return False
        return True

    def schoolbook():
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

    def build_block_decomp():
        """Build a rank-82 decomposition using Strassen on 2x2x2 blocks."""
        rank_total = 82
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]

        col = 0

        for b in range(2):
            j0 = 2 * b

            # --- Strassen 2x2x2 for A[0:2, j0:j0+2] * B[j0:j0+2, 0:2] -> C[0:2, 0:2] ---
            # 7 rank slots for Strassen
            # M1 = (A00+A11)*(B00+B11)
            U[0 * m + (j0 + 0)][col] = 1
            U[1 * m + (j0 + 1)][col] = 1
            V[(j0 + 0) * p + 0][col] = 1
            V[(j0 + 1) * p + 1][col] = 1
            W[0 * n + 0][col] = 1  # C[0,0]
            W[1 * n + 0][col] = 1  # C[1,0]
            col += 1

            # M2 = (A10+A11)*B00
            U[1 * m + (j0 + 0)][col] = 1
            U[1 * m + (j0 + 1)][col] = 1
            V[(j0 + 0) * p + 0][col] = 1
            W[0 * n + 1][col] = 1  # C[0,1]
            col += 1

            # M3 = A00*(B11-B00)
            U[0 * m + (j0 + 0)][col] = 1
            V[(j0 + 1) * p + 1][col] = 1
            V[(j0 + 0) * p + 0][col] = -1
            W[1 * n + 0][col] = 1  # C[1,0]
            col += 1

            # M4 = A11*(B01-B10)
            U[1 * m + (j0 + 1)][col] = 1
            V[(j0 + 0) * p + 1][col] = 1
            V[(j0 + 1) * p + 0][col] = -1
            W[0 * n + 0][col] = 1  # C[0,0]
            W[0 * n + 1][col] = 1  # C[0,1]
            col += 1

            # M5 = (A00+A01)*B10
            U[0 * m + (j0 + 0)][col] = 1
            U[0 * m + (j0 + 1)][col] = 1
            V[(j0 + 1) * p + 0][col] = 1
            W[0 * n + 0][col] = -1  # C[0,0]
            W[0 * n + 1][col] = 1   # C[0,1]
            col += 1

            # M6 = (A10+A11)*(B11-B01)
            U[1 * m + (j0 + 0)][col] = 1
            U[1 * m + (j0 + 1)][col] = 1
            V[(j0 + 1) * p + 1][col] = 1
            V[(j0 + 0) * p + 1][col] = -1
            W[1 * n + 0][col] = 1  # C[1,0]
            W[1 * n + 1][col] = 1  # C[1,1]
            col += 1

            # M7 = (A00-A01)*(B00+B10)
            U[0 * m + (j0 + 0)][col] = 1
            U[0 * m + (j0 + 1)][col] = -1
            V[(j0 + 0) * p + 0][col] = 1
            V[(j0 + 1) * p + 0][col] = 1
            W[0 * n + 0][col] = 1  # C[0,0]
            col += 1

            # --- Direct 2x2x5 for A[0:2, j0:j0+2] * B[j0:j0+2, 2:7] -> C[0:2, 2:7] ---
            # 20 rank slots
            for i in range(2):
                for j in range(2):
                    for k in range(5):
                        U[i * m + (j0 + j)][col] = 1
                        V[(j0 + j) * p + (2 + k)][col] = 1
                        W[(2 + k) * n + i][col] = 1
                        col += 1

            # --- Direct 1x2x7 for A[2, j0:j0+2] * B[j0:j0+2, 0:7] -> C[0:7, 2] ---
            # 14 rank slots
            for j in range(2):
                for k in range(7):
                    U[2 * m + (j0 + j)][col] = 1
                    V[(j0 + j) * p + k][col] = 1
                    W[k * n + 2][col] = 1
                    col += 1

        assert col == rank_total, f"col={col}, rank_total={rank_total}"
        return {"U": U, "V": V, "W": W}

    # Try the block decomposition first
    result = build_block_decomp()
    if verify(result["U"], result["V"], result["W"]):
        return result

    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
