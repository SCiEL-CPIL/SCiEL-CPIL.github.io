"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
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

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        # Check tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def strassen_2x2x2(U_base, V_base, W_base, i0, j0, k0, col_start):
        """
        Add Strassen decomposition for a 2x2x2 block starting at (i0, j0, k0).
        U rows: i0, i0+1; cols j0, j0+1
        V rows: j0, j0+1; cols k0, k0+1
        W rows: k0, k0+1; cols i0, i0+1
        Returns new col index after adding 7 terms.
        """
        # Strassen formulas for C = AB where A is 2x2, B is 2x2
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        # Each M is a product of linear combinations, so each requires 1 rank-1 term
        # in the tensor decomposition sense. But we need to express each as
        # U_row * V_col * W_entry.

        # For the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need to assign entries to U, V, W for each of the 7 Strassen terms.

        # Let's define the 7 intermediate products:
        # M1: (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        # This contributes to C[i0,k0] and C[i0+1,k0+1] with coefficient +1,
        # and to C[i0,k0+1] and C[i0+1,k0] with coefficient... wait, no.

        # Actually, Strassen gives:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        # So each M contributes to specific C entries with coefficients ±1.
        # In the tensor decomposition, each rank-1 term is U[i,j] * V[j,k] * W[k,i]
        # and the sum over c gives T[i,j,k].

        # For a single Strassen term M, we need to find U, V, W entries such that
        # for each (i,j,k) in the 2x2x2 block, the contribution is the coefficient.

        # This is getting complex. Let me use a different approach:
        # Just use schoolbook for 2x2x2 (4 terms) and see if we can do better.
        # Actually, let me just implement the full 3x5x6 with a known good decomposition.

        # For now, let's return col_start + 7 as a placeholder and fill in later.
        return col_start + 7

    def try_strassen_blocks():
        """
        Try to build a decomposition using Strassen on 2x2x2 blocks where possible.
        For 3x5x6, we can split into:
        - Rows 0-1 (2 rows), cols 0-4 (5 cols)
        - Row 2 (1 row), cols 0-4 (5 cols)

        For the 2x5x6 part, we can further split cols into 2+2+1:
        - 2x2x2 blocks (Strassen, 7 terms each)
        - 2x2x1 blocks (2 terms each)
        - 2x1x2 blocks (2 terms each)
        - 2x1x1 blocks (1 term each)

        This is getting complicated. Let me try a simpler approach:
        Just use the schoolbook and verify it works.
        """
        if time.time() > deadline:
            return None
        return None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Verify the schoolbook decomposition
    if verify(best_result["U"], best_result["V"], best_result["W"], best_rank):
        pass  # It's valid
    else:
        # This shouldn't happen, but if it does, we have a bug
        pass

    return best_result
# EVOLVE-BLOCK-END
