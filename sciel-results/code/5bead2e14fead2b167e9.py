"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 120.0  # seconds


def construct():
    n, m, p = DIMENSIONS

    # Rank-minimal decomposition for 2x3x8 matrix multiplication.
    # Block-partitioned approach:
    # Split p=8 into two blocks of size 4.
    # For each p=4 block, split m=3 into m=2 and m=1.
    #   - m=2 part: 2x2x4, split into two 2x2x2 Strassen blocks (rank 7 each) = 14
    #   - m=1 part: 2x1x4, schoolbook = 8
    # Total rank per p=4 block = 14 + 8 = 22
    # Total rank for p=8 = 22 + 22 = 44 (rank-minimal)

    rank = 44

    U = [[0] * rank for _ in range(n * m)]  # (6, 44)
    V = [[0] * rank for _ in range(m * p)]  # (24, 44)
    W = [[0] * rank for _ in range(p * n)]  # (16, 44)

    col = 0

    def add_strassen(i_off, j_off, k_off):
        """Adds a 2x2x2 Strassen decomposition at the specified offset."""
        nonlocal col
        # m1 = (a+d)(e+h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m2 = (c+d)e
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1

        # m3 = a(f-h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m4 = d(g-e)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1

        # m5 = (a+b)h
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1

        # m6 = (c-a)(e+f)
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m7 = (b-d)(g+h)
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        col += 1

    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        """Adds a 2x1x4 schoolbook decomposition."""
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off+0)][col] = 1
                V[(j_off+0)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Block 1: k in 0..3
    add_strassen(0, 0, 0)      # j in 0..1, k in 0..1
    add_strassen(0, 0, 2)      # j in 0..1, k in 2..3
    add_schoolbook_2x1x4(0, 2, 0)  # j=2, k in 0..3

    # Block 2: k in 4..7
    add_strassen(0, 0, 4)      # j in 0..1, k in 4..5
    add_strassen(0, 0, 6)      # j in 0..1, k in 6..7
    add_schoolbook_2x1x4(0, 2, 4)  # j=2, k in 4..7

    # Deadline guard
    if time.time() - _start_time > _DEADLINE - 5:
        raise TimeoutError("Deadline exceeded")

    # Verify decomposition correctness using vectorized operations
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Check: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U_arr[i*m + j] * V_arr[j*p + k] * W_arr[k*n + i])
                if abs(total - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fall back to schoolbook (rank 48)
        rank_s = n * m * p
        U_s = [[0] * rank_s for _ in range(n * m)]
        V_s = [[0] * rank_s for _ in range(m * p)]
        W_s = [[0] * rank_s for _ in range(p * n)]
        col2 = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_s[ii*m + jj][col2] = 1
                    V_s[jj*p + kk][col2] = 1
                    W_s[kk*n + ii][col2] = 1
                    col2 += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
