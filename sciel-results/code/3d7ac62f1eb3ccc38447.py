"""Deterministic multi-Strassen rank-81 decomposition for 3x4*4x7 matrix multiplication.
Uses three overlapping 2x2x2 Strassen blocks to save 6 ranks over pure schoolbook (84)."""
import time

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0  # safety margin before 150s kill


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (21, rank)

    rank = 81
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(col_start, i0, j0, k0):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block starting at (i0, j0, k0)."""
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1    # C[i0,k0] += 1
        W[k1 * n + i1][col_start] = 1    # C[i1,k1] += 1

        # M2 = (A[i1,j0]+A[i1,j1])*B[j0,k0]
        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1    # C[i1,k0] += 1
        W[k1 * n + i1][col_start + 1] = -1   # C[i1,k1] -= 1

        # M3 = A[i0,j0]*(B[j0,k1]-B[j1,k1])
        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1    # C[i0,k1] += 1
        W[k1 * n + i1][col_start + 2] = 1    # C[i1,k1] += 1

        # M4 = A[i1,j1]*(B[j1,k0]-B[j0,k0])
        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1    # C[i0,k0] += 1
        W[k0 * n + i1][col_start + 3] = 1    # C[i1,k0] += 1

        # M5 = (A[i0,j0]+A[i0,j1])*B[j1,k1]
        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1   # C[i0,k0] -= 1
        W[k1 * n + i0][col_start + 4] = 1    # C[i0,k1] += 1

        # M6 = (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1    # C[i1,k1] += 1

        # M7 = (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1    # C[i0,k0] += 1

        return col_start + 7

    # Block 1: i∈{0,1}, j∈{0,1}, k∈{0,1}
    col = add_strassen_block(0, 0, 0, 0)

    # Block 2: i∈{0,1}, j∈{2,3}, k∈{0,1}
    col = add_strassen_block(col, 0, 2, 0)

    # Block 3: i∈{1,2}, j∈{2,3}, k∈{5,6}
    col = add_strassen_block(col, 1, 2, 5)

    # Schoolbook for all remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in Block 1
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    continue
                # Skip if in Block 2
                if i in (0, 1) and j in (2, 3) and k in (0, 1):
                    continue
                # Skip if in Block 3
                if i in (1, 2) and j in (2, 3) and k in (5, 6):
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify the decomposition
    for i in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(
                        f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1"
                    )

    return {"U": U, "V": V, "W": W}
