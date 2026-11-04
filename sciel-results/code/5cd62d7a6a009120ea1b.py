"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # all 5

    # Partition indices into 2+2+1 blocks
    # Apply Strassen (rank 7) to 2x2x2 diagonal blocks
    # Apply schoolbook to all other blocks

    n_parts = [2, 2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]

    n_starts = [0]
    m_starts = [0]
    p_starts = [0]
    for i in range(1, len(n_parts)):
        n_starts.append(n_starts[-1] + n_parts[i-1])
    for i in range(1, len(m_parts)):
        m_starts.append(m_starts[-1] + m_parts[i-1])
    for i in range(1, len(p_parts)):
        p_starts.append(p_starts[-1] + p_parts[i-1])

    # Calculate total rank
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if n_parts[i] == 2 and m_parts[j] == 2 and p_parts[k] == 2:
                    total_rank += 7
                else:
                    total_rank += n_parts[i] * m_parts[j] * p_parts[k]

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    for i in range(3):
        for j in range(3):
            for k in range(3):
                if time.time() > deadline:
                    # Return what we have; this should never happen for this small problem
                    return {"U": U, "V": V, "W": W}

                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]

                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]

                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen for 2x2x2 block
                    i0, i1 = i_start, i_start + 1
                    j0, j1 = j_start, j_start + 1
                    k0, k1 = k_start, k_start + 1

                    # M1 = (a00+a11)(b00+b11)
                    U[i0 * m + j0][col] = 1
                    U[i1 * m + j1][col] = 1
                    V[j0 * p + k0][col] = 1
                    V[j1 * p + k1][col] = 1
                    W[k0 * n + i0][col] = 1
                    W[k1 * n + i1][col] = 1
                    col += 1

                    # M2 = (a10+a11)(b00)
                    U[i1 * m + j0][col] = 1
                    U[i1 * m + j1][col] = 1
                    V[j0 * p + k0][col] = 1
                    W[k0 * n + i1][col] = 1
                    W[k1 * n + i1][col] = -1
                    col += 1

                    # M3 = (a00)(b10-b11)
                    U[i0 * m + j0][col] = 1
                    V[j0 * p + k1][col] = 1
                    V[j1 * p + k1][col] = -1
                    W[k1 * n + i0][col] = 1
                    W[k1 * n + i1][col] = 1
                    col += 1

                    # M4 = (a11)(b01-b10)
                    U[i1 * m + j1][col] = 1
                    V[j1 * p + k0][col] = 1
                    V[j0 * p + k0][col] = -1
                    W[k0 * n + i0][col] = 1
                    W[k0 * n + i1][col] = 1
                    col += 1

                    # M5 = (a00+a01)(b11)
                    U[i0 * m + j0][col] = 1
                    U[i0 * m + j1][col] = 1
                    V[j1 * p + k1][col] = 1
                    W[k0 * n + i0][col] = -1
                    W[k1 * n + i0][col] = 1
                    col += 1

                    # M6 = (a01-a11)(b10+b11)
                    U[i1 * m + j0][col] = 1
                    U[i0 * m + j0][col] = -1
                    V[j0 * p + k0][col] = 1
                    V[j0 * p + k1][col] = 1
                    W[k1 * n + i1][col] = -1
                    col += 1

                    # M7 = (a10-a00)(b00+b01)
                    U[i0 * m + j1][col] = 1
                    U[i1 * m + j1][col] = -1
                    V[j1 * p + k0][col] = 1
                    V[j1 * p + k1][col] = 1
                    W[k0 * n + i0][col] = -1
                    col += 1

                else:
                    # Schoolbook for this block
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U[(i_start + u) * m + (j_start + v)][col] = 1
                                V[(j_start + v) * p + (k_start + w)][col] = 1
                                W[(k_start + w) * n + (i_start + u)][col] = 1
                                col += 1

    # Verify the decomposition
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    C_computed = np.zeros((n, p))

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    a_flat = A_test.flatten()
    b_flat = B_test.flatten()

    for t in range(total_rank):
        m_t = np.dot(U_arr[:, t], a_flat) * np.dot(V_arr[:, t], b_flat)
        for k in range(p):
            for i in range(n):
                C_computed[i, k] += W_arr[k * n + i, t] * m_t

    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
