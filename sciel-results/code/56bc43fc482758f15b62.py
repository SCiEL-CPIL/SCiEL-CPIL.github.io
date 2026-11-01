"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 22 decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x1x2 blocks.
    n, m, p = DIMENSIONS  # 2, 3, 4
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]  # shape (6, 22)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (12, 22)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (8, 22)

    def idx_U(i, j):
        return i * m + j

    def idx_V(j, k):
        return j * p + k

    def idx_W(k, i):
        return k * n + i

    # Strassen coefficients for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b11+b01)
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M7

    def add_strassen(offset, i0, j0, k0):
        """
        Add Strassen terms for C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        offset is the starting column index.
        """
        a_indices = [[idx_U(i0 + di, j0 + dj) for dj in range(2)] for di in range(2)]
        b_indices = [[idx_V(j0 + dj, k0 + dk) for dk in range(2)] for dj in range(2)]
        c_indices = [[idx_W(k0 + dk, i0 + di) for di in range(2)] for dk in range(2)]

        # M1
        c = offset
        U[a_indices[0][0]][c] += 1
        U[a_indices[1][1]][c] += 1
        V[b_indices[0][0]][c] += 1
        V[b_indices[1][1]][c] += 1
        W[c_indices[0][0]][c] += 1
        W[c_indices[1][1]][c] += 1

        # M2
        c = offset + 1
        U[a_indices[1][0]][c] += 1
        U[a_indices[1][1]][c] += 1
        V[b_indices[0][0]][c] += 1
        W[c_indices[0][1]][c] += 1
        W[c_indices[1][1]][c] += 1

        # M3
        c = offset + 2
        U[a_indices[0][0]][c] += 1
        V[b_indices[1][0]][c] += 1
        V[b_indices[1][1]][c] -= 1
        W[c_indices[0][1]][c] += 1
        W[c_indices[1][1]][c] += 1

        # M4
        c = offset + 3
        U[a_indices[1][1]][c] += 1
        V[b_indices[0][1]][c] += 1
        V[b_indices[0][0]][c] -= 1
        W[c_indices[0][0]][c] += 1
        W[c_indices[1][1]][c] += 1

        # M5
        c = offset + 4
        U[a_indices[0][0]][c] += 1
        U[a_indices[0][1]][c] += 1
        V[b_indices[1][1]][c] += 1
        W[c_indices[0][0]][c] -= 1
        W[c_indices[1][1]][c] += 1

        # M6
        c = offset + 5
        U[a_indices[1][0]][c] += 1
        U[a_indices[0][0]][c] -= 1
        V[b_indices[0][0]][c] += 1
        V[b_indices[1][0]][c] += 1
        W[c_indices[1][0]][c] += 1

        # M7
        c = offset + 6
        U[a_indices[0][1]][c] += 1
        U[a_indices[1][1]][c] -= 1
        V[b_indices[1][1]][c] += 1
        V[b_indices[0][1]][c] += 1
        W[c_indices[0][0]][c] -= 1
        W[c_indices[1][1]][c] += 1

    def add_schoolbook_2x1x2(offset, i0, j0, k0):
        """
        Add schoolbook terms for C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0] * B[j0, k0:k0+2]
        offset is the starting column index.
        """
        a_indices = [idx_U(i0 + di, j0) for di in range(2)]
        b_indices = [idx_V(j0, k0 + dk) for dk in range(2)]
        c_indices = [[idx_W(k0 + dk, i0 + di) for di in range(2)] for dk in range(2)]

        # c00 = a0 * b0
        c = offset
        U[a_indices[0]][c] += 1
        V[b_indices[0]][c] += 1
        W[c_indices[0][0]][c] += 1

        # c01 = a0 * b1
        c = offset + 1
        U[a_indices[0]][c] += 1
        V[b_indices[1]][c] += 1
        W[c_indices[1][0]][c] += 1

        # c10 = a1 * b0
        c = offset + 2
        U[a_indices[1]][c] += 1
        V[b_indices[0]][c] += 1
        W[c_indices[0][1]][c] += 1

        # c11 = a1 * b1
        c = offset + 3
        U[a_indices[1]][c] += 1
        V[b_indices[1]][c] += 1
        W[c_indices[1][1]][c] += 1

    # Left half: C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2] * B[2, 0:2]
    # Strassen for A[:, 0:2] * B[0:2, 0:2]
    add_strassen(0, i0=0, j0=0, k0=0)
    # Schoolbook for A[:, 2] * B[2, 0:2]
    add_schoolbook_2x1x2(7, i0=0, j0=2, k0=0)

    # Right half: C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2] * B[2, 2:4]
    # Strassen for A[:, 0:2] * B[0:2, 2:4]
    add_strassen(11, i0=0, j0=0, k0=2)
    # Schoolbook for A[:, 2] * B[2, 2:4]
    add_schoolbook_2x1x2(18, i0=0, j0=2, k0=2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    import numpy as np
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Verify
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((p, n))
    for c in range(rank):
        a_factor = U[:, c]
        v_factor = V[:, c]
        w_factor = W[:, c]
        # a_factor is length n*m, reshape to (n, m)
        a_mat = a_factor.reshape(n, m)
        v_mat = v_factor.reshape(m, p)
        # Compute scalar product (a_mat * v_mat)
        prod = np.sum(a_mat * v_mat)
        c_mat = w_factor.reshape(p, n) * prod
        C_computed += c_mat
    err = np.max(np.abs(C_expected - C_computed.T))
    print(f"Max error: {err}")
    print(f"Rank: {rank}")
