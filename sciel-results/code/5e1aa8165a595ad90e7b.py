import numpy as np
import time

start_time = time.time()
TIME_LIMIT = 118.0


def construct():
    n, m, p = 2, 3, 9

    # Rank 50 decomposition for 2x3 x 3x9
    # Part 1: Use Strassen-style 7-mult multiplication for the 2x2 x 2x8 block
    # (rows 0,1 of A; cols 0,1 of A; rows 0,1 of B; cols 0-7 of B)
    # Part 2: Handle remaining terms directly
    #   - A[:,0:2] x B[0:2, 8] (1 column)
    #   - A[:,2:3] x B[2, :] (3rd row of A with all 9 cols of B)

    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col_idx = 0

    # Part 1: 2x2 Strassen blocks for 4 pairs of columns (0,1), (2,3), (4,5), (6,7)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1

        # V indices for B[0,k0], B[0,k1], B[1,k0], B[1,k1]
        v_e = 0 * p + k0   # B[0, k0]
        v_f = 0 * p + k1   # B[0, k1]
        v_g = 1 * p + k0   # B[1, k0]
        v_h = 1 * p + k1   # B[1, k1]

        # U indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
        u_a = 0
        u_b = 1
        u_c = 3
        u_d = 4

        # Strassen 7 products:
        # M1 = (A00+A11)(B00+B11)
        U[u_a][col_idx] += 1
        U[u_d][col_idx] += 1
        V[v_e][col_idx] += 1
        V[v_h][col_idx] += 1

        # M2 = (A10+A11)B00
        U[u_c][col_idx + 1] += 1
        U[u_d][col_idx + 1] += 1
        V[v_e][col_idx + 1] += 1

        # M3 = A00(B01-B11)
        U[u_a][col_idx + 2] += 1
        V[v_f][col_idx + 2] += 1
        V[v_h][col_idx + 2] -= 1

        # M4 = A11(B10-B00)
        U[u_d][col_idx + 3] += 1
        V[v_g][col_idx + 3] += 1
        V[v_e][col_idx + 3] -= 1

        # M5 = (A00+A01)B11
        U[u_a][col_idx + 4] += 1
        U[u_b][col_idx + 4] += 1
        V[v_h][col_idx + 4] += 1

        # M6 = (A10-A00)(B00+B01)
        U[u_c][col_idx + 5] += 1
        U[u_a][col_idx + 5] -= 1
        V[v_e][col_idx + 5] += 1
        V[v_f][col_idx + 5] += 1

        # M7 = (A01-A11)(B10+B11)
        U[u_b][col_idx + 6] += 1
        U[u_d][col_idx + 6] -= 1
        V[v_g][col_idx + 6] += 1
        V[v_h][col_idx + 6] += 1

        # W indices: C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        # Convention: W[k*n+i, r] = C[i, k]
        w_00 = k0 * n + 0   # C[0, k0]
        w_01 = k1 * n + 0   # C[0, k1]
        w_10 = k0 * n + 1   # C[1, k0]
        w_11 = k1 * n + 1   # C[1, k1]

        # C[0,k0] = M1 + M4 + M5 - M7
        W[w_00][col_idx] += 1
        W[w_00][col_idx + 3] += 1
        W[w_00][col_idx + 4] += 1
        W[w_00][col_idx + 6] -= 1

        # C[0,k1] = M3 + M5 + M7
        W[w_01][col_idx + 2] += 1
        W[w_01][col_idx + 4] += 1
        W[w_01][col_idx + 6] += 1

        # C[1,k0] = M2 + M6 + M7
        W[w_10][col_idx + 1] += 1
        W[w_10][col_idx + 5] += 1
        W[w_10][col_idx + 6] += 1

        # C[1,k1] = M1 - M2 + M3 + M4 + M6
        W[w_11][col_idx] += 1
        W[w_11][col_idx + 1] -= 1
        W[w_11][col_idx + 2] += 1
        W[w_11][col_idx + 3] += 1
        W[w_11][col_idx + 5] += 1

        col_idx += 7

    # Part 2a: A[:, 0:2] x B[0:2, 8] - 4 direct products
    k8 = 8
    v_e8 = 0 * p + k8   # B[0, 8]
    v_g8 = 1 * p + k8   # B[1, 8]

    u_a = 0; u_b = 1; u_c = 3; u_d = 4

    # A00 * B08 -> C[0,8]
    U[u_a][col_idx] += 1
    V[v_e8][col_idx] += 1
    W[k8 * n + 0][col_idx] += 1

    # A01 * B18 -> C[0,8]
    U[u_b][col_idx + 1] += 1
    V[v_g8][col_idx + 1] += 1
    W[k8 * n + 0][col_idx + 1] += 1

    # A10 * B08 -> C[1,8]
    U[u_c][col_idx + 2] += 1
    V[v_e8][col_idx + 2] += 1
    W[k8 * n + 1][col_idx + 2] += 1

    # A11 * B18 -> C[1,8]
    U[u_d][col_idx + 3] += 1
    V[v_g8][col_idx + 3] += 1
    W[k8 * n + 1][col_idx + 3] += 1

    col_idx += 4

    # Part 2b: A[:, 2:3] x B[2, :] - 18 direct products
    for k in range(9):
        v_b = 2 * p + k  # B[2, k]

        # A02 * B2k -> C[0, k]
        U[2][col_idx] += 1
        V[v_b][col_idx] += 1
        W[k * n + 0][col_idx] += 1

        # A12 * B2k -> C[1, k]
        U[5][col_idx + 1] += 1
        V[v_b][col_idx + 1] += 1
        W[k * n + 1][col_idx + 1] += 1

        col_idx += 2

    # Time guard
    if time.time() - start_time > TIME_LIMIT:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    n, m, p = 2, 3, 9
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B

    A_flat = A.flatten()
    B_flat = B.flatten()

    C_flat = np.zeros(n * p)
    U_arr = np.array(res["U"], dtype=float)
    V_arr = np.array(res["V"], dtype=float)
    W_arr = np.array(res["W"], dtype=float)

    for s in range(len(U_arr[0])):
        val = np.dot(U_arr[:, s], A_flat) * np.dot(V_arr[:, s], B_flat)
        C_flat += val * W_arr[:, s]

    C_actual = C_flat.reshape((p, n)).T

    if not np.allclose(C_actual, C_expected, atol=1e-6):
        print("Mismatch detected")
    else:
        print("Validation Passed")
