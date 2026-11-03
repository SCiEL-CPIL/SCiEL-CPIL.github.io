import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0  # safety margin before 150s kill

    # We use the Kronecker product of Strassen (rank 7 for 2x2) and schoolbook 3x3 (rank 27)
    # to get a rank-189 decomposition for 6x6 matrix multiplication.
    #
    # The 6x6 matrix is viewed as a 3x3 block matrix of 2x2 blocks.
    # The multiplication C = A @ B is:
    #   C[i,j] = sum_k A[i,k] @ B[k,j]   (block multiplication)
    #
    # Each 2x2 block product A[i,k] @ B[k,j] can be computed with Strassen's 7-term algorithm.
    # The 3x3 block structure uses schoolbook (27 multiplications).
    # Total rank = 27 * 7 = 189.
    #
    # The decomposition is:
    #   T[i,k,j] = sum_{(s,t)} U[i,k,s,t] * V[k,j,s,t] * W[j,i,s,t]
    # where (s,t) ranges over 3*3*7 = 189 terms.
    #
    # Actually, let me be more precise. The tensor T for 6x6 matmul has:
    #   T[a, b, c] = 1 if a = i*6+k, b = k*6+j, c = j*6+i (using trace convention for W)
    #
    # Wait, let me reconsider the indexing.
    # U has shape (n*m, rank) = (36, rank), indexed by (i, k) -> i*m+k
    # V has shape (m*p, rank) = (36, rank), indexed by (k, j) -> k*p+j
    # W has shape (p*n, rank) = (36, rank), indexed by (j, i) -> j*n+i (trace convention)
    #
    # The decomposition satisfies:
    #   For all i, k, j: delta_{i,k,j} = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # where delta_{i,k,j} = 1 (this is the matmul tensor).
    #
    # Actually the matmul tensor is T[i,k,j] = 1 for all i,k,j (it's the identity in the
    # appropriate sense). The decomposition is:
    #   1 = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]   for all i,k,j
    #
    # This means that for each (i,k,j), the sum over r of the product of the three
    # factor entries equals 1.

    # Strassen's 7 terms for 2x2 matrix multiplication:
    # Let A = [[a00, a01], [a10, a11]] and B = [[b00, b01], [b10, b11]]
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    rank = 189
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # For each block position (i, k, j) in the 3x3 grid, and each Strassen term s (0..6):
    # We create one rank term.
    #
    # The A-block A[i,k] has entries at:
    #   (2i, 2k), (2i, 2k+1), (2i+1, 2k), (2i+1, 2k+1)
    # In flattened U index: (2i)*m + 2k, (2i)*m + 2k+1, (2i+1)*m + 2k, (2i+1)*m + 2k+1
    #
    # The B-block B[k,j] has entries at:
    #   (2k, 2j), (2k, 2j+1), (2k+1, 2j), (2k+1, 2j+1)
    # In flattened V index: (2k)*p + 2j, (2k)*p + 2j+1, (2k+1)*p + 2j, (2k+1)*p + 2j+1
    #
    # The C-block C[i,j] has entries at:
    #   (2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1)
    # In flattened W index (trace convention W[j*n+i]): (2j)*n + 2i, (2j)*n + 2i+1, (2j+1)*n + 2i, (2j+1)*n + 2i+1

    # Strassen coefficient tables:
    # For each Strassen term s, we have coefficients for A-block entries and B-block entries.
    # U coefficients (for A-block): [coeff_a00, coeff_a01, coeff_a10, coeff_a11]
    # V coefficients (for B-block): [coeff_b00, coeff_b01, coeff_b10, coeff_b11]
    # W coefficients (for C-block): [coeff_c00, coeff_c01, coeff_c10, coeff_c11]

    strassen_U = [
        [1, 0, 0, 1],   # M1: a00 + a11
        [0, 0, 1, 1],   # M2: a10 + a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00 + a01
        [-1, 0, 1, 0],  # M6: a10 - a00
        [0, 1, 0, -1],  # M7: a01 - a11
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: b00 + b11
        [1, 0, 0, 0],   # M2: b00
        [0, 1, 0, -1],  # M3: b01 - b11
        [0, 0, 1, -1],  # M4: b10 - b00
        [0, 0, 0, 1],   # M5: b11
        [1, 1, 0, 0],   # M6: b00 + b01
        [0, 0, 1, 1],   # M7: b10 + b11
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00 += 1, C11 += 1
        [0, 0, 1, -1],  # M2: C10 += 1, C11 -= 1
        [0, 1, 0, 1],   # M3: C01 += 1, C11 += 1
        [1, 0, 1, 0],   # M4: C00 += 1, C10 += 1
        [-1, 1, 0, 0],  # M5: C00 -= 1, C01 += 1
        [0, 0, 0, 1],   # M6: C11 += 1
        [1, 0, 0, 0],   # M7: C00 += 1
    ]

    idx = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    # A-block entry indices in U (flattened)
                    a00 = (2*i) * m + 2*k
                    a01 = (2*i) * m + 2*k + 1
                    a10 = (2*i+1) * m + 2*k
                    a11 = (2*i+1) * m + 2*k + 1

                    # B-block entry indices in V (flattened)
                    b00 = (2*k) * p + 2*j
                    b01 = (2*k) * p + 2*j + 1
                    b10 = (2*k+1) * p + 2*j
                    b11 = (2*k+1) * p + 2*j + 1

                    # C-block entry indices in W (flattened, trace convention)
                    c00 = (2*j) * n + 2*i
                    c01 = (2*j) * n + 2*i + 1
                    c10 = (2*j+1) * n + 2*i
                    c11 = (2*j+1) * n + 2*i + 1

                    # Set U coefficients
                    U[a00, idx] = strassen_U[s][0]
                    U[a01, idx] = strassen_U[s][1]
                    U[a10, idx] = strassen_U[s][2]
                    U[a11, idx] = strassen_U[s][3]

                    # Set V coefficients
                    V[b00, idx] = strassen_V[s][0]
                    V[b01, idx] = strassen_V[s][1]
                    V[b10, idx] = strassen_V[s][2]
                    V[b11, idx] = strassen_V[s][3]

                    # Set W coefficients
                    W[c00, idx] = strassen_W[s][0]
                    W[c01, idx] = strassen_W[s][1]
                    W[c10, idx] = strassen_W[s][2]
                    W[c11, idx] = strassen_W[s][3]

                    idx += 1

    assert idx == rank, f"Expected {rank} terms, got {idx}"

    # Verify the decomposition with a quick numerical check
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test

        U_arr = U.astype(float)
        V_arr = V.astype(float)
        W_arr = W.astype(float)

        # Reshape for efficient computation
        # U_arr: (n*m, rank) -> (n, m, rank)
        # V_arr: (m*p, rank) -> (m, p, rank)
        # W_arr: (p*n, rank) -> (p, n, rank)
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)

        # C[i,j] = sum_k sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
        # = sum_r W[j,i,r] * (sum_k U[i,k,r] * V[k,j,r])

        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                for r in range(rank):
                    inner = np.dot(U_reshaped[i, :, r], V_reshaped[:, j, r])
                    C_computed[i, j] += W_reshaped[j, i, r] * inner

        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook if verification fails
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school), dtype=int)
            V_s = np.zeros((m * p, rank_school), dtype=int)
            W_s = np.zeros((p * n, rank_school), dtype=int)

            idx_s = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U_s[i * m + k, idx_s] = 1
                        V_s[k * p + j, idx_s] = 1
                        W_s[j * n + i, idx_s] = 1
                        idx_s += 1

            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
