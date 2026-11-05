"""
Rank-55 exact tensor decomposition for 2x3 * 3x10 matrix multiplication.

Strategy:
  Split the middle dimension m=3 into a 2+1 partition.
  - The 2x2 sub-block (A[0:2,0:2] * B[0:2,:]) is handled per 2-column block
    using the rank-7 Strassen algorithm (5 blocks -> 35 rank terms).
  - The 1x1 remainder (A[:,2] * B[2,:]) is handled with schoolbook rank 20.
  Total rank: 35 + 20 = 55.

  This matches the best known score (1.0909) and is verified correct.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds, with margin before 150s kill


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Strassen coefficients for 2x2 * 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # L vectors: coefficients on A entries (local indices 0=a00, 1=a01, 2=a10, 3=a11)
    L = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1}       # M7
    ]

    # R vectors: coefficients on B entries (local indices 0=b00, 1=b01, 2=b10, 3=b11)
    R = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1}        # M7
    ]

    # W vectors: coefficients on C entries (local indices 0=c00, 1=c01, 2=c10, 3=c11)
    W_coeffs = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: -1},      # M2
        {1: 1, 3: 1},       # M3
        {0: 1, 2: 1},       # M4
        {1: 1, 0: -1},       # M5
        {3: 1},             # M6
        {0: 1}              # M7
    ]

    # Fill 5 Strassen blocks (each covers 2 columns of B)
    for block in range(5):
        k_offset = block * 2
        for s in range(7):
            r = col
            # Fill U: local idx -> global U index i*m+j
            for local_idx, val in L[s].items():
                i = local_idx // 2
                j = local_idx % 2
                U[i * m + j][r] = val

            # Fill V: local idx -> global V index j*p+k
            for local_idx, val in R[s].items():
                j = local_idx // 2
                k = local_idx % 2
                V[j * p + (k_offset + k)][r] = val

            # Fill W: local idx -> global W index k*n+i
            for local_idx, val in W_coeffs[s].items():
                i = local_idx // 2
                k = local_idx % 2
                W[(k_offset + k) * n + i][r] = val

            col += 1

    # Schoolbook part: A[:,2] * B[2,:] contributes 20 rank-1 terms
    for i in range(2):
        for k in range(10):
            r = col
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            col += 1

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

    # Verify correctness
    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]

        A_weighted = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_weighted[i, j] = u_r[i * m + j]

        B_weighted = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_weighted[j, k] = v_r[j * p + k]

        prod_r = np.sum(A_weighted * A) * np.sum(B_weighted * B)

        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += prod_r * w_r[k * n + i]

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
