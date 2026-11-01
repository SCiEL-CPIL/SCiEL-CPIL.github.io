import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 10.0  # seconds, safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 13

    # Direct rank-52 decomposition: 4 terms per column
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # A index: i*m+j = 2*i+j
    # B index: j*p+k = 13*j+k
    # C index (W): k*n+i = 2*k+i

    rank = 4 * p  # 52
    U = [[0] * rank for _ in range(n * m)]  # 4 x 52
    V = [[0] * rank for _ in range(m * p)]  # 26 x 52
    W = [[0] * rank for _ in range(p * n)]  # 26 x 52

    r = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                # Term: A[i,j] * B[j,k] -> C[i,k]
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1

        if time.time() - start_time > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
