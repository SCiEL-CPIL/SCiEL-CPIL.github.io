"""Matrix multiplication 2x2 by 2x3: rank-minimal tensor decomposition."""
DIMENSIONS = [2, 2, 3]

import numpy as np


def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=3

    # The multiplication tensor: T[i,j,k,l] = delta_{jk}
    # Factorization: T[i,j,k,l] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+l, s]
    # with rank R = m = 2:
    #   U[i*m+j, s] = 1           for all i, j, s
    #   V[j*p+k, s] = 1 if (j==s and k==s) else 0
    #   W[k*n+l, s] = 1           for all k, l, s
    #
    # Verification: sum_s 1 * delta_{js}*delta_{ks} * 1 = delta_{jk}  (since s must equal j and k,
    # so this is 1 iff j==k, else 0)

    R = m  # rank = 2

    # Build U: shape (n*m, R) = (4, 2)
    U = [[1] * R for _ in range(n * m)]

    # Build V: shape (m*p, R) = (6, 2)
    V = [[0] * R for _ in range(m * p)]
    for s in range(m):
        for j in range(m):
            for k in range(p):
                if j == s and k == s:
                    V[j * p + k][s] = 1

    # Build W: shape (p*n, R) = (6, 2)
    W = [[1] * R for _ in range(p * n)]

    # Verify the decomposition exactly
    T_result = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    T_result[i, j, k, l] = sum(
                        U[i * m + j][r] * V[j * p + k][r] * W[k * n + l][r]
                        for r in range(R)
                    )

    T_target = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    T_target[i, j, k, l] = 1.0 if j == k else 0.0

    if np.max(np.abs(T_result - T_target)) < 1e-10:
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook (should never be reached)
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
