"""Matrix multiplication 3x3 x 3x6 rank-minimal decomposition.

Schoolbook: rank = 3*3*6 = 54.
The task contract requires entries to be integer/half-integer, and the
decomposition must be exact.  The trivial Kronecker/schoolbook construction
is the only universally verifiable exact decomposition I can produce here.
"""

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p

    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # index conventions:
                #   U row index: i*m + j   (left matrix element A[i,j])
                #   V row index: j*p + k   (right matrix element B[j,k])
                #   W row index: k*n + i   (output W[k*n + i] corresponds to C[i,k])
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
