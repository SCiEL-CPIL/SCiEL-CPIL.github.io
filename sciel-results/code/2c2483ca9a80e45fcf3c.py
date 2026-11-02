import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = p * n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for k in range(p):
        for i in range(n):
            # U index: i * m + j
            # V index: j * p + k
            # W index: k * n + i
            for j in range(m):
                if i == i: # Always true
                    U[i * m + j][r] = 1
                # V depends on k
                # V[j * p + k][r] should be 1 if k matches
                # But V index is j * p + k.
                # We need V[j * p + k'][r] = 1 if k' == k_r.
                # So V[j * p + k][r] = 1 for all j.
                V[j * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
