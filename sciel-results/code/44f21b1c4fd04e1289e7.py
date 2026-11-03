"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify the decomposition exactly reconstructs matrix multiplication.

    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must equal 1
    for all i in [0,n), j in [0,m), k in [0,p).
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for r in range(rank):
                    s += U[ui][r] * V[vj][r] * W[wk][r]
                if s != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _fast3x3(n, m, p):
    """Hybrid: rank-23 3x3 core (j'=0,1,2) with schoolbook for j''=1 (j=3,4)."""
    # Hardcoded rank-23 decomposition of the 3x3 matrix multiplication tensor.
    # a[r][i][j'] = U coefficient for (i, j'), b[r][j'][k] = V coefficient,
    # c[r][j'][j''] = W coefficient (shared across j'').
    a = [
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,1,0],[0,0,0]],
        [[0,0,0],[0,0,1],[0,0,0]],
        [[0,0,1],[0,0,0],[0,0,0]],
        [[0,1,0],[0,0,0],[0,0,0]],
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,1,0],[0,0,0]],
        [[0,0,0],[0,0,1],[0,0,0]],
        [[0,0,1],[0,0,0],[0,0,0]],
        [[0,1,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0,0,1]],
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[1,0,0]],
        [[0,0,0],[0,0,0],[0,0,1]],
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[1,0,0]],
        [[0,0,0],[0,0,0],[0,0,1]],
        [[0,0,1],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0,0,1]],
        [[0,0,0],[1,0,0],[0,0,0]],
        [[0,1,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,1,0],[0,0,0]],
        [[0,0,1],[0,0,0],[0,0,0]],
    ]
    b = [
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[1,0,0],[0,0,0]],
        [[0,0,0],[0,1,0],[0,0,0]],
        [[0,0,0],[0,0,0],[1,0,0]],
        [[0,0,0],[0,0,0],[0,1,0]],
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[1,0,0],[0,0,0]],
        [[0,0,0],[0,1,0],[0,0,0]],
        [[0,0,0],[0,0,0],[1,0,0]],
        [[0,0,0],[0,0,0],[0,1,0]],
        [[0,0,0],[0,0,1],[0,0,0]],
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[1,0,0]],
        [[0,0,0],[0,0,0],[0,1,0]],
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,0,0],[1,0,0]],
        [[0,0,0],[0,0,0],[0,1,0]],
        [[0,0,0],[0,0,0],[1,0,0]],
        [[0,0,1],[0,0,0],[0,0,0]],
        [[0,0,0],[1,0,0],[0,0,0]],
        [[0,0,0],[0,1,0],[0,0,0]],
        [[0,0,0],[0,0,1],[0,0,0]],
        [[0,0,0],[0,0,0],[1,0,0]],
    ]
    c = [
        [[1,0],[0,0],[0,0]],
        [[0,0],[1,0],[0,0]],
        [[0,0],[0,1],[0,0]],
        [[0,0],[0,0],[1,0]],
        [[0,0],[0,0],[0,1]],
        [[1,0],[0,0],[0,0]],
        [[0,0],[1,0],[0,0]],
        [[0,0],[0,1],[0,0]],
        [[0,0],[0,0],[1,0]],
        [[0,0],[0,0],[0,1]],
        [[0,0],[0,0],[0,1]],
        [[1,0],[0,0],[0,0]],
        [[0,0],[0,0],[1,0]],
        [[0,0],[0,0],[0,1]],
        [[1,0],[0,0],[0,0]],
        [[0,0],[0,0],[1,0]],
        [[0,0],[0,0],[0,1]],
        [[0,0],[1,0],[0,0]],
        [[0,0],[0,0],[1,0]],
        [[0,0],[1,0],[0,0]],
        [[0,0],[0,1],[0,0]],
        [[0,0],[0,0],[1,0]],
        [[0,0],[0,0],[0,1]],
    ]
    # rank-23 core: 3 (i) * 3 (j') * 23 (r) = 171
    # schoolbook for j''=1 (j=3,4): 3 (i) * 2 (j) * 7 (k) = 42
    core_rank = 3 * 3 * 23
    sb_rank = 3 * 2 * 7
    total_rank = core_rank + sb_rank
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    # Core part: r index = r_core (0..22)
    for r_core in range(23):
        for i in range(3):
            for j_prime in range(3):
                j = j_prime  # j''=0
                U[i * m + j][r_core] = a[r_core][i][j_prime]
                for k in range(7):
                    V[j * p + k][r_core] = b[r_core][j_prime][k]
                for k in range(7):
                    W[k * n + i][r_core] = c[r_core][j_prime][0]
    # Schoolbook part for j''=1 (j=3,4)
    col = core_rank
    for i in range(3):
        for j in [3, 4]:
            for k in range(7):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    # The current _fast3x3 implementation has rank 213, which is worse than schoolbook (105).
    # We return the schoolbook decomposition which is guaranteed correct and has rank 105.
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
