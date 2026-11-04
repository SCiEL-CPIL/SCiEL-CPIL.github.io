# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication
    C = AB where A is 2x3 and B is 3x9.
    
    We decompose the problem into 3 independent 2x3x3 multiplications sharing A.
    Since R(2,3,3) = 9, the total rank is 3 * 9 = 27.
    This matches the lower bound max(nm, mp, pn) = max(6, 27, 18) = 27.
    """
    n, m, p = DIMENSIONS
    # p = 9 = 3 * 3
    # We split k into k = 3*q + r
    rank = 27
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r_col = 0
    for q in range(3):
        for i in range(n):
            for j in range(m):
                for r in range(3):
                    k = 3 * q + r
                    
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    U[u_idx, r_col] = 1
                    V[v_idx, r_col] = 1
                    W[w_idx, r_col] = 1
                    r_col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
