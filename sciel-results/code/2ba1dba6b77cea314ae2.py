"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(b, r_start):
        if time.time() - start_time > 140:
            return
        k0 = 2 * b
        terms = [
            ([(0,0), (1,1)], [1, 1], [(0, k0), (1, k0+1)], [1, 1], [(k0, 0), (k0+1, 1)], [1, 1]),
            ([(1,0), (1,1)], [1, 1], [(0, k0)], [1], [(k0, 1), (k0+1, 1)], [1, -1]),
            ([(0,0)], [1], [(0, k0+1), (1, k0+1)], [1, -1], [(k0+1, 0), (k0+1, 1)], [1, 1]),
            ([(1,1)], [1], [(1, k0), (0, k0)], [1, -1], [(k0, 0), (k0, 1)], [1, 1]),
            ([(0,0), (0,1)], [1, 1], [(1, k0+1)], [1], [(k0, 0), (k0+1, 0)], [-1, 1]),
            ([(1,0), (0,0)], [1, -1], [(0, k0), (0, k0+1)], [1, 1], [(k0+1, 1)], [1]),
            ([(0,1), (1,1)], [1, -1], [(1, k0), (1, k0+1)], [1, 1], [(k0, 0)], [1]),
        ]
        
        for t_idx, (u_idx, u_c, v_idx, v_c, w_idx, w_c) in enumerate(terms):
            r = r_start + t_idx
            for (i, j), c in zip(u_idx, u_c):
                U[i * m + j][r] += c
            for (j, k), c in zip(v_idx, v_c):
                V[j * p + k][r] += c
            for (k, i), c in zip(w_idx, w_c):
                W[k * n + i][r] += c

    for b in range(5):
        add_strassen_block(b, b * 7)
        
    r = 35
    for i in range(n):
        U[i * m + 2][r] = 1
    for k in range(p):
        V[2 * p + k][r] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
