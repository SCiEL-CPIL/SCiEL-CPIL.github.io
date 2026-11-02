"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition of 2x2x2 tensor
    # Each entry is (u_list, v_list, w_list)
    # u_list: [(i, j, coeff), ...]
    # v_list: [(j, k, coeff), ...]
    # w_list: [(i, k, coeff), ...]
    factors = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_list, v_list, w_list = factors[s]
            
            for i, j, c in u_list:
                U[i*m + j][col] = c
                
            for j, k, c in v_list:
                V[j*p + 2*t + k][col] = c
                
            for i, k, c in w_list:
                W[(2*t + k)*n + i][col] = c
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
