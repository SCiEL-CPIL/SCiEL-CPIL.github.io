"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Part 1: j=0 -> 2x1x6 -> rank 12
    # Part 2: j=1,2 -> 2x2x6 -> rank 21 (3 Strassen blocks)
    # Part 3: j=3,4 -> 2x2x6 -> rank 21 (3 Strassen blocks)
    # Total Rank = 12 + 21 + 21 = 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j = 0
    for i in range(n):
        for k in range(p):
            U[i * m][col] = 1
            V[k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Strassen terms for 2x2x2
    # u: (li, lj, val), v: (lj, lk, val), w: (li, lk, val)
    strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    # Part 2 & 3: j in {1,2} and {3,4}
    for block_idx in range(2):
        j_base = 1 + block_idx * 2
        for s in range(3):
            k_base = s * 2
            for term in strassen_terms:
                u_list, v_list, w_list = term
                for (li, lj, val) in u_list:
                    U[li * m + (j_base + lj)][col] += val
                for (lj, lk, val) in v_list:
                    V[(j_base + lj) * p + (k_base + lk)][col] += val
                for (li, lk, val) in w_list:
                    W[(k_base + lk) * n + li][col] += val
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
