"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    _start_time = time.time()
    _deadline = 120.0
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Part 1: 2x1x6 -> 12
    # Part 2: 2x4x6 -> 2 * Rank(2x2x6)
    # Rank(2x2x6) using 3 Strassen blocks -> 3 * 7 = 21
    # Total Rank = 12 + 2 * 21 = 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j in 1..4, split into two blocks of 2x2x6
    strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)],          [(1,0,1), (1,1,-1)]),
        ([(0,0,1)],          [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)],          [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)],          [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)],[(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)],[(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    for block_idx in range(2):
        j_base = 1 + block_idx * 2
        for s in range(3):
            k_base = s * 2
            for term in strassen_terms:
                u_list, v_list, w_list = term
                for (li, lj, val) in u_list:
                    U[li * m + j_base + lj][col] += val
                for (lj, lk, val) in v_list:
                    V[(j_base + lj) * p + k_base + lk][col] += val
                for (li, lk, val) in w_list:
                    W[(k_base + lk) * n + li][col] += val
                col += 1
                
    if time.time() > _deadline - 5.0:
        # Safety margin: return what we have
        pass
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
