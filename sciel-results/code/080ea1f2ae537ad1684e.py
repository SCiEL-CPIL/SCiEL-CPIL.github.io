"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    ]
    
    col = 0
    for alpha, beta in terms:
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    for I in range(2):
                        for J in range(2):
                            coeff_a = alpha.get((I, J), 0)
                            if coeff_a != 0:
                                u_idx = (2 * I + i) * m + (2 * J + j)
                                U[u_idx][col] += coeff_a
                    for J in range(2):
                        for K in range(3):
                            coeff_b = beta.get((J, K), 0)
                            if coeff_b != 0:
                                v_idx = (2 * J + j) * p + (3 * K + k)
                                V[v_idx][col] += coeff_b
                    for I in range(2):
                        for K in range(3):
                            w_idx = (3 * K + k) * n + (2 * I + i)
                            W[w_idx][col] += 1
                    col += 1
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
