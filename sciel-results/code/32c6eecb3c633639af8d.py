"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Rank 189 decomposition: 3x3 blocks of 2x2, Strassen for each 2x2
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # l: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {(r,s): coeff} for X
    # v_coeffs: dict {(s',t): coeff} for Y
    # w_coeffs: dict {(r,t): coeff} for Z
    strassen_terms = [
        # l=0
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # l=1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # l=2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # l=3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # l=4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # l=5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # l=6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    for alpha in range(3):
        for beta in range(3):
            for gamma in range(3):
                for l in range(7):
                    u_coeffs, v_coeffs, w_coeffs = strassen_terms[l]
                    
                    # Update U
                    for (r, s), coeff in u_coeffs.items():
                        i = 2 * alpha + r
                        j = 2 * beta + s
                        U[i * m + j][col] += coeff
                    
                    # Update V
                    for (s_prime, t), coeff in v_coeffs.items():
                        j = 2 * beta + s_prime
                        k = 2 * gamma + t
                        V[j * p + k][col] += coeff
                    
                    # Update W
                    for (r, t), coeff in w_coeffs.items():
                        i = 2 * alpha + r
                        k = 2 * gamma + t
                        W[k * n + i][col] += coeff
                    
                    col += 1
    
    if time.time() > deadline:
        # Should not happen for this static construction, but good practice
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
