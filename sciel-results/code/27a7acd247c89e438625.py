import time
import sys

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # t: 0 to 6
    # u[i, j, t], v[j, k, t], w[k, i, t]
    # Using 0-based indices for local i, j, k
    
    # M1: (A00+A11)(B00+B11)
    # w: C00+=1, C01+=1, C10+=1, C11+=1
    u_coeffs = [
        [(1, 0, 1), (1, 1, 1)],  # t=0
        [(1, 0, 1), (1, 1, 1)],  # t=1
        [(0, 0, 1)],             # t=2
        [(1, 1, 1)],             # t=3
        [(0, 0, 1), (0, 1, 1)],  # t=4
        [(1, 0, 1), (0, 0, -1)], # t=5
        [(0, 1, 1), (1, 1, -1)]  # t=6
    ]
    
    v_coeffs = [
        [(0, 0, 1), (1, 1, 1)],  # t=0
        [(0, 0, 1)],             # t=1
        [(0, 1, 1), (1, 1, -1)], # t=2
        [(1, 0, 1), (0, 0, -1)], # t=3
        [(1, 1, 1)],             # t=4
        [(0, 0, 1), (0, 1, 1)],  # t=5
        [(1, 0, 1), (1, 1, 1)]   # t=6
    ]
    
    w_coeffs = [
        [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)], # t=0
        [(1, 0, 1), (1, 1, -1)],                       # t=1
        [(0, 1, 1), (1, 1, 1)],                        # t=2
        [(0, 0, 1), (1, 0, 1)],                        # t=3
        [(0, 0, -1), (0, 1, 1)],                       # t=4
        [(1, 1, 1)],                                   # t=5
        [(0, 0, 1)]                                    # t=6
    ]
    
    # Fill U, V, W
    # U depends on i, j, t. Same for all blocks b.
    # V depends on j, k, t. k is global 2b + k_local.
    # W depends on k, i, t. k is global 2b + k_local.
    
    for b in range(3):
        for t in range(7):
            global_t = 7 * b + t
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    # Check if (i,j) is in u_coeffs[t]
                    val = 0
                    for (ii, jj, coeff) in u_coeffs[t]:
                        if ii == i and jj == j:
                            val += coeff
                    U[i * m + j][global_t] = val
            
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    val = 0
                    for (jj, kk, coeff) in v_coeffs[t]:
                        if jj == j and kk == k_local:
                            val += coeff
                    V[j * p + k_global][global_t] = val
            
            # Fill W
            for k_local in range(2):
                k_global = 2 * b + k_local
                for i in range(2):
                    val = 0
                    for (kk, ii, coeff) in w_coeffs[t]:
                        if kk == k_local and ii == i:
                            val += coeff
                    W[k_global * n + i][global_t] = val

    return {"U": U, "V": V, "W": W}
