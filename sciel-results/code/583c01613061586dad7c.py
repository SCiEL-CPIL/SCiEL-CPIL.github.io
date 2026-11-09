import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin from 120s limit
    
    n, m, p = 2, 4, 8
    
    # Pre-computed Strassen coefficients for 2x2 matrix multiplication
    # M_{<2,2,2>} rank 7
    X_forms = [
        [1, 0, 0, 1],   # m0
        [0, 0, 1, 1],   # m1
        [1, 0, 0, 0],   # m2
        [0, 0, 0, 1],   # m3
        [1, 1, 0, 0],   # m4
        [-1, 0, 1, 0],  # m5
        [0, 1, 0, -1],  # m6
    ]
    
    Y_forms = [
        [1, 0, 0, 1],   # m0
        [1, 0, 0, 0],   # m1
        [0, 1, 0, -1],  # m2
        [-1, 0, 1, 0],  # m3
        [0, 0, 0, 1],   # m4
        [1, 1, 0, 0],   # m5
        [0, 0, 1, 1],   # m6
    ]
    
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],   # z00
        [0, 0, 1, 0, 1, 0, 0],    # z01
        [0, 1, 0, 1, 0, 0, 0],    # z10
        [1, -1, 1, 0, 0, 1, 0],   # z11
    ]
    
    num_groups = 4
    num_ablocks = 2
    num_strassen = 7
    rank = num_groups * num_ablocks * num_strassen
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx][r] = X_forms[s][i * 2 + jp]
                
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx][r] = Y_forms[s][jp * 2 + kp]
                
                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx][r] = Z_combos[i * 2 + kp][s]
                
                r += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == '__main__':
    res = construct()
    print("Constructed successfully.")
