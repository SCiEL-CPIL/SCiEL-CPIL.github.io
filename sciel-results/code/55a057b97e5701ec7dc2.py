import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for u_idx in range(n * m):
        for v_idx in range(m * p):
            for w_idx in range(p * n):
                val = sum(U[u_idx][r] * V[v_idx][r] * W[w_idx][r] for r in range(rank))
                i = u_idx // m
                j = u_idx % m
                j2 = v_idx // p
                k = v_idx % p
                k2 = w_idx // n
                i2 = w_idx % n
                if i == i2 and j == j2 and k == k2:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    def strassen_decomposition():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        U_strassen = [
            {(0, 0): 1, (1, 1): 1},
            {(1, 0): 1, (1, 1): 1},
            {(0, 0): 1},
            {(1, 1): 1},
            {(0, 0): 1, (0, 1): 1},
            {(1, 0): 1, (0, 0): -1},
            {(0, 1): 1, (1, 1): -1}
        ]
        V_strassen = [
            {(0, 0): 1, (1, 1): 1},
            {(0, 0): 1},
            {(0, 1): 1, (1, 1): -1},
            {(1, 0): 1, (0, 0): -1},
            {(1, 1): 1},
            {(0, 0): 1, (0, 1): 1},
            {(1, 0): 1, (1, 1): 1}
        ]
        W_strassen = [
            {(0, 0): 1, (1, 1): 1},
            {(0, 1): 1, (1, 1): -1},
            {(1, 0): 1, (1, 1): 1},
            {(0, 0): 1, (0, 1): 1},
            {(0, 0): -1, (1, 0): 1},
            {(1, 1): 1},
            {(0, 0): 1}
        ]
        
        col = 0
        j_groups = [(0, 1), (2, 3)]
        k_groups = [(0, 1), (2, 3), (4, 5)]
        
        for jg in j_groups:
            j_base = jg[0]
            for kg in k_groups:
                k_base = kg[0]
                for s in range(7):
                    for (i_loc, j_loc), coeff in U_strassen[s].items():
                        i = i_loc
                        j = j_base + j_loc
                        u_idx = i * m + j
                        U[u_idx][col] += coeff
                    for (j_loc, k_loc), coeff in V_strassen[s].items():
                        j = j_base + j_loc
                        k = k_base + k_loc
                        v_idx = j * p + k
                        V[v_idx][col] += coeff
                    for (k_loc, i_loc), coeff in W_strassen[s].items():
                        k = k_base + k_loc
                        i = i_loc
                        w_idx = k * n + i
                        W[w_idx][col] += coeff
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Directly return the Strassen rank-42 decomposition
    result = strassen_decomposition()
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        raise ValueError("Invalid decomposition")
    return result
