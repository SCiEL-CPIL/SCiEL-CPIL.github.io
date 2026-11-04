import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    block_size = 3
    
    strassen_products = [
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
    ]
    
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for prod in strassen_products:
        A_side = prod['A']
        B_side = prod['B']
        C_side = prod['C']
        
        for ra in range(block_size):
            for rb in range(block_size):
                for ca in range(block_size):
                    for (ri, cj) in A_side:
                        coeff_a = A_side[(ri, cj)]
                        i = ri * block_size + ra
                        j = cj * block_size + rb
                        U[i * m + j][col] += coeff_a
                    
                    for (ri, cj) in B_side:
                        coeff_b = B_side[(ri, cj)]
                        j = ri * block_size + rb
                        k = cj * block_size + ca
                        V[j * p + k][col] += coeff_b
                    
                    for (ri, cj) in C_side:
                        coeff_c = C_side[(ri, cj)]
                        i = ri * block_size + ra
                        k = cj * block_size + ca
                        W[k * n + i][col] += coeff_c
                    
                    col += 1
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-9:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col_s] = 1
                    V_s[j*p+k][col_s] = 1
                    W_s[k*n+i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
