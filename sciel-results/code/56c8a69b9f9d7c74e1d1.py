import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for local 2x2 matrices
    # U: (i, j_local) -> coeff
    # V: (j_prime_local, k_local) -> coeff
    # W: (i, k_local) -> coeff
    
    # M1: (a11+a22)(b11+b22)
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1} # c11, c22
    
    # M2: (a21+a22)b11
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(1,0): 1, (1,1): -1} # c21, c22
    
    # M3: a11(b12-b22)
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    w3 = {(0,1): 1, (1,1): 1} # c12, c22
    
    # M4: a22(b21-b11)
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    w4 = {(0,0): -1, (1,0): 1} # c11, c21
    
    # M5: (a11+a12)b22
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): 1} # c11
    
    # M6: (a21-a11)(b11+b12)
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1} # c22
    
    # M7: (a12-a22)(b21+b22)
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): -1, (0,1): 1} # c11, c12
    
    mults = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    for s in range(8):
        part = s // 4  # 0: A_L (cols 0,1), 1: A_R (cols 2,3)
        q = s % 4      # 0..3: Block index in B (cols 2q, 2q+1)
        
        for u_dict, v_dict, w_dict in mults:
            # Fill U
            for (i, j_loc), coeff in u_dict.items():
                j_global = 2 * part + j_loc
                u_idx = i * m + j_global
                U[u_idx][col] += coeff
            
            # Fill V
            for (j_prime_loc, k_loc), coeff in v_dict.items():
                if part == 0:
                    j_global = j_prime_loc # Rows 0,1
                else:
                    j_global = 2 + j_prime_loc # Rows 2,3
                k_global = 2 * q + k_loc
                v_idx = j_global * p + k_global
                V[v_idx][col] += coeff
            
            # Fill W
            for (i, k_loc), coeff in w_dict.items():
                k_global = 2 * q + k_loc
                w_idx = k_global * n + i
                W[w_idx][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
