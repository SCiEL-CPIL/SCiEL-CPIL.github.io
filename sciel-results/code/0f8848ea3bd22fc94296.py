import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition
    strassen_2x2 = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),
        ((0, 0, 0, 1), (0, -1, 1, 0), (1, 0, 1, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),
        ((0, -1, 0, 1), (0, 0, 1, 1), (1, 0, 0, 0))
    ]
    
    outer_a = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 1]
    ]
    
    outer_b = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    U_4x4 = []
    V_4x4 = []
    W_4x4 = []
    
    for k_out in range(7):
        if time.time() > deadline:
            # Fallback to schoolbook if time runs out (unlikely)
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
            
        a_coeff = outer_a[k_out]
        b_coeff = outer_b[k_out]
        
        for u_in, v_in, w_in in strassen_2x2:
            u_full = [0] * 16
            v_full = [0] * 16
            w_full = [0] * 16
            
            for b in range(4):
                coeff = u_in[b]
                if coeff == 0: continue
                r_op = b // 2
                c_op = b % 2
                for blk in range(4):
                    if a_coeff[blk] == 0: continue
                    r_4x4 = (blk // 2) * 2 + r_op
                    c_4x4 = (blk % 2) * 2 + c_op
                    idx = r_4x4 * 4 + c_4x4
                    u_full[idx] += coeff * a_coeff[blk]
            
            for b in range(4):
                coeff = v_in[b]
                if coeff == 0: continue
                r_op = b // 2
                c_op = b % 2
                for blk in range(4):
                    if b_coeff[blk] == 0: continue
                    r_4x4 = (blk // 2) * 2 + r_op
                    c_4x4 = (blk % 2) * 2 + c_op
                    idx = r_4x4 * 4 + c_4x4
                    v_full[idx] += coeff * b_coeff[blk]
            
            for b in range(4):
                coeff = w_in[b]
                if coeff == 0: continue
                r_op = b // 2
                c_op = b % 2
                for dr in range(2):
                    for dc in range(2):
                        r_4x4 = r_op * 2 + dr
                        c_4x4 = c_op * 2 + dc
                        idx = c_4x4 * 4 + r_4x4
                        w_full[idx] += coeff
            
            U_4x4.append(u_full)
            V_4x4.append(v_full)
            W_4x4.append(w_full)
            
    U_full = []
    V_full = []
    W_full = []
    
    for u, v, w in zip(U_4x4, V_4x4, W_4x4):
        v1 = v[:]
        v2 = [0] * 32
        for idx, val in enumerate(v):
            j = idx // 4
            k = idx % 4
            new_idx = j * 8 + (k + 4)
            v2[new_idx] = val
        
        U_full.append(u)
        V_full.append(v1)
        W_full.append(w)
        
        U_full.append(u)
        V_full.append(v2)
        W_full.append(w)
        
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
