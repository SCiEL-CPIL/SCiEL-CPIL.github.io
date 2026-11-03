import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100 # 100s safety margin
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: Strassen for 4x4 x 4x4 (rank 49)
    # Part 2: Schoolbook for 4x4 x 4x1 (rank 16)
    
    total_rank = 49 + 16
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: Strassen ---
    
    # Define 2x2 Strassen terms
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])  # M7
    ]
    
    col = 0
    for k in range(7):
        a_blocks = [0, 0, 0, 0]
        b_blocks = [0, 0, 0, 0]
        
        if k == 0: # M1
            a_blocks = [1, 0, 0, 1]
            b_blocks = [1, 0, 0, 1]
        elif k == 1: # M2
            a_blocks = [0, 0, 1, 1]
            b_blocks = [1, 0, 0, 0]
        elif k == 2: # M3
            a_blocks = [1, 0, 0, 0]
            b_blocks = [0, 1, 0, -1]
        elif k == 3: # M4
            a_blocks = [0, 0, 0, 1]
            b_blocks = [0, 0, 1, -1]
        elif k == 4: # M5
            a_blocks = [1, 1, 0, 0]
            b_blocks = [0, 0, 0, 1]
        elif k == 5: # M6
            a_blocks = [-1, 0, 1, 0]
            b_blocks = [1, 1, 0, 0]
        elif k == 6: # M7
            a_blocks = [0, 1, 0, -1]
            b_blocks = [0, 0, 1, 1]
            
        for s in range(7):
            x_s, y_s, z_s = strassen_terms[s]
            
            def local_idx(r, c):
                return r * 2 + c
            
            for b in range(4):
                if a_blocks[b] == 0:
                    continue
                I = b // 2
                J = b % 2
                for lr in range(2):
                    for lc in range(2):
                        coeff = a_blocks[b] * x_s[local_idx(lr, lc)]
                        if coeff != 0:
                            gi = I * 2 + lr
                            gj = J * 2 + lc
                            u_idx = gi * 4 + gj
                            U[u_idx][col] += coeff
                            
            for b in range(4):
                if b_blocks[b] == 0:
                    continue
                J = b // 2
                K = b % 2
                for lr in range(2):
                    for lc in range(2):
                        coeff = b_blocks[b] * y_s[local_idx(lr, lc)]
                        if coeff != 0:
                            gj = J * 2 + lr
                            gk = K * 2 + lc
                            v_idx = gj * 4 + gk
                            V[v_idx][col] += coeff
                            
            c_targets = []
            if k == 0: # M1
                c_targets = [(0, 0, 1), (1, 1, 1)]
            elif k == 1: # M2
                c_targets = [(1, 0, 1), (1, 1, -1)]
            elif k == 2: # M3
                c_targets = [(0, 1, 1), (1, 1, 1)]
            elif k == 3: # M4
                c_targets = [(0, 0, 1), (1, 0, 1)]
            elif k == 4: # M5
                c_targets = [(0, 0, -1), (0, 1, 1)]
            elif k == 5: # M6
                c_targets = [(1, 1, 1)]
            elif k == 6: # M7
                c_targets = [(0, 0, 1), (1, 1, 1)]
                
            for lr in range(2):
                for lc in range(2):
                    z_coeff = z_s[local_idx(lr, lc)]
                    if z_coeff == 0:
                        continue
                    for (cI, cJ, sign) in c_targets:
                        coeff = sign * z_coeff
                        if coeff == 0:
                            continue
                        gi = cI * 2 + lr
                        gk = cJ * 2 + lc
                        w_idx = gk * 4 + gi
                        W[w_idx][col] += coeff
                        
            col += 1
            
    # --- Part 2: Schoolbook ---
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = 16 + j
            w_idx = 16 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
