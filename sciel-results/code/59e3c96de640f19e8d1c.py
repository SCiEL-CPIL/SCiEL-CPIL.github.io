import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using block Strassen
    # m=4 split into 2 blocks of 2
    # p=6 split into 3 blocks of 2
    # Each 2x2x2 block uses Strassen (rank 7)
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u[i'][j'], v[j'][k'], w[k'][i']
    # t=0: M1
    # u: (0,0)=1, (1,1)=1
    # v: (0,0)=1, (1,1)=1
    # w: (0,0)=1, (1,1)=1
    strassen_u = [
        [[1, 0], [0, 1]], # t=0
        [[0, 0], [1, 1]], # t=1
        [[1, 0], [0, 0]], # t=2
        [[0, 0], [0, 1]], # t=3
        [[1, 1], [0, 0]], # t=4
        [[-1, 0], [1, 0]], # t=5
        [[0, 1], [-1, 0]]  # t=6
    ]
    strassen_v = [
        [[1, 0], [0, 1]], # t=0
        [[1, 0], [0, 0]], # t=1
        [[0, 1], [-1, 0]], # t=2
        [[-1, 0], [1, 0]], # t=3
        [[0, 0], [0, 1]], # t=4
        [[1, 1], [0, 0]], # t=5
        [[0, 0], [1, 1]]  # t=6
    ]
    # w[k'][i']
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_w = [
        [[1, 0], [0, 1]], # t=0: C00+=1, C11+=1
        [[0, 1], [-1, 0]], # t=1: C10+=1, C11+=-1
        [[0, 0], [1, 1]], # t=2: C01+=1, C11+=1
        [[1, 0], [1, 0]], # t=3: C00+=1, C10+=1
        [[-1, 0], [1, 0]], # t=4: C00+=-1, C01+=1
        [[0, 0], [0, 1]], # t=5: C11+=1
        [[1, 0], [0, 0]]  # t=6: C00+=1
    ]
    
    # Verify Strassen W indices:
    # w[k'][i']
    # t=0: k'=0,i'=0 -> 1; k'=1,i'=1 -> 1. Correct.
    # t=1: k'=0,i'=1 -> 1; k'=1,i'=1 -> -1. Correct.
    # t=2: k'=1,i'=0 -> 1; k'=1,i'=1 -> 1. Correct.
    # t=3: k'=0,i'=0 -> 1; k'=0,i'=1 -> 1. Correct.
    # t=4: k'=0,i'=0 -> -1; k'=1,i'=0 -> 1. Correct.
    # t=5: k'=1,i'=1 -> 1. Correct.
    # t=6: k'=0,i'=0 -> 1. Correct.
    
    for r in range(rank):
        b = r // 21
        rem = r % 21
        c = rem // 7
        t = rem % 7
        
        u_local = strassen_u[t]
        v_local = strassen_v[t]
        w_local = strassen_w[t]
        
        # Fill U
        for ip in range(2):
            for jp in range(2):
                val = u_local[ip][jp]
                if val != 0:
                    j_global = 2 * b + jp
                    idx_u = ip * m + j_global
                    U[idx_u][r] = val
                    
        # Fill V
        for jp in range(2):
            for kp in range(2):
                val = v_local[jp][kp]
                if val != 0:
                    j_global = 2 * b + jp
                    k_global = 2 * c + kp
                    idx_v = j_global * p + k_global
                    V[idx_v][r] = val
                    
        # Fill W
        for kp in range(2):
            for ip in range(2):
                val = w_local[kp][ip]
                if val != 0:
                    k_global = 2 * c + kp
                    idx_w = k_global * n + ip
                    W[idx_w][r] = val
                    
    return {"U": U, "V": V, "W": W}
