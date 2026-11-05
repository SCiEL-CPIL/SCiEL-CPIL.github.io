import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    strassen_a = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 1],
    ]
    strassen_b = [
        [1, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
    ]
    strassen_c = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    U = []
    V = []
    W = []
    
    m_groups = [[0, 1], [2, 3]]
    p_groups = [[0, 1], [2, 3], [4, 5]]
    
    for jg in m_groups:
        for kg in p_groups:
            for s in range(7):
                u_global = [0] * (n * m)
                for li in range(2):
                    for lj in range(2):
                        u_global[li * m + jg[lj]] = strassen_a[s][li * 2 + lj]
                
                v_global = [0] * (m * p)
                for lj in range(2):
                    for lk in range(2):
                        v_global[jg[lj] * p + kg[lk]] = strassen_b[s][lj * 2 + lk]
                
                w_global = [0] * (p * n)
                for lk in range(2):
                    for li in range(2):
                        w_global[kg[lk] * n + li] = strassen_c[s][lk * 2 + li]
                
                U.append(u_global)
                V.append(v_global)
                W.append(w_global)
            
            for r in range(2):
                u_global = [0] * (n * m)
                u_global[2 * m + jg[r]] = 1
                
                v_global = [0] * (m * p)
                for lk in range(2):
                    v_global[jg[r] * p + kg[lk]] = 1
                
                w_global = [0] * (p * n)
                for lk in range(2):
                    w_global[kg[lk] * n + 2] = 1
                
                U.append(u_global)
                V.append(v_global)
                W.append(w_global)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
