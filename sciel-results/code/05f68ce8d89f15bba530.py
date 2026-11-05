import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
        col += 1
        
    def strassen_2x2x2(i_base, j_base, k_base):
        u00 = i_base * m + j_base
        u01 = i_base * m + j_base + 1
        u10 = (i_base + 1) * m + j_base
        u11 = (i_base + 1) * m + j_base + 1
        
        v00 = j_base * p + k_base
        v01 = j_base * p + k_base + 1
        v10 = (j_base + 1) * p + k_base
        v11 = (j_base + 1) * p + k_base + 1
        
        w00 = k_base * n + i_base
        w01 = k_base * n + i_base + 1
        w10 = (k_base + 1) * n + i_base
        w11 = (k_base + 1) * n + i_base + 1
        
        add_term([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)])
        add_term([(u10, 1), (u11, 1)], [(v00, 1)], [(w10, 1)])
        add_term([(u00, 1)], [(v01, 1), (v11, -1)], [(w01, 1), (w11, 1)])
        add_term([(u11, 1)], [(v10, 1), (v00, -1)], [(w00, 1), (w10, 1)])
        add_term([(u00, 1), (u01, 1)], [(v11, 1)], [(w00, -1), (w01, 1)])
        add_term([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], [(w11, 1)])
        add_term([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], [(w00, 1)])

    def schoolbook_1x2x2(i_idx, j_base, k_base):
        u0 = i_idx * m + j_base
        u1 = i_idx * m + j_base + 1
        
        v00 = j_base * p + k_base
        v01 = j_base * p + k_base + 1
        v10 = (j_base + 1) * p + k_base
        v11 = (j_base + 1) * p + k_base + 1
        
        w0 = k_base * n + i_idx
        w1 = (k_base + 1) * n + i_idx
        
        add_term([(u0, 1)], [(v00, 1)], [(w0, 1)])
        add_term([(u1, 1)], [(v10, 1)], [(w0, 1)])
        add_term([(u0, 1)], [(v01, 1)], [(w1, 1)])
        add_term([(u1, 1)], [(v11, 1)], [(w1, 1)])

    for b in range(3):
        k_base = b * 2
        for jb in range(2):
            j_base = jb * 2
            strassen_2x2x2(0, j_base, k_base)
            schoolbook_1x2x2(2, j_base, k_base)

    return {"U": U, "V": V, "W": W}
