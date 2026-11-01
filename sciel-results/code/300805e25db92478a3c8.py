import numpy as np

def construct():
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # u: (i, j), v: (j, k), w: (k, i)
    # indices 0..3 for u, v, w local
    # 0:00, 1:01, 2:10, 3:11
    
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b01)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    factors = []
    # (U_coeffs, V_coeffs, W_coeffs)
    # U: [a00, a01, a10, a11]
    # V: [b00, b01, b10, b11]
    # W: [c00, c01, c10, c11]
    
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])) # m1
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1])) # m2
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1])) # m3
    factors.append(([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0])) # m4
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0])) # m5
    factors.append(([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1])) # m6
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])) # m7
    
    for q in range(3):
        for t in range(7):
            c = 7 * q + t
            u_coeff, v_coeff, w_coeff = factors[t]
            
            # Map local indices to global
            # U: i in 0..1, j in 0..1. u_idx = i*2 + j
            for li in range(2):
                for lj in range(2):
                    u_idx = li * 2 + lj
                    U[u_idx][c] = u_coeff[li * 2 + lj]
                    
            # V: j in 0..1, k in 0..1 (local). v_idx = j*6 + k_global
            # k_global = 2*q + lk
            for lj in range(2):
                for lk in range(2):
                    v_idx = lj * 6 + (2 * q + lk)
                    V[v_idx][c] = v_coeff[lj * 2 + lk]
                    
            # W: k in 0..1 (local), i in 0..1. w_idx = k_global*2 + li
            # k_global = 2*q + lk
            for lk in range(2):
                for li in range(2):
                    w_idx = (2 * q + lk) * 2 + li
                    W[w_idx][c] = w_coeff[lk * 2 + li]
                    
    return {"U": U, "V": V, "W": W}
