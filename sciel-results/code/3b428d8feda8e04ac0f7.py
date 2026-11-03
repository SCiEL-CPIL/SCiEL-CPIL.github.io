import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 4, 8
    
    # Strassen 7-term decomposition for 2x2 * 2x2
    # Each term: (u_coeffs, v_coeffs, w_coeffs) with local 2x2 indices
    # Local index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)],       [(2,1),(3,-1)]),
        ([(0,1)],         [(1,1),(3,-1)],[(1,1),(3,1)]),
        ([(3,1)],         [(2,1),(0,-1)],[(0,1),(2,1)]),
        ([(0,1),(1,1)],   [(3,1)],       [(0,-1),(1,1)]),
        ([(2,1),(0,-1)],  [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)],  [(2,1),(3,1)], [(0,1)]),
    ]
    
    total_rank = 56
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    r = 0
    for h in range(2):
        for j in range(4):
            for u_c, v_c, w_c in strassen:
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                
                for li, c in u_c:
                    i, ll = li // 2, li % 2
                    U[i*m + 2*h + ll][r] += c
                for li, c in v_c:
                    ll, kl = li // 2, li % 2
                    V[(2*h + ll)*p + 2*j + kl][r] += c
                for li, c in w_c:
                    i, kl = li // 2, li % 2
                    W[(2*j + kl)*n + i][r] += c
                r += 1
    
    return {"U": U, "V": V, "W": W}
