import time

def construct():
    start_time = time.time()
    N, M, P = 4, 4, 5
    terms = []
    
    def add_term(n, m, p, off_n, off_m, off_p, u_sp, v_sp, w_sp):
        gu = []
        for idx, c in u_sp:
            i, j = idx // m, idx % m
            gidx = (off_n + i) * M + (off_m + j)
            gu.append((gidx, c))
        gv = []
        for idx, c in v_sp:
            j, k = idx // p, idx % p
            gidx = (off_m + j) * P + (off_p + k)
            gv.append((gidx, c))
        gw = []
        for idx, c in w_sp:
            i, k = idx % n, idx // n
            gidx = (off_p + k) * N + (off_n + i)
            gw.append((gidx, c))
        terms.append((gu, gv, gw))

    def solve(n, m, p, off_n, off_m, off_p):
        if n == 2 and m == 2 and p == 2:
            # Strassen
            # M1
            u = [(0, 1), (3, 1)]
            v = [(0, 1), (3, 1)]
            w = [(0, 1), (3, 1)]
            add_term(n, m, p, off_n, off_m, off_p, u, v, w)
            
            # M2
            u = [(2, 1), (3, 1)]
            v = [(0, 1)]
            w = [(2, 1)] # C10
            add_term(n, m, p, off_n, off_m, off_p, u, v, w)
            
            # M3
            u = [(0, 1)]
            v = [(2, 1), (3, -1)]
            w = [(1, 1)] # C01
            add_term(n, m, p, off_n, off_m, off_p, u, v, w)
            
            # M4
            u = [(3, 1)]
            v = [(1, 1), (0, -1)]
            w = [(0, 1), (2, 1)] # C00, C10
            add_term(n, m, p, off_n, off_m, off_p, u, v, w)
            
            # M5
            u = [(0, 1), (1, 1)]
            v = [(3, 1)]
            w = [(0, -1), (1, 1)] # -C00, C01
            add_term(n, m, p, off_n, off_m, off_p, u, v, w)
            
            # M6
            u = [(2, 1), (0, -1)]
            v = [(0, 1), (1, 1)]
            w = [(3, 1)] # C11
            add_term(n, m, p, off_n, off_m, off_p, u, v, w)
            
            # M7
            u = [(1, 1), (3, -1)]
            v = [(2, 1), (3, 1)]
            w = [(0, 1)] # C00
            add_term(n, m, p, off_n, off_m, off_p, u, v, w)
            
        elif n == 2 and m == 2 and p == 1:
            # Schoolbook 2x2x1
            # A00 B00 -> C00
            add_term(n, m, p, off_n, off_m, off_p, [(0, 1)], [(0, 1)], [(0, 1)])
            # A01 B10 -> C00
            add_term(n, m, p, off_n, off_m, off_p, [(1, 1)], [(1, 1)], [(0, 1)])
            # A10 B00 -> C10
            add_term(n, m, p, off_n, off_m, off_p, [(2, 1)], [(0, 1)], [(1, 1)])
            # A11 B10 -> C10
            add_term(n, m, p, off_n, off_m, off_p, [(3, 1)], [(1, 1)], [(1, 1)])
            
        elif n == 4 and m == 4 and p == 5:
            solve(2, 4, 5, 0, off_m, off_p)
            solve(2, 4, 5, 2, off_m, off_p)
        elif n == 2 and m == 4 and p == 5:
            solve(2, 2, 5, off_n, 0, off_p)
            solve(2, 2, 5, off_n, 2, off_p)
        elif n == 2 and m == 2 and p == 5:
            solve(2, 2, 4, off_n, off_m, 0)
            solve(2, 2, 1, off_n, off_m, 4)
        elif n == 2 and m == 2 and p == 4:
            solve(2, 2, 2, off_n, off_m, 0)
            solve(2, 2, 2, off_n, off_m, 2)
        else:
            raise ValueError("Unexpected dimensions")

    solve(4, 4, 5, 0, 0, 0)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    
    for c, (gu, gv, gw) in enumerate(terms):
        for idx, val in gu:
            U[idx][c] = val
        for idx, val in gv:
            V[idx][c] = val
        for idx, val in gw:
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
