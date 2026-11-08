import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
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

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def laderman_3x3x3_terms():
        terms = []
        
        # M1: U={(0,0):1,(1,0):1,(1,1):1,(2,1):1}, V={(0,0):1,(1,0):1,(1,1):1,(2,1):1}
        # W: appears in C00(+),C10(+),C20(+) → (k,i)=(0,0):+1,(0,1):+1,(0,2):+1
        terms.append(({(0,0):1,(1,0):1,(1,1):1,(2,1):1}, 
                      {(0,0):1,(1,0):1,(1,1):1,(2,1):1},
                      {(0,0):1,(0,1):1,(0,2):1}))
        
        # M2: U={(0,1):1,(0,2):1,(1,2):1,(2,2):1}, V={(0,1):1,(0,2):1,(1,2):1,(2,2):1}
        # W: C01(+),C02(+),C21(+) → (k,i)=(1,0):+1,(2,0):+1,(1,2):+1
        terms.append(({(0,1):1,(0,2):1,(1,2):1,(2,2):1}, 
                      {(0,1):1,(0,2):1,(1,2):1,(2,2):1},
                      {(1,0):1,(2,0):1,(1,2):1}))
        
        # M3: U={(1,1):1,(2,2):1}, V={(1,1):1,(2,2):1}
        # W: C00(-),C10(-),C11(+),C12(-),C22(+) → (0,0):-1,(0,1):-1,(1,1):+1,(2,1):-1,(2,2):+1
        terms.append(({(1,1):1,(2,2):1}, 
                      {(1,1):1,(2,2):1},
                      {(0,0):-1,(0,1):-1,(1,1):1,(2,1):-1,(2,2):1}))
        
        # M4: U={(0,0):1,(1,1):1}, V={(0,0):1,(1,1):1}
        # W: C00(-),C10(-),C11(+) → (0,0):-1,(0,1):-1,(1,1):+1
        terms.append(({(0,0):1,(1,1):1}, 
                      {(0,0):1,(1,1):1},
                      {(0,0):-1,(0,1):-1,(1,1):1}))
        
        # M5: U={(0,1):1,(1,2):1}, V={(0,1):1,(1,2):1}
        # W: C01(-),C02(-),C11(+),C12(+) → (1,0):-1,(2,0):-1,(1,1):+1,(2,1):+1
        terms.append(({(0,1):1,(1,2):1}, 
                      {(0,1):1,(1,2):1},
                      {(1,0):-1,(2,0):-1,(1,1):1,(2,1):1}))
        
        # M6: U={(1,0):1,(2,1):1}, V={(1,0):1,(2,1):1}
        # W: C00(-),C10(+),C20(+) → (0,0):-1,(0,1):+1,(0,2):+1
        terms.append(({(1,0):1,(2,1):1}, 
                      {(1,0):1,(2,1):1},
                      {(0,0):-1,(0,1):1,(0,2):1}))
        
        # M7: U={(0,0):1,(2,2):1}, V={(0,0):1,(2,2):1}
        # W: C00(+),C02(-),C11(-),C12(-),C20(-),C21(+),C22(+) → (0,0):+1,(2,0):-1,(1,1):-1,(2,1):-1,(0,2):-1,(1,2):+1,(2,2):+1
        terms.append(({(0,0):1,(2,2):1}, 
                      {(0,0):1,(2,2):1},
                      {(0,0):1,(2,0):-1,(1,1):-1,(2,1):-1,(0,2):-1,(1,2):1,(2,2):1}))
        
        return terms

    def build_decomposition():
        i_groups = [[0, 1, 2]]
        j_groups = [[0, 1, 2], [3, 4]]
        k_groups = [[0, 1, 2], [3, 4, 5]]
        
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 3 and sj == 3 and sk == 3:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        laderman_terms = laderman_3x3x3_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 3 and sj == 3 and sk == 3:
                for (u_pat, v_pat, w_pat) in laderman_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    result = build_decomposition()
    rank = result.pop("rank")
    
    if verify(result["U"], result["V"], result["W"], rank):
        return result
    
    return schoolbook()
