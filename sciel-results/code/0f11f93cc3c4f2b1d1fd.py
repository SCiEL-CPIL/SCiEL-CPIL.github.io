import time
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 3, 5, 6

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

    def strassen_2x2x2_terms():
        terms = []
        terms.append({
            'U': {(0,0):1, (1,1):1},
            'V': {(0,0):1, (1,1):1},
            'W': {(0,0):1, (1,1):1}
        })
        terms.append({
            'U': {(1,0):1, (1,1):1},
            'V': {(0,0):1},
            'W': {(0,1):1, (1,1):-1}
        })
        terms.append({
            'U': {(0,0):1},
            'V': {(0,1):1, (1,1):-1},
            'W': {(1,0):1, (1,1):1}
        })
        terms.append({
            'U': {(1,1):1},
            'V': {(1,0):1, (0,0):-1},
            'W': {(0,0):1, (0,1):1}
        })
        terms.append({
            'U': {(0,0):1, (0,1):1},
            'V': {(1,1):1},
            'W': {(0,0):-1, (1,0):1}
        })
        terms.append({
            'U': {(1,0):1, (0,0):-1},
            'V': {(0,0):1, (0,1):1},
            'W': {(1,1):1}
        })
        terms.append({
            'U': {(0,1):1, (1,1):-1},
            'V': {(1,0):1, (1,1):1},
            'W': {(0,0):1}
        })
        return terms

    def build_strassen_decomposition():
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for term in strassen_terms:
                    for (li, lj), val_u in term['U'].items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in term['V'].items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in term['W'].items():
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

    def construct_laderman_decomposition():
        # Partition: i=3, j=3+2, k=3+3
        # Blocks:
        # 1. 3x3x3 (Laderman 23)
        # 2. 3x3x3 (Laderman 23)
        # 3. 3x2x3 (Block 17)
        # 4. 3x2x3 (Block 17)
        # Total 80
        
        # Define Laderman terms
        # I will use a placeholder for now and fill it with correct terms if I can.
        # For safety, I will return None if I can't verify.
        pass

    best_result = schoolbook()
    best_rank = 90

    strassen_res = build_strassen_decomposition()
    strassen_rank = strassen_res.pop("rank")
    if verify(strassen_res["U"], strassen_res["V"], strassen_res["W"], strassen_rank):
        best_result = strassen_res
        best_rank = strassen_rank

    return best_result
# EVOLVE-BLOCK-END
