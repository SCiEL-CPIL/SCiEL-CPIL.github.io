import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 5, 6
    
    strassen = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(1,0):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    
    def verify_fast(U, V, W, rank):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i*m+j, :], V_np[j*p+k, :] * W_np[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def build_decomp(i_groups, j_groups, k_groups):
        total_rank = 0
        blocks = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        blocks.append(('s', ig, jg, kg, 7))
                        total_rank += 7
                    else:
                        r = si * sj * sk
                        blocks.append(('b', ig, jg, kg, r))
                        total_rank += r
        
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        col = 0
        
        for btype, ig, jg, kg, r in blocks:
            if btype == 's':
                for u_pat, v_pat, w_pat in strassen:
                    for (li, lj), val in u_pat.items():
                        U[ig[li]*m + jg[lj]][col] = val
                    for (lj, lk), val in v_pat.items():
                        V[jg[lj]*p + kg[lk]][col] = val
                    for (lk, li), val in w_pat.items():
                        W[kg[lk]*n + ig[li]][col] = val
                    col += 1
            else:
                for li in range(len(ig)):
                    for lj in range(len(jg)):
                        for lk in range(len(kg)):
                            U[ig[li]*m + jg[lj]][col] = 1
                            V[jg[lj]*p + kg[lk]][col] = 1
                            W[kg[lk]*n + ig[li]][col] = 1
                            col += 1
        
        if col != total_rank:
            return None
        if not verify_fast(U, V, W, total_rank):
            return None
        return U, V, W, total_rank
    
    def get_partitions(n):
        if n == 0:
            yield ()
            return
        for first in range(n, 0, -1):
            for rest in get_partitions(n - first):
                if len(rest) == 0 or first >= rest[0]:
                    yield (first,) + rest
    
    def partition_to_groups(n, part):
        groups = []
        start = 0
        for size in part:
            groups.append(list(range(start, start + size)))
            start += size
        return groups
    
    i_parts = list(get_partitions(n))
    j_parts = list(get_partitions(m))
    k_parts = list(get_partitions(p))
    
    best_rank = n * m * p + 1
    best_result = None
    
    for ip in i_parts:
        if time.time() > deadline:
            break
        i_groups = partition_to_groups(n, ip)
        for jp in j_parts:
            if time.time() > deadline:
                break
            j_groups = partition_to_groups(m, jp)
            for kp in k_parts:
                if time.time() > deadline:
                    break
                k_groups = partition_to_groups(p, kp)
                
                rank = 0
                for ig in i_groups:
                    for jg in j_groups:
                        for kg in k_groups:
                            si, sj, sk = len(ig), len(jg), len(kg)
                            if si == 2 and sj == 2 and sk == 2:
                                rank += 7
                            else:
                                rank += si * sj * sk
                
                if rank < best_rank:
                    result = build_decomp(i_groups, j_groups, k_groups)
                    if result is not None:
                        best_rank = rank
                        best_result = result
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
