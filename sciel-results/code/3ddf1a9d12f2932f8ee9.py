import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 71
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    col = 0
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (1, 2, 0, 1, 2, 3),
        (1, 2, 2, 3, 0, 1)
    ]
    
    used = set()
    
    for i0, i1, j0, j1, k0, k1 in blocks:
        # M1
        u = {idx_u(i0, j0): 1, idx_u(i1, j1): 1}
        v = {idx_v(j0, k0): 1, idx_v(j1, k1): 1}
        w = {idx_w(k0, i0): 1, idx_w(k1, i1): 1}
        for idx, val in u.items(): U[idx][col] += val
        for idx, val in v.items(): V[idx][col] += val
        for idx, val in w.items(): W[idx][col] += val
        col += 1
        
        # M2
        u = {idx_u(i1, j0): 1, idx_u(i1, j1): 1}
        v = {idx_v(j0, k0): 1}
        w = {idx_w(k0, i1): 1}
        for idx, val in u.items(): U[idx][col] += val
        for idx, val in v.items(): V[idx][col] += val
        for idx, val in w.items(): W[idx][col] += val
        col += 1
        
        # M3
        u = {idx_u(i0, j0): 1}
        v = {idx_v(j0, k1): 1, idx_v(j1, k1): -1}
        w = {idx_w(k1, i0): 1, idx_w(k1, i1): 1}
        for idx, val in u.items(): U[idx][col] += val
        for idx, val in v.items(): V[idx][col] += val
        for idx, val in w.items(): W[idx][col] += val
        col += 1
        
        # M4
        u = {idx_u(i1, j1): 1}
        v = {idx_v(j1, k0): 1, idx_v(j0, k1): -1}
        w = {idx_w(k0, i0): 1, idx_w(k0, i1): 1}
        for idx, val in u.items(): U[idx][col] += val
        for idx, val in v.items(): V[idx][col] += val
        for idx, val in w.items(): W[idx][col] += val
        col += 1
        
        # M5
        u = {idx_u(i0, j0): 1, idx_u(i0, j1): 1}
        v = {idx_v(j1, k1): 1}
        w = {idx_w(k0, i0): -1, idx_w(k1, i0): 1}
        for idx, val in u.items(): U[idx][col] += val
        for idx, val in v.items(): V[idx][col] += val
        for idx, val in w.items(): W[idx][col] += val
        col += 1
        
        # M6
        u = {idx_u(i1, j0): 1, idx_u(i0, j0): -1}
        v = {idx_v(j0, k0): 1, idx_v(j0, k1): 1}
        w = {idx_w(k1, i1): 1}
        for idx, val in u.items(): U[idx][col] += val
        for idx, val in v.items(): V[idx][col] += val
        for idx, val in w.items(): W[idx][col] += val
        col += 1
        
        # M7
        u = {idx_u(i0, j1): 1, idx_u(i1, j1): -1}
        v = {idx_v(j1, k0): 1, idx_v(j1, k1): 1}
        w = {idx_w(k0, i0): 1}
        for idx, val in u.items(): U[idx][col] += val
        for idx, val in v.items(): V[idx][col] += val
        for idx, val in w.items(): W[idx][col] += val
        col += 1
        
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    used.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used:
                    U[idx_u(i, j)][col] = 1
                    V[idx_v(j, k)][col] = 1
                    W[idx_w(k, i)][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
