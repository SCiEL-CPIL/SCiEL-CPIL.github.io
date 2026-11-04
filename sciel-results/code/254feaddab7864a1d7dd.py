import time
import numpy as np

def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 5
    
    rank = 65
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    r = 0
    
    def add_term(u_entries, v_entries, w_entries):
        nonlocal r
        if r >= rank:
            raise ValueError("Rank overflow")
        for idx, coeff in u_entries:
            U[idx][r] += coeff
        for idx, coeff in v_entries:
            V[idx][r] += coeff
        for idx, coeff in w_entries:
            W[idx][r] += coeff
        r += 1
    
    # Top-level Strassen products for 4x4x4
    products = [
        {'S': [((0,0), 1.0), ((1,1), 1.0)], 'T': [((0,0), 1.0), ((1,1), 1.0)], 'C': {((0,0),): 1.0, ((1,1),): 1.0}},
        {'S': [((1,0), 1.0), ((1,1), 1.0)], 'T': [((0,0), 1.0)], 'C': {((1,0),): 1.0, ((1,1),): -1.0}},
        {'S': [((0,0), 1.0)], 'T': [((1,0), 1.0), ((1,1), -1.0)], 'C': {((0,1),): 1.0, ((1,1),): 1.0}},
        {'S': [((1,1), 1.0)], 'T': [((0,1), 1.0), ((0,0), -1.0)], 'C': {((0,0),): 1.0}},
        {'S': [((0,0), 1.0), ((0,1), 1.0)], 'T': [((1,0), 1.0)], 'C': {((0,0),): -1.0, ((0,1),): 1.0}},
        {'S': [((1,0), 1.0), ((0,0), -1.0)], 'T': [((0,0), 1.0), ((0,1), 1.0)], 'C': {((1,0),): 1.0}},
        {'S': [((0,1), 1.0), ((1,1), -1.0)], 'T': [((1,0), 1.0), ((1,1), 1.0)], 'C': {((0,0),): 1.0}},
    ]
    
    sub_products = [
        ({(0,0): 1.0, (1,1): 1.0}, {(0,0): 1.0, (1,1): 1.0}, {(0,0): 1.0, (1,1): 1.0}),
        ({(1,0): 1.0, (1,1): 1.0}, {(0,0): 1.0}, {(0,1): 1.0, (1,1): -1.0}),
        ({(0,0): 1.0}, {(1,0): 1.0, (1,1): -1.0}, {(0,0): 1.0, (1,1): 1.0}),
        ({(1,1): 1.0}, {(0,1): 1.0, (0,0): -1.0}, {(0,0): 1.0}),
        ({(0,0): 1.0, (0,1): 1.0}, {(1,0): 1.0}, {(0,0): -1.0, (0,1): 1.0}),
        ({(1,0): 1.0, (0,0): -1.0}, {(0,0): 1.0, (0,1): 1.0}, {(0,1): 1.0}),
        ({(0,1): 1.0, (1,1): -1.0}, {(1,0): 1.0, (1,1): 1.0}, {(0,0): 1.0}),
    ]
    
    for prod in products:
        S = {}
        for (u, v), coeff in prod['S']:
            for li in range(2):
                for lj in range(2):
                    if (li, lj) not in S:
                        S[(li, lj)] = []
                    S[(li, lj)].append((u_idx(2*u + li, 2*v + lj), coeff))
        
        T = {}
        for (v, w), coeff in prod['T']:
            for lj in range(2):
                for lk in range(2):
                    if (lj, lk) not in T:
                        T[(lj, lk)] = []
                    T[(lj, lk)].append((v_idx(2*v + lj, 2*w + lk), coeff))
        
        for S_coeffs, T_coeffs, C_coeffs in sub_products:
            u_entries = []
            for (li, lj), sc in S_coeffs.items():
                for (gidx, bc) in S[(li, lj)]:
                    u_entries.append((gidx, sc * bc))
            
            v_entries = []
            for (lj, lk), tc in T_coeffs.items():
                for (gidx, bc) in T[(lj, lk)]:
                    v_entries.append((gidx, tc * bc))
            
            w_entries = []
            for (lk, li), cc in C_coeffs.items():
                for (uw), cc_block in prod['C'].items():
                    u_block, w_block = uw
                    global_w = w_idx(2*w_block + lk, 2*u_block + li)
                    w_entries.append((global_w, cc * cc_block))
            
            add_term(u_entries, v_entries, w_entries)
    
    for i in range(n):
        for j in range(m):
            add_term(
                [(u_idx(i, j), 1.0)],
                [(v_idx(j, 4), 1.0)],
                [(w_idx(4, i), 1.0)]
            )
    
    assert r == rank, f"Expected rank {rank}, used {r}"
    
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    for i in range(n):
        for k in range(p):
            s = 0.0
            for rr in range(rank):
                inner = 0.0
                for j in range(m):
                    inner += U[i * m + j][rr] * V[j * p + k][rr]
                s += W[k * n + i][rr] * inner
            if abs(s - C_expected[i, k]) > 1e-9:
                raise ValueError(f"Verification failed at ({i},{k}): expected {C_expected[i,k]}, got {s}")
    
    return {"U": U, "V": V, "W": W}
