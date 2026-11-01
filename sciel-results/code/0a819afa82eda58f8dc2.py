import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen's algorithm for 2x2x2 has rank 7.
    # We have 6 independent 2x2x2 blocks in the 2x2x12 multiplication.
    # Total rank = 6 * 7 = 42.
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for each of the 7 terms.
    # U coeffs for A (size 4: 00, 01, 10, 11)
    # V coeffs for B block (size 4: 00, 01, 10, 11)
    # W coeffs for C block (size 4: 00, 01, 10, 11)
    
    # Term 1: (a+d)(e+h) -> M1
    # A: a(0), d(3)
    # B: e(0), h(3)
    # C: c00(0), c11(3)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    
    # Term 2: (c+d)e -> M2
    # A: c(2), d(3)
    # B: e(0)
    # C: c10(2), c11(3)
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1]
    
    # Term 3: a(f-h) -> M3
    # A: a(0)
    # B: f(1), h(3)
    # C: c01(1), c11(3)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1]
    
    # Term 4: d(g-e) -> M4
    # A: d(3)
    # B: g(2), e(0)
    # C: c00(0), c10(2)
    u4 = [0, 0, 0, 1]
    v4 = [-1, 0, 1, 0]
    w4 = [1, 0, 1, 0]
    
    # Term 5: (a+b)h -> M5
    # A: a(0), b(1)
    # B: h(3)
    # C: c00(0), c01(1)
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0]
    
    # Term 6: (c-a)(e+f) -> M6
    # A: c(2), a(0)
    # B: e(0), f(1)
    # C: c11(3)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    
    # Term 7: (b-d)(g+h) -> M7
    # A: b(1), d(3)
    # B: g(2), h(3)
    # C: c00(0)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0]
    
    terms_u = [u1, u2, u3, u4, u5, u6, u7]
    terms_v = [v1, v2, v3, v4, v5, v6, v7]
    terms_w = [w1, w2, w3, w4, w5, w6, w7]
    
    for block in range(6):
        # Global indices for B and C in this block
        # B is 2x12, so block b uses cols 2b, 2b+1.
        # V index is j*p + k. j in 0..1, k in 0..11.
        # Local j=0,1. Local k=0,1.
        # Global k = 2*block + local_k.
        # V global idx = j*12 + (2*block + local_k)
        
        # W is 12x2. Index is k*n + i. k in 0..11, i in 0..1.
        # Global k = 2*block + local_k.
        # W global idx = (2*block + local_k)*2 + i
        
        for t in range(7):
            r = block * 7 + t
            
            # Fill U (A is common, so same for all blocks)
            for i in range(n):
                for j in range(m):
                    U[i*m+j][r] = terms_u[t][i*m+j]
                    
            # Fill V
            for j in range(m):
                for local_k in range(2):
                    local_idx = j*2 + local_k
                    global_k = 2*block + local_k
                    global_v_idx = j*p + global_k
                    V[global_v_idx][r] = terms_v[t][local_idx]
                    
            # Fill W
            for local_k in range(2):
                for i in range(n):
                    local_idx = local_k*2 + i
                    global_k = 2*block + local_k
                    global_w_idx = global_k*n + i
                    W[global_w_idx][r] = terms_w[t][local_idx]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
