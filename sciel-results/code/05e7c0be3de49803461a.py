import numpy as np

def construct():
    n, m, p = 2, 3, 4
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, r):
        # u_indices: list of (idx, val)
        # v_indices: list of (idx, val)
        # w_indices: list of (idx, val)
        for idx, val in u_indices:
            U[idx][r] += val
        for idx, val in v_indices:
            V[idx][r] += val
        for idx, val in w_indices:
            W[idx][r] += val
            
    def block_decomposition(offset_k):
        # offset_k is 0 for block 1, 2 for block 2
        # Returns list of terms
        terms = []
        
        # Strassen 7 terms
        # A indices: i in {0,1}, j in {0,1}
        # B indices: j in {0,1}, k in {offset_k, offset_k+1}
        # C indices: i in {0,1}, k in {offset_k, offset_k+1}
        
        # m1 = (a+d)(e+h)
        # a=A00, d=A11, e=B00, h=B11 (local)
        # Global A: 00, 11
        # Global B: 0*4+k0, 1*4+k1
        # Global C: k0*2+0, k1*2+1
        
        k0 = offset_k
        k1 = offset_k + 1
        
        # m1
        u = [(0, 1), (3, 1)] # A00, A11
        v = [(k0, 1), (4+k1, 1)] # B0k0, B1k1
        w = [(2*k0+0, 1), (2*k1+1, 1)] # C0k0, C1k1
        terms.append((u, v, w))
        
        # m2 = (c+d)e
        # c=A10, d=A11, e=B00
        u = [(2, 1), (3, 1)] # A10, A11
        v = [(k0, 1)] # B0k0
        w = [(2*k0+1, 1), (2*k1+1, -1)] # C1k0, C1k1
        terms.append((u, v, w))
        
        # m3 = a(f-h)
        # a=A00, f=B01, h=B11
        u = [(0, 1)] # A00
        v = [(4+k0, 1), (4+k1, -1)] # B0k1? No.
        # Local B: e=B00, f=B01, g=B10, h=B11.
        # So f is B_{0, k1}, h is B_{1, k1}.
        # Global B index: j*4 + k.
        # B_{0, k1} is 0*4 + k1 = k1.
        # B_{1, k1} is 1*4 + k1 = 4+k1.
        v = [(k1, 1), (4+k1, -1)]
        w = [(2*k0+1, 1), (2*k1+1, 1)] # C0k1? No.
        # q = C0k1, s = C1k1.
        # q = m3 + m5.
        # So m3 contributes to C0k1 and C1k1?
        # s = m1 - m2 + m3 + m6.
        # p = m1 + m4 - m5 + m7.
        # r = m2 + m4.
        # q = m3 + m5.
        # So m3 contributes to q (C0k1) and s (C1k1).
        # Global C: C_{0, k1} is 2*k1 + 0. C_{1, k1} is 2*k1 + 1.
        w = [(2*k1+0, 1), (2*k1+1, 1)]
        terms.append((u, v, w))
        
        # m4 = d(g-e)
        # d=A11, g=B10, e=B00
        u = [(3, 1)] # A11
        v = [(4+k0, 1), (k0, -1)] # B1k0, B0k0
        w = [(2*k0+0, 1), (2*k0+1, 1)] # C0k0, C1k0
        terms.append((u, v, w))
        
        # m5 = (a+b)h
        # a=A00, b=A01, h=B11
        u = [(0, 1), (1, 1)] # A00, A01
        v = [(4+k1, 1)] # B1k1
        w = [(2*k0+0, -1), (2*k1+0, 1)] # C0k0, C0k1
        terms.append((u, v, w))
        
        # m6 = (c-d)(f+g)
        # c=A10, d=A11, f=B01, g=B10
        u = [(2, 1), (3, -1)] # A10, A11
        v = [(k1, 1), (4+k0, 1)] # B0k1, B1k0
        w = [(2*k1+1, 1)] # C1k1
        terms.append((u, v, w))
        
        # m7 = (b-c)(e+g)
        # b=A01, c=A10, e=B00, g=B10
        u = [(1, 1), (2, -1)] # A01, A10
        v = [(k0, 1), (4+k0, 1)] # B0k0, B1k0
        w = [(2*k0+0, 1)] # C0k0
        terms.append((u, v, w))
        
        # Schoolbook 4 terms for A12 * B12
        # A12 is col 2 of A. Indices 4, 5.
        # B12 is row 2 of B (local). Indices 8+k0, 8+k1.
        # C is 2x2.
        
        # C0k0 = A02 * B2k0
        u = [(4, 1)]
        v = [(8+k0, 1)]
        w = [(2*k0+0, 1)]
        terms.append((u, v, w))
        
        # C0k1 = A02 * B2k1
        u = [(4, 1)]
        v = [(8+k1, 1)]
        w = [(2*k1+0, 1)]
        terms.append((u, v, w))
        
        # C1k0 = A12 * B2k0
        u = [(5, 1)]
        v = [(8+k0, 1)]
        w = [(2*k0+1, 1)]
        terms.append((u, v, w))
        
        # C1k1 = A12 * B2k1
        u = [(5, 1)]
        v = [(8+k1, 1)]
        w = [(2*k1+1, 1)]
        terms.append((u, v, w))
        
        return terms

    r = 0
    for offset in [0, 2]:
        terms = block_decomposition(offset)
        for u, v, w in terms:
            add_term(u, v, w, r)
            r += 1
            
    return {"U": U, "V": V, "W": W}
