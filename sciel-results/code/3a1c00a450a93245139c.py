import numpy as np
import time

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    def u_idx(i, j): return i * 4 + j
    def v_idx(i, j): return i * 4 + j
    def w_idx(i, j): return j * 4 + i

    def combine(ref1, ref2, sign=1):
        d = {}
        for r, c, v in ref1:
            d[(r, c)] = d.get((r, c), 0) + v
        for r, c, v in ref2:
            d[(r, c)] = d.get((r, c), 0) + sign * v
        return [(k[0], k[1], v) for k, v in d.items() if v != 0]

    def a_ref(bi, bj, r, c, coeff=1): return [(2*bi+r, 2*bj+c, coeff)]
    def b_ref(bi, bj, r, c, coeff=1): return [(2*bi+r, 2*bj+c, coeff)]

    # Block Terms L and R
    # Term 0: (A00+A11)(B00+B11)
    L0 = [[combine(a_ref(0,0,r,c), a_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    R0 = [[combine(b_ref(0,0,r,c), b_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    
    # Term 1: (A10+A11)B00
    L1 = [[combine(a_ref(1,0,r,c), a_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    R1 = [[b_ref(0,0,r,c) for c in range(2)] for r in range(2)]
    
    # Term 2: A00(B01-B11)
    L2 = [[a_ref(0,0,r,c) for c in range(2)] for r in range(2)]
    R2 = [[combine(b_ref(0,1,r,c), b_ref(1,1,r,c), -1) for c in range(2)] for r in range(2)]
    
    # Term 3: A11(B10-B00)
    L3 = [[a_ref(1,1,r,c) for c in range(2)] for r in range(2)]
    R3 = [[combine(b_ref(1,0,r,c), b_ref(0,0,r,c), -1) for c in range(2)] for r in range(2)]
    
    # Term 4: (A00+A01)B11
    L4 = [[combine(a_ref(0,0,r,c), a_ref(0,1,r,c)) for c in range(2)] for r in range(2)]
    R4 = [[b_ref(1,1,r,c) for c in range(2)] for r in range(2)]
    
    # Term 5: (A10-A00)(B00+B01)
    L5 = [[combine(a_ref(1,0,r,c), a_ref(0,0,r,c), -1) for c in range(2)] for r in range(2)]
    R5 = [[combine(b_ref(0,0,r,c), b_ref(0,1,r,c)) for c in range(2)] for r in range(2)]
    
    # Term 6: (A01-A11)(B10+B11)
    L6 = [[combine(a_ref(0,1,r,c), a_ref(1,1,r,c), -1) for c in range(2)] for r in range(2)]
    R6 = [[combine(b_ref(1,0,r,c), b_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    
    L_terms = [L0, L1, L2, L3, L4, L5, L6]
    R_terms = [R0, R1, R2, R3, R4, R5, R6]
    
    block_contribs = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    sub_contrib = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]
    
    for t in range(7):
        L_t = L_terms[t]
        R_t = R_terms[t]
        contrib_t = block_contribs[t]
        
        a, b, c, d = L_t[0][0], L_t[0][1], L_t[1][0], L_t[1][1]
        e, f, g, h = R_t[0][0], R_t[0][1], R_t[1][0], R_t[1][1]
        
        sub_products = [
            (combine(a, d), combine(e, h)),
            (combine(c, d), e),
            (a, combine(f, h, -1)),
            (d, combine(g, e, -1)),
            (combine(a, b), h),
            (combine(c, a, -1), combine(e, f)),
            (combine(b, d, -1), combine(g, h))
        ]
        
        for k in range(7):
            x_k, y_k = sub_products[k]
            rank_idx = t * 7 + k
            
            for (row, col, coeff) in x_k:
                U[u_idx(row, col), rank_idx] += coeff
            for (row, col, coeff) in y_k:
                V[v_idx(row, col), rank_idx] += coeff
            
            for (pr, pc, sub_coeff) in sub_contrib[k]:
                for (br, bc, block_coeff) in contrib_t:
                    out_row = 2 * br + pr
                    out_col = 2 * bc + pc
                    total_coeff = sub_coeff * block_coeff
                    W[w_idx(out_row, out_col), rank_idx] += total_coeff

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
