import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def build_decomposition():
        if time.time() > deadline:
            return None

        strassen_terms = strassen_2x2x2_terms()
        
        # Partition indices
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        U_rows = n * m
        V_rows = m * p
        W_rows = p * n
        
        U = []
        V = []
        W = []
        
        # We build column by column (rank dimension)
        # But it's easier to build the lists of non-zeros and then assemble
        
        # Let's just append to the lists
        # Initialize with zeros
        # We don't know total rank in advance, so we'll collect terms
        
        terms_list = []
        
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si = len(ig)
                    sj = len(jg)
                    sk = len(kg)
                    
                    if si == 2 and sj == 2 and sk == 2:
                        # Strassen
                        for (u_pat, v_pat, w_pat) in strassen_terms:
                            u_term = {}
                            v_term = {}
                            w_term = {}
                            
                            for (li, lj), val_u in u_pat.items():
                                gi, gj = ig[li], jg[lj]
                                u_term[(gi, gj)] = val_u
                                
                            for (lj, lk), val_v in v_pat.items():
                                gj, gk = jg[lj], kg[lk]
                                v_term[(gj, gk)] = val_v
                                
                            for (lk, li), val_w in w_pat.items():
                                gk, gi = kg[lk], ig[li]
                                w_term[(gk, gi)] = val_w
                                
                            terms_list.append((u_term, v_term, w_term))
                    else:
                        # Schoolbook
                        for li in range(si):
                            for lj in range(sj):
                                for lk in range(sk):
                                    gi = ig[li]
                                    gj = jg[lj]
                                    gk = kg[lk]
                                    
                                    u_term = {(gi, gj): 1}
                                    v_term = {(gj, gk): 1}
                                    w_term = {(gk, gi): 1}
                                    
                                    terms_list.append((u_term, v_term, w_term))
        
        rank = len(terms_list)
        
        # Assemble U, V, W
        U = [[0] * rank for _ in range(U_rows)]
        V = [[0] * rank for _ in range(V_rows)]
        W = [[0] * rank for _ in range(W_rows)]
        
        for col, (u_term, v_term, w_term) in enumerate(terms_list):
            for (i, j), val in u_term.items():
                U[i * m + j][col] = val
            for (j, k), val in v_term.items():
                V[j * p + k][col] = val
            for (k, i), val in w_term.items():
                W[k * n + i][col] = val
                
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

    result = build_decomposition()
    
    if result is not None:
        rank = len(result["U"][0])
        if verify(result["U"], result["V"], result["W"], rank):
            return result
            
    # Fallback
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
# EVOLVE-BLOCK-END
