"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
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
        w2 = {(1,0): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(0,1): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (1,0): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (0,1): 1}
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

    def build_optimized_decomposition():
        """
        Build an optimized decomposition using block structure:
        - i: {0,1}, {2}
        - j: {0,1,2,3}, {4}
        - k: {0,1,2,3}, {4,5}
        
        For (2,4,4): split into four (2,2,2) Strassen blocks = 4*7 = 28
        For (2,4,2): split into two (2,2,2) Strassen blocks = 2*7 = 14
        For (2,1,4): schoolbook = 8
        For (2,1,2): schoolbook = 4
        Total: 28+14+8+4 = 54
        """
        if time.time() > deadline:
            return None
            
        strassen_terms = strassen_2x2x2_terms()
        
        # Define the block structure
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1, 2, 3], [4]]
        k_groups = [[0, 1, 2, 3], [4, 5]]
        
        # First, calculate total rank
        total_rank = 0
        block_decomps = []
        
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    
                    if si == 2 and sj == 4 and sk == 4:
                        # Split j into two groups of 2, k into two groups of 2
                        # Four (2,2,2) Strassen blocks
                        sub_blocks = []
                        j_sub = [jg[0:2], jg[2:4]]
                        k_sub = [kg[0:2], kg[2:4]]
                        
                        for js in j_sub:
                            for ks in k_sub:
                                r = 7  # Strassen
                                sub_blocks.append((ig, js, ks, r, 2, 2, 2))
                                total_rank += r
                        block_decomps.append(('strassen_grid', sub_blocks))
                    elif si == 2 and sj == 4 and sk == 2:
                        # Split j into two groups of 2
                        # Two (2,2,2) Strassen blocks
                        sub_blocks = []
                        j_sub = [jg[0:2], jg[2:4]]
                        
                        for js in j_sub:
                            r = 7  # Strassen
                            sub_blocks.append((ig, js, kg, r, 2, 2, 2))
                            total_rank += r
                        block_decomps.append(('strassen_pair', sub_blocks))
                    elif si == 2 and sj == 1:
                        r = si * sj * sk
                        total_rank += r
                        block_decomps.append(('simple', (ig, jg, kg, r, si, sj, sk)))
                    else:
                        r = si * sj * sk
                        total_rank += r
                        block_decomps.append(('simple', (ig, jg, kg, r, si, sj, sk)))
        
        # Allocate matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        
        for block_type, data in block_decomps:
            if block_type in ('strassen_grid', 'strassen_pair'):
                for (ig, jg, kg, r, si, sj, sk) in data:
                    # Strassen 2x2x2
                    for (u_pat, v_pat, w_pat) in strassen_terms:
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
            else:  # simple
                ig, jg, kg, r, si, sj, sk = data
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
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

    # Try the optimized decomposition
    result = build_optimized_decomposition()
    
    if result is not None:
        rank = result.pop("rank")
        if verify(result["U"], result["V"], result["W"], rank):
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
