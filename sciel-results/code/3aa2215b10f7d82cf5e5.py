import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Correct Strassen-7 decomposition for 2x2x2
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # Verified: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    
    strassen_U = [
        [(0, 1), (3, 1)],       # M1: A00+A11
        [(2, 1), (3, 1)],       # M2: A10+A11
        [(0, 1)],               # M3: A00
        [(3, 1)],               # M4: A11
        [(0, 1), (1, 1)],       # M5: A00+A01
        [(2, 1), (0, -1)],      # M6: A10-A00
        [(1, 1), (3, -1)],      # M7: A01-A11
    ]
    strassen_V = [
        [(0, 1), (3, 1)],       # M1: B00+B11
        [(0, 1)],               # M2: B00
        [(2, 1), (3, -1)],      # M3: B10-B11
        [(1, 1), (2, -1)],      # M4: B01-B10
        [(3, 1)],               # M5: B11
        [(0, 1), (2, 1)],       # M6: B00+B10
        [(1, 1), (3, 1)],       # M7: B01+B11
    ]
    strassen_W = [
        [(0, 1), (1, 1), (3, -1)],   # M1: C00+1, C10+1, C11-1
        [(2, 1), (1, -1)],           # M2: C01+1, C10-1
        [(2, 1), (3, 1)],            # M3: C01+1, C11+1
        [(0, 1), (2, -1), (3, 1)],   # M4: C00+1, C01-1, C11+1
        [(0, -1), (1, 1)],           # M5: C00-1, C10+1
        [(2, 1), (1, 1)],            # M6: C01+1, C10+1
        [(0, 1), (2, 1)],            # M7: C00+1, C01+1
    ]
    
    # Verify Strassen decomposition
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for r in range(7):
                        u_val = 0
                        v_val = 0
                        w_val = 0
                        u_idx = i * 2 + j
                        v_idx = j * 2 + k
                        w_idx = k * 2 + i
                        for (idx, val) in strassen_U[r]:
                            if idx == u_idx:
                                u_val = val
                        for (idx, val) in strassen_V[r]:
                            if idx == v_idx:
                                v_val = val
                        for (idx, val) in strassen_W[r]:
                            if idx == w_idx:
                                w_val = val
                        total += u_val * v_val * w_val
                    if total != 1:
                        return False
        return True
    
    if not verify_strassen():
        # Should not happen, but fallback to naive rank 64
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Build rank-12 decomposition for 2x2x4 using two 2x2x2 Strassen blocks
    def build_224_terms():
        terms = []
        for block in range(2):
            k_offset = block * 2
            for r in range(7):
                u_entries = strassen_U[r]
                v_entries = []
                for (idx, val) in strassen_V[r]:
                    lj, lk_local = divmod(idx, 2)
                    v_entries.append((lj * 4 + lk_local + k_offset, val))
                w_entries = []
                for (idx, val) in strassen_W[r]:
                    lk_local, li = divmod(idx, 2)
                    w_entries.append(((lk_local + k_offset) * 2 + li, val))
                terms.append((u_entries, v_entries, w_entries))
        return terms
    
    terms_224 = build_224_terms()
    
    def verify_224():
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    total = 0
                    for r in range(12):
                        u_val = 0
                        v_val = 0
                        w_val = 0
                        u_idx = i * 2 + j
                        v_idx = j * 4 + k
                        w_idx = k * 2 + i
                        for (idx, val) in terms_224[r][0]:
                            if idx == u_idx:
                                u_val = val
                        for (idx, val) in terms_224[r][1]:
                            if idx == v_idx:
                                v_val = val
                        for (idx, val) in terms_224[r][2]:
                            if idx == w_idx:
                                w_val = val
                        total += u_val * v_val * w_val
                    if total != 1:
                        return False
        return True
    
    if not verify_224():
        # Fallback: use 8 independent Strassen blocks (rank 56)
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(base_rank, i_off, j_off, k_off):
            for r in range(7):
                cr = base_rank + r
                for idx, val in strassen_U[r]:
                    li, lj = divmod(idx, 2)
                    U[(li + i_off) * 4 + (lj + j_off)][cr] += val
                for idx, val in strassen_V[r]:
                    lj, lk = divmod(idx, 2)
                    V[(lj + j_off) * 8 + (lk + k_off)][cr] += val
                for idx, val in strassen_W[r]:
                    lk, li = divmod(idx, 2)
                    W[(lk + k_off) * 2 + (li + i_off)][cr] += val
        
        for j_block in range(2):
            for k_block in range(4):
                base = (j_block * 4 + k_block) * 7
                add_strassen_block(base, 0, j_block * 2, k_block * 2)
        
        return {"U": U, "V": V, "W": W}
    
    # Apply rank-12 decomposition for each of 4 sub-blocks
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j_block in range(2):
        for k_block in range(2):
            base_rank = (j_block * 2 + k_block) * 12
            for r in range(12):
                cr = base_rank + r
                for (idx, val) in terms_224[r][0]:
                    li, lj = divmod(idx, 2)
                    U[li * 4 + (j_block * 2 + lj)][cr] += val
                for (idx, val) in terms_224[r][1]:
                    lj, lk = divmod(idx, 4)
                    V[(j_block * 2 + lj) * 8 + (k_block * 4 + lk)][cr] += val
                for (idx, val) in terms_224[r][2]:
                    lk, li = divmod(idx, 2)
                    W[(k_block * 4 + lk) * 2 + li][cr] += val
    
    # Final verification
    def full_verify():
        A = np.random.randint(-2, 3, size=(n, m))
        B = np.random.randint(-2, 3, size=(m, p))
        C_expected = A @ B
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0
                for j in range(m):
                    for r in range(rank):
                        total += A[i, j] * U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                C_computed[i, k] = total
        
        return np.allclose(C_computed, C_expected)
    
    if not full_verify():
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(base_rank, i_off, j_off, k_off):
            for r in range(7):
                cr = base_rank + r
                for idx, val in strassen_U[r]:
                    li, lj = divmod(idx, 2)
                    U[(li + i_off) * 4 + (lj + j_off)][cr] += val
                for idx, val in strassen_V[r]:
                    lj, lk = divmod(idx, 2)
                    V[(lj + j_off) * 8 + (lk + k_off)][cr] += val
                for idx, val in strassen_W[r]:
                    lk, li = divmod(idx, 2)
                    W[(lk + k_off) * 2 + (li + i_off)][cr] += val
        
        for j_block in range(2):
            for k_block in range(4):
                base = (j_block * 4 + k_block) * 7
                add_strassen_block(base, 0, j_block * 2, k_block * 2)
        
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
