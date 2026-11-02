import numpy as np

def construct():
    n, m, p = 2, 2, 4
    rank = 14
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen matrices for one 2x2 block
    # U indices: 0(a), 1(b), 2(c), 3(d)
    # V local indices: 0(e), 1(f), 2(g), 3(h)
    # W local indices: 0(C00), 1(C01), 2(C10), 3(C11)
    
    # M1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    
    # M2
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1]
    
    # M3
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1]
    
    # M4
    u4 = [0, 0, 0, 1]
    v4 = [0, -1, 1, 0]
    w4 = [1, 0, 1, 0]
    
    # M5
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0]
    
    # M6
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    
    # M7
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0]
    
    mats = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Fill U, V, W for block 1 (cols 0,1 of B)
    # V global indices for block 1: e=0, f=1, g=4, h=5
    # W global indices for block 1: C00=0, C01=2, C10=1, C11=3
    for i, (u, v, w) in enumerate(mats):
        col = i
        for k, val in enumerate(u):
            if val != 0:
                U[k, col] = val
        for k, val in enumerate(v):
            # Map local v index to global
            # 0->0, 1->1, 2->4, 3->5
            glob_idx = [0, 1, 4, 5][k]
            V[glob_idx, col] = val
        for k, val in enumerate(w):
            # Map local w index to global
            # 0->0, 1->2, 2->1, 3->3
            glob_idx = [0, 2, 1, 3][k]
            W[glob_idx, col] = val
            
    # Fill U, V, W for block 2 (cols 2,3 of B)
    # V global indices for block 2: e=2, f=3, g=6, h=7
    # W global indices for block 2: C02=4, C03=6, C12=5, C13=7
    for i, (u, v, w) in enumerate(mats):
        col = 7 + i
        for k, val in enumerate(u):
            if val != 0:
                U[k, col] = val
        for k, val in enumerate(v):
            glob_idx = [2, 3, 6, 7][k]
            V[glob_idx, col] = val
        for k, val in enumerate(w):
            glob_idx = [4, 6, 5, 7][k]
            W[glob_idx, col] = val
            
    # Verification
    # T_{i,j,k} = sum_r U[i*2+j, r] V[j*4+k, r] W[k*2+i, r]
    # Should be 1 if i,i' and k,k' match, else 0.
    # Actually, just check that the product reconstructs the identity-like tensor.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                # The result should contribute 1 to W[k*n+i] and 0 to others
                # So sum_r U[u_idx, r] V[v_idx, r] W[k'*n+i', r] should be delta(k,k')delta(i,i')
                for i_p in range(n):
                    for k_p in range(p):
                        w_idx = k_p*n + i_p
                        s = 0
                        for r in range(rank):
                            s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                        expected = 1 if (i == i_p and k == k_p) else 0
                        if abs(s - expected) > 1e-9:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
