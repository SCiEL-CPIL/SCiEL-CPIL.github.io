# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    t_limit = 110.0 # Safety margin
    
    n, m, p = 2, 4, 8
    
    # Try to find a better decomposition for 2x4x2 (rank 13)
    # If found, replicate 4 times for 2x4x8 (rank 52)
    
    import numpy as np
    import random
    
    def find_rank_13_2x4x2():
        nn, mm, pp = 2, 4, 2
        R = 13
        size_U = nn * mm
        size_V = mm * pp
        size_W = pp * nn
        
        for attempt in range(200):
            if time.time() - t0 > t_limit - 10:
                return None
            U = np.random.choice([-1, 0, 1], size=(size_U, R), p=[0.3, 0.4, 0.3])
            V = np.random.choice([-1, 0, 1], size=(size_V, R), p=[0.3, 0.4, 0.3])
            
            num_eqs = nn * mm * mm * pp
            M = np.zeros((num_eqs, size_W * R))
            y = np.zeros(num_eqs)
            
            idx = 0
            for i in range(nn):
                for la in range(mm):
                    for lb in range(mm):
                        for k in range(pp):
                            y[idx] = 1.0 if la == lb else 0.0
                            for r in range(R):
                                u_val = U[i*mm + la, r]
                                v_val = V[lb*pp + k, r]
                                if u_val != 0 and v_val != 0:
                                    w_idx = k*nn + i
                                    col_idx = w_idx * R + r
                                    M[idx, col_idx] = u_val * v_val
                            idx += 1
            
            try:
                W_flat, res, rank, sv = np.linalg.lstsq(M, y, rcond=None)
            except:
                continue
                
            if np.max(np.abs(M @ W_flat - y)) > 1e-9:
                continue
                
            W = W_flat.reshape(size_W, R)
            
            if not np.allclose(U*2, np.round(U*2)): continue
            if not np.allclose(V*2, np.round(V*2)): continue
            if not np.allclose(W*2, np.round(W*2)): continue
            
            if np.max(np.abs(W)) > 3: continue
            if np.max(np.abs(U)) > 2: continue
            if np.max(np.abs(V)) > 2: continue
            
            return U, V, W
            
        return None

    U_2x4x2 = None
    V_2x4x2 = None
    W_2x4x2 = None
    
    if time.time() - t0 < t_limit - 20:
        res = find_rank_13_2x4x2()
        if res:
            U_2x4x2, V_2x4x2, W_2x4x2 = res

    if U_2x4x2 is not None:
        # Construct 2x4x8 from 4 copies of 2x4x2
        total_rank = 4 * 13
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        # Mapping:
        # 2x4x2 uses A(2x4), B(4x2), C(2x2)
        # 2x4x8 uses A(2x4), B(4x8), C(2x8)
        # B is split into 4 blocks of 4x2.
        # C is split into 4 blocks of 2x2.
        
        for j in range(4):
            base_r = j * 13
            # A is same
            U[:, base_r:base_r+13] = U_2x4x2
            
            # B block j: rows 0..3, cols 2j..2j+1
            # V_2x4x2 index: l*2 + k (where k is 0,1)
            # V_2x4x8 index: l*8 + k_global
            for l in range(4):
                for k_local in range(2):
                    v_val = V_2x4x2[l*2 + k_local, :]
                    if np.any(v_val != 0):
                        V[l*8 + 2*j + k_local, base_r:base_r+13] = v_val
                        
            # C block j: rows 0..1, cols 2j..2j+1
            # W_2x4x2 index: k*2 + i (k is 0,1)
            # W_2x4x8 index: k_global*2 + i
            for i in range(2):
                for k_local in range(2):
                    w_val = W_2x4x2[k_local*2 + i, :]
                    if np.any(w_val != 0):
                        W[(2*j + k_local)*2 + i, base_r:base_r+13] = w_val
                        
        return {"U": U, "V": V, "W": W}
        
    # Fallback to rank 56
    # ... (existing code) ...
