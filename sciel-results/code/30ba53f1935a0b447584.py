import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 15, 15
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 45)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Attempt Rank 44 via Strassen on 2x2x2 sub-tensor (i,j,k in {0,1})
    U_new = []
    V_new = []
    W_new = []
    
    # Add the 37 non-overlapping schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in [0, 1] and j in [0, 1] and k in [0, 1]:
                    continue
                a = i * m + j
                b = j * p + k
                c = k * n + i
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                u[a] = 1
                v[b] = 1
                w[c] = 1
                U_new.append(u)
                V_new.append(v)
                W_new.append(w)
    
    # Standard Strassen terms for 2x2x2
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        # M1
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        # M2
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])),
        # M3
        (np.array([1, 0, 0, 0]), np.array([0, 1, -1, 0]), np.array([0, 1, 0, 0])),
        # M4
        (np.array([0, 0, 0, 1]), np.array([0, 1, -1, 0]), np.array([0, 0, 1, 0])),
        # M5
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([1, 0, 0, 0])),
        # M6
        (np.array([0, 0, 1, -1]), np.array([1, 1, 0, 0]), np.array([0, 0, 1, 0])),
        # M7
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0]))
    ]
    
    # Mappings for sub-tensor to global indices
    # a = 3*i + j. i,j in {0,1}
    # 00->0, 01->1, 10->3, 11->4
    map_a = {0: 0, 1: 1, 2: 3, 3: 4}
    # b = 5*j + k. j,k in {0,1}
    # 00->0, 01->1, 10->5, 11->6
    map_b = {0: 0, 1: 1, 2: 5, 3: 6}
    # c = 3*k + i. k,i in {0,1}
    # 00->0, 01->1, 10->3, 11->4
    map_c = {0: 0, 1: 1, 2: 3, 3: 4}
    
    for u_sub, v_sub, w_sub in strassen_terms:
        u_global = np.zeros(nm)
        v_global = np.zeros(mp)
        w_global = np.zeros(pn)
        
        for s, g in map_a.items():
            u_global[g] = u_sub[s]
        for s, g in map_b.items():
            v_global[g] = v_sub[s]
        for s, g in map_c.items():
            w_global[g] = w_sub[s]
            
        U_new.append(u_global)
        V_new.append(v_global)
        W_new.append(w_global)
        
    U_final = np.column_stack(U_new)
    V_final = np.column_stack(V_new)
    W_final = np.column_stack(W_new)
    
    # Verify
    T_recon = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
    if not np.allclose(T_recon, T, atol=1e-10):
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
