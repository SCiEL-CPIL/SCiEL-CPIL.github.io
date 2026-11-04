"""Matrix multiplication 3,3,5 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 15, 15
    
    # Build the target tensor T (flattened)
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for consistent i,j,k
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
    
    # Strategy: Replace the 8 schoolbook terms for i,j,k in {0,1} with 7 Strassen terms
    # This reduces rank from 45 to 44.
    
    # Indices for the 2x2x2 sub-tensor (i,j,k in {0,1})
    # a in {0, 1, 3, 4}
    # b in {0, 1, 5, 6}
    # c in {0, 1, 3, 4}
    
    # Identify the 8 columns in U_school, V_school, W_school corresponding to these terms
    # Term (i,j,k) is at index i*m*p + j*p + k in the loop? No, col increments.
    # We can just rebuild the factors.
    
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
    
    # Add the 7 Strassen terms for the 2x2x2 sub-tensor
    # The sub-tensor is defined on i,j,k in {0,1}.
    # We need to map the 2x2 indices (0,1) to the global indices.
    # Global a = 3*i + j. For i,j in {0,1}:
    # (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4.
    # Map: sub_idx -> global_idx
    # sub 0 (0,0) -> 0
    # sub 1 (0,1) -> 1
    # sub 2 (1,0) -> 3
    # sub 3 (1,1) -> 4
    
    # b = 5*j + k. For j,k in {0,1}:
    # (0,0)->0, (0,1)->1, (1,0)->5, (1,1)->6.
    # Map:
    # sub 0 (0,0) -> 0
    # sub 1 (0,1) -> 1
    # sub 2 (1,0) -> 5
    # sub 3 (1,1) -> 6
    
    # c = 3*k + i. For k,i in {0,1}:
    # (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4.
    # Map:
    # sub 0 (0,0) -> 0
    # sub 1 (0,1) -> 1
    # sub 2 (1,0) -> 3
    # sub 3 (1,1) -> 4
    
    def embed_sub(vec_sub, mapping):
        """Embed a 4-dim vector into global dim using mapping {sub_idx: global_idx}"""
        vec_global = np.zeros(nm if mapping == 'a' else mp if mapping == 'b' else pn)
        # Actually, we need to know which dimension it is.
        return vec_global

    # We will construct the 7 vectors explicitly.
    # Strassen terms (u, v, w) in 2x2 basis (e0, e1, e2, e3)
    # e0=(1,0,0,0), e1=(0,1,0,0), e2=(0,0,1,0), e3=(0,0,0,1)
    
    strassen_terms = [
        # M1
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        # M2
        (np.array([0, 1, 0, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])),
        # M3
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 0])),
        # M4
        (np.array([0, 0, 0, 1]), np.array([0, 1, -1, 0]), np.array([0, 0, 1, 0])),
        # M5
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([1, 0, 0, 0])),
        # M6
        (np.array([0, 1, -1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1])),
        # M7
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0]))
    ]
    
    # Mappings
    map_a = {0: 0, 1: 1, 2: 3, 3: 4}
    map_b = {0: 0, 1: 1, 2: 5, 3: 6}
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
        # Fallback to schoolbook if verification fails
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
