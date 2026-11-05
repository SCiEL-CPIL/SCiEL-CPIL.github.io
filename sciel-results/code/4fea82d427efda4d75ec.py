import numpy as np

def construct():
    # 3x3x5 matrix multiplication tensor decomposition
    n, m, p = 3, 3, 5
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
    
    # Strassen terms for 2x2x2 (local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1))
    strassen_terms = [
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])),
        (np.array([1, 0, 0, 0]), np.array([0, 1, -1, 0]), np.array([0, 1, 0, 0])),
        (np.array([0, 0, 0, 1]), np.array([0, 1, -1, 0]), np.array([0, 0, 1, 0])),
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([1, 0, 0, 0])),
        (np.array([0, 0, 1, -1]), np.array([1, 1, 0, 0]), np.array([0, 0, 1, 0])),
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0]))
    ]
    
    U_list = []
    V_list = []
    W_list = []
    
    # Try all possible pairs of 2x2x2 Strassen blocks
    # A block is defined by choosing 2 i-indices, 2 j-indices, 2 k-indices
    i_choices = [(0,1), (0,2), (1,2)]
    j_choices = [(0,1), (0,2), (1,2)]
    k_choices = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    
    best_rank = nm * mp * pn  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    for i_vals in i_choices:
        for j_vals in j_choices:
            for k_vals in k_choices:
                # Determine which tensor entries are covered by this block
                block_terms = set()
                for ii in i_vals:
                    for jj in j_vals:
                        for kk in k_vals:
                            a = ii * m + jj
                            b = jj * p + kk
                            c = kk * n + ii
                            block_terms.add((a, b, c))
                
                # Build decomposition for this block
                u_list = []
                v_list = []
                w_list = []
                
                # Non-block terms as rank-1
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            if (a, b, c) in block_terms:
                                continue
                            u = np.zeros(nm)
                            v = np.zeros(mp)
                            w = np.zeros(pn)
                            u[a] = 1
                            v[b] = 1
                            w[c] = 1
                            u_list.append(u)
                            v_list.append(v)
                            w_list.append(w)
                
                # Apply Strassen to the block
                i0, i1 = i_vals
                j0, j1 = j_vals
                k0, k1 = k_vals
                
                map_a = {0: i0*m + j0, 1: i0*m + j1, 2: i1*m + j0, 3: i1*m + j1}
                map_b = {0: j0*p + k0, 1: j0*p + k1, 2: j1*p + k0, 3: j1*p + k1}
                map_c = {0: k0*n + i0, 1: k0*n + i1, 2: k1*n + i0, 3: k1*n + i1}
                
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
                    
                    u_list.append(u_global)
                    v_list.append(v_global)
                    w_list.append(w_global)
                
                rank = len(u_list)
                
                if rank < best_rank:
                    # Verify
                    U_final = np.column_stack(u_list)
                    V_final = np.column_stack(v_list)
                    W_final = np.column_stack(w_list)
                    
                    T_recon = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
                    
                    if np.allclose(T_recon, T, atol=1e-10):
                        best_rank = rank
                        best_U = U_final
                        best_V = V_final
                        best_W = W_final
    
    # Fallback to schoolbook if no valid decomposition found
    if best_U is None:
        U_school = np.zeros((nm, 45))
        V_school = np.zeros((mp, 45))
        W_school = np.zeros((pn, 45))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
