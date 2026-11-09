import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    total_rank = 28
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0

    def add_component(u_vec, v_vec, w_vec, i_off, j_off, k_off, n_l, m_l, p_l):
        nonlocal rank_offset
        # u_vec length n_l * m_l
        # v_vec length m_l * p_l
        # w_vec length p_l * n_l
        
        for idx_u in range(n_l * m_l):
            i_l, j_l = divmod(idx_u, m_l)
            i_g = i_off + i_l
            j_g = j_off + j_l
            U[i_g * m + j_g][rank_offset] += u_vec[idx_u]
            
        for idx_v in range(m_l * p_l):
            j_l, k_l = divmod(idx_v, p_l)
            j_g = j_off + j_l
            k_g = k_off + k_l
            V[j_g * p + k_g][rank_offset] += v_vec[idx_v]
            
        for idx_w in range(p_l * n_l):
            k_l, i_l = divmod(idx_w, n_l)
            k_g = k_off + k_l
            i_g = i_off + i_l
            W[k_g * n + i_g][rank_offset] += w_vec[idx_w]
            
        rank_offset += 1

    # Helper to generate Strassen components for 2x2x2
    def get_strassen_components():
        # Returns list of (u, v, w) for 7 terms
        # u, v, w are lists of 4 elements (flattened 2x2)
        # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        components = []
        
        # P1 = (A00+A11)(B00+B11)
        # u: 1, 0, 0, 1
        # v: 1, 0, 0, 1
        # w: 1, 0, 0, 1 (C00+=1, C11+=1? No, w maps to C)
        # C00 = P1 + ...
        # C11 = P1 - ...
        # So w for P1 is 1 at C00 and 1 at C11.
        # w indices: k*n+i.
        # C00: k=0, i=0 -> idx 0
        # C11: k=1, i=1 -> idx 3
        # w: 1, 0, 0, 1
        
        # Let's define w vectors carefully.
        # w is length 4 (2x2).
        # w[idx] contributes to C[i, k] where idx = k*2 + i.
        
        # P1
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        w1 = [1, 0, 0, 1] # C00 += 1, C11 += 1
        components.append((u1, v1, w1))
        
        # P2 = (A10+A11)B00
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        # C10 += P2
        # C10: k=0, i=1 -> idx 1
        w2 = [0, 1, 0, 0]
        components.append((u2, v2, w2))
        
        # P3 = A00(B10-B11)
        u3 = [1, 0, 0, 0]
        v3 = [0, 0, 1, -1]
        # C01 += P3
        # C01: k=1, i=0 -> idx 2
        w3 = [0, 0, 1, 0]
        components.append((u3, v3, w3))
        
        # P4 = A11(B01-B00)
        u4 = [0, 0, 0, 1]
        v4 = [-1, 1, 0, 0]
        # C00 += P4, C10 += P4
        # C00: idx 0, C10: idx 1
        w4 = [1, 1, 0, 0]
        components.append((u4, v4, w4))
        
        # P5 = (A00+A01)B11
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        # C00 -= P5, C01 += P5
        # C00: idx 0, C01: idx 2
        w5 = [-1, 0, 1, 0]
        components.append((u5, v5, w5))
        
        # P6 = (A10-A00)(B00+B01)
        u6 = [-1, 0, 1, 0]
        v6 = [1, 1, 0, 0]
        # C11 += P6
        # C11: idx 3
        w6 = [0, 0, 0, 1]
        components.append((u6, v6, w6))
        
        # P7 = (A01-A11)(B10+B11)
        u7 = [0, 1, 0, -1]
        v7 = [0, 0, 1, 1]
        # C00 += P7
        # C00: idx 0
        w7 = [1, 0, 0, 0]
        components.append((u7, v7, w7))
        
        return components

    # Helper to generate Schoolbook components for n_l x m_l x p_l
    def get_schoolbook_components(n_l, m_l, p_l):
        components = []
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    u = [0] * (n_l * m_l)
                    v = [0] * (m_l * p_l)
                    w = [0] * (p_l * n_l)
                    
                    u[i * m_l + j] = 1
                    v[j * p_l + k] = 1
                    w[k * n_l + i] = 1
                    components.append((u, v, w))
        return components

    # Block 1a: 2x2x2 at (0,0,0)
    comps = get_strassen_components()
    for u, v, w in comps:
        add_component(u, v, w, 0, 0, 0, 2, 2, 2)

    # Block 1b: 2x1x2 at (0,2,0)
    comps = get_schoolbook_components(2, 1, 2)
    for u, v, w in comps:
        add_component(u, v, w, 0, 2, 0, 2, 1, 2)

    # Block 2a-i: 2x2x2 at (0,0,2)
    comps = get_strassen_components()
    for u, v, w in comps:
        add_component(u, v, w, 0, 0, 2, 2, 2, 2)

    # Block 2a-ii: 2x2x1 at (0,0,4)
    comps = get_schoolbook_components(2, 2, 1)
    for u, v, w in comps:
        add_component(u, v, w, 0, 0, 4, 2, 2, 1)

    # Block 2b: 2x1x3 at (0,2,2)
    comps = get_schoolbook_components(2, 1, 3)
    for u, v, w in comps:
        add_component(u, v, w, 0, 2, 2, 2, 1, 3)

    return {"U": U, "V": V, "W": W}
