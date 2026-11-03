import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We will build the decomposition for 4x4x4 (rank 49) and 4x4x1 (rank 16).
    # Total rank = 49 + 49 + 16 = 114.
    
    # Helper to add terms to U, V, W
    # U: 16 x rank
    # V: 36 x rank
    # W: 36 x rank
    
    U = []
    V = []
    W = []
    
    def add_term(u_dict, v_dict, w_dict):
        """
        u_dict: { (i, j): val }
        v_dict: { (j, k): val }
        w_dict: { (k, i): val }
        """
        u_vec = [0.0] * 16
        for (i, j), val in u_dict.items():
            u_vec[i * 4 + j] = val
            
        v_vec = [0.0] * 36
        for (j, k), val in v_dict.items():
            v_vec[j * 9 + k] = val
            
        w_vec = [0.0] * 36
        for (k, i), val in w_dict.items():
            w_vec[k * 4 + i] = val
            
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # --- 2x2x2 Strassen Terms ---
    # Returns list of 7 dicts: (u_dict, v_dict, w_dict)
    def get_2x2_strassen_terms():
        terms = []
        # M1 = a(f-h) -> C00, C11
        # a: A[0,0], f: B[0,1], h: B[1,1]
        # C00: C[0,0], C11: C[1,1]
        terms.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,0): 1, (1,1): 1}
        ))
        # M2 = (a+c)h -> C10
        # a: A[0,0], c: A[1,0], h: B[1,1]
        # C10: C[1,0]
        terms.append((
            {(0,0): 1, (1,0): 1},
            {(1,1): 1},
            {(1,0): 1}
        ))
        # M3 = (b-d)(g+h) -> C01, C11
        # b: A[0,1], d: A[1,1], g: B[1,0], h: B[1,1]
        # C01: C[0,1], C11: C[1,1]
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,1): 1, (1,1): 1}
        ))
        # M4 = d(e-f) -> C00, C10
        # d: A[1,1], e: B[0,0], f: B[0,1]
        # C00: C[0,0], C10: C[1,0]
        terms.append((
            {(1,1): 1},
            {(0,0): 1, (0,1): -1},
            {(0,0): 1, (1,0): 1}
        ))
        # M5 = (a+b)e -> C00, C01
        # a: A[0,0], b: A[0,1], e: B[0,0]
        # C00: C[0,0], C01: C[0,1]
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(0,0): 1},
            {(0,0): 1, (0,1): 1}
        ))
        # M6 = (c-a)(e+f) -> C11
        # c: A[1,0], a: A[0,0], e: B[0,0], f: B[0,1]
        # C11: C[1,1]
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ))
        # M7 = (d-c)(g-h) -> C00, C11
        # d: A[1,1], c: A[1,0], g: B[1,0], h: B[1,1]
        # C00: C[0,0], C11: C[1,1]
        terms.append((
            {(1,1): 1, (1,0): -1},
            {(1,0): 1, (1,1): -1},
            {(0,0): 1, (1,1): 1}
        ))
        return terms

    inner_terms = get_2x2_strassen_terms()

    # --- 4x4x4 Strassen Terms ---
    # Outer Strassen:
    # M1 = A00 * (B01 - B11) -> C00, C11
    # M2 = (A00 + A10) * B11 -> C10
    # M3 = (A01 - A11) * (B10 + B11) -> C01, C11
    # M4 = A11 * (B00 - B01) -> C00, C10
    # M5 = (A00 + A01) * B00 -> C00, C01
    # M6 = (A10 - A00) * (B00 + B01) -> C11
    # M7 = (A11 - A10) * (B10 - B11) -> C00, C11

    # We need to map the inner 2x2 terms to the 4x4 blocks.
    # For each outer term, we define:
    # - A_block_map: {(local_i, local_j): (global_i, global_j, coeff)}
    # - B_block_map: {(local_j, local_k): (global_j, global_k, coeff)}
    # - C_block_map: {(local_k, local_i): (global_k, global_i, coeff)}

    def process_outer_term(a_map, b_map, c_map):
        for u_inner, v_inner, w_inner in inner_terms:
            u_global = {}
            v_global = {}
            w_global = {}
            
            # Expand U
            for (li, lj), val in u_inner.items():
                if (li, lj) in a_map:
                    gi, gj, coeff = a_map[(li, lj)]
                    key = (gi, gj)
                    u_global[key] = u_global.get(key, 0) + val * coeff
            
            # Expand V
            for (lj, lk), val in v_inner.items():
                if (lj, lk) in b_map:
                    gj, gk, coeff = b_map[(lj, lk)]
                    key = (gj, gk)
                    v_global[key] = v_global.get(key, 0) + val * coeff
            
            # Expand W
            for (lk, li), val in w_inner.items():
                if (lk, li) in c_map:
                    gk, gi, coeff = c_map[(lk, li)]
                    key = (gk, gi)
                    w_global[key] = w_global.get(key, 0) + val * coeff
            
            if u_global and v_global and w_global:
                add_term(u_global, v_global, w_global)

    # Define maps for each outer term
    # A00: (0,0)-(1,1), A01: (0,2)-(1,3), A10: (2,0)-(3,1), A11: (2,2)-(3,3)
    # B00: (0,0)-(1,1), B01: (0,2)-(1,3), B10: (2,0)-(3,1), B11: (2,2)-(3,3)
    # C00: (0,0)-(1,1), C01: (0,2)-(1,3), C10: (2,0)-(3,1), C11: (2,2)-(3,3)
    
    def block_map(start_i, start_j, sign=1):
        m = {}
        for i in range(2):
            for j in range(2):
                m[(i,j)] = (start_i + i, start_j + j, sign)
        return m

    # M1 = A00 * (B01 - B11) -> C00, C11
    a_map = block_map(0, 0, 1)
    b_map = {}
    b_map.update(block_map(0, 2, 1)) # B01
    b_map.update(block_map(2, 2, -1)) # B11
    c_map = {}
    c_map.update(block_map(0, 0, 1)) # C00
    c_map.update(block_map(2, 2, 1)) # C11
    process_outer_term(a_map, b_map, c_map)

    # M2 = (A00 + A10) * B11 -> C10
    a_map = {}
    a_map.update(block_map(0, 0, 1)) # A00
    a_map.update(block_map(2, 0, 1)) # A10
    b_map = block_map(2, 2, 1) # B11
    c_map = block_map(2, 0, 1) # C10
    process_outer_term(a_map, b_map, c_map)

    # M3 = (A01 - A11) * (B10 + B11) -> C01, C11
    a_map = {}
    a_map.update(block_map(0, 2, 1)) # A01
    a_map.update(block_map(2, 2, -1)) # A11
    b_map = {}
    b_map.update(block_map(2, 0, 1)) # B10
    b_map.update(block_map(2, 2, 1)) # B11
    c_map = {}
    c_map.update(block_map(0, 2, 1)) # C01
    c_map.update(block_map(2, 2, 1)) # C11
    process_outer_term(a_map, b_map, c_map)

    # M4 = A11 * (B00 - B01) -> C00, C10
    a_map = block_map(2, 2, 1) # A11
    b_map = {}
    b_map.update(block_map(0, 0, 1)) # B00
    b_map.update(block_map(0, 2, -1)) # B01
    c_map = {}
    c_map.update(block_map(0, 0, 1)) # C00
    c_map.update(block_map(2, 0, 1)) # C10
    process_outer_term(a_map, b_map, c_map)

    # M5 = (A00 + A01) * B00 -> C00, C01
    a_map = {}
    a_map.update(block_map(0, 0, 1)) # A00
    a_map.update(block_map(0, 2, 1)) # A01
    b_map = block_map(0, 0, 1) # B00
    c_map = {}
    c_map.update(block_map(0, 0, 1)) # C00
    c_map.update(block_map(0, 2, 1)) # C01
    process_outer_term(a_map, b_map, c_map)

    # M6 = (A10 - A00) * (B00 + B01) -> C11
    a_map = {}
    a_map.update(block_map(2, 0, 1)) # A10
    a_map.update(block_map(0, 0, -1)) # A00
    b_map = {}
    b_map.update(block_map(0, 0, 1)) # B00
    b_map.update(block_map(0, 2, 1)) # B01
    c_map = block_map(2, 2, 1) # C11
    process_outer_term(a_map, b_map, c_map)

    # M7 = (A11 - A10) * (B10 - B11) -> C00, C11
    a_map = {}
    a_map.update(block_map(2, 2, 1)) # A11
    a_map.update(block_map(2, 0, -1)) # A10
    b_map = {}
    b_map.update(block_map(2, 0, 1)) # B10
    b_map.update(block_map(2, 2, -1)) # B11
    c_map = {}
    c_map.update(block_map(0, 0, 1)) # C00
    c_map.update(block_map(2, 2, 1)) # C11
    process_outer_term(a_map, b_map, c_map)

    # Now add the second 4x4x4 block (k=4..7)
    # We need to shift the k indices in V and W by 4.
    # U is the same.
    
    U_start = len(U)
    # Copy U
    for u in U:
        U.append(u[:])
    
    # Shift V and W
    for idx in range(U_start):
        v_vec = V[idx][:]
        w_vec = W[idx][:]
        
        v_new = [0.0] * 36
        w_new = [0.0] * 36
        
        for j in range(4):
            for k in range(4):
                # Original k is 0..3, new k is 4..7
                v_new[j * 9 + (k + 4)] = v_vec[j * 9 + k]
                
        for k in range(4):
            for i in range(4):
                # Original k is 0..3, new k is 4..7
                w_new[(k + 4) * 4 + i] = w_vec[k * 4 + i]
                
        V.append(v_new)
        W.append(w_new)

    # Now add the 4x4x1 block (k=8)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0.0] * 16
            u_vec[i * 4 + j] = 1.0
            
            v_vec = [0.0] * 36
            v_vec[j * 9 + 8] = 1.0
            
            w_vec = [0.0] * 36
            w_vec[8 * 4 + i] = 1.0
            
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)

    # Convert to lists of integers (or floats if needed, but they are integers)
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
