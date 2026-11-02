import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Finds a rank-50 exact tensor decomposition for 2x4x7 matrix multiplication
    using a structural combination of Strassen and schoolbook algorithms.
    """
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    dim_u = n * m  # 8
    dim_v = m * p  # 28
    dim_w = p * n  # 14
    
    # We will build the decomposition by collecting rank-1 terms (U, V, W vectors)
    # and then concatenating them into the final matrices.
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    def add_term(u_vec, v_vec, w_vec):
        terms_U.append(u_vec)
        terms_V.append(v_vec)
        terms_W.append(w_vec)
        
    # Helper to generate Strassen terms for 2x2x2
    # u_map, v_map, w_map: functions mapping local index to global index
    def add_strassen(u_map, v_map, w_map):
        # Local indices for A (2x2): 0,1,2,3
        # Local indices for B (2x2): 0,1,2,3
        # Local indices for C (2x2): 0,1,2,3
        
        # M1 = (A00 + A11)(B00 + B11)
        u = np.zeros(4); u[0]=1; u[3]=1
        v = np.zeros(4); v[0]=1; v[3]=1
        w = np.zeros(4); w[0]=1; w[3]=1 # C00 + C11
        # Need to map w to global.
        # We pass maps that take local index.
        # But w_vec needs to be global size.
        # Let's create global vectors directly.
        
        def make_global(vec, map_fn, dim):
            g = np.zeros(dim)
            for i, val in enumerate(vec):
                if val != 0:
                    g[map_fn(i)] = val
            return g

        # M1
        u_g = make_global([1, 0, 0, 1], u_map, dim_u)
        v_g = make_global([1, 0, 0, 1], v_map, dim_v)
        w_g = make_global([1, 0, 0, 1], w_map, dim_w)
        add_term(u_g, v_g, w_g)
        
        # M2 = (A10 + A11)B00
        u_g = make_global([0, 0, 1, 1], u_map, dim_u)
        v_g = make_global([1, 0, 0, 0], v_map, dim_v)
        w_g = make_global([0, 0, 1, -1], w_map, dim_w) # C10 + C11(-1)
        add_term(u_g, v_g, w_g)
        
        # M3 = A00(B01 - B11)
        u_g = make_global([1, 0, 0, 0], u_map, dim_u)
        v_g = make_global([0, 1, 0, -1], v_map, dim_v)
        w_g = make_global([0, 1, 0, 1], w_map, dim_w) # C01 + C11
        add_term(u_g, v_g, w_g)
        
        # M4 = A11(B10 - B00)
        u_g = make_global([0, 0, 0, 1], u_map, dim_u)
        v_g = make_global([0, 0, 1, -1], v_map, dim_v)
        w_g = make_global([1, 0, 1, 0], w_map, dim_w) # C00 + C10
        add_term(u_g, v_g, w_g)
        
        # M5 = (A00 + A01)B11
        u_g = make_global([1, 1, 0, 0], u_map, dim_u)
        v_g = make_global([0, 0, 0, 1], v_map, dim_v)
        w_g = make_global([-1, 1, 0, 0], w_map, dim_w) # -C00 + C01
        add_term(u_g, v_g, w_g)
        
        # M6 = (A10 - A00)(B00 + B01)
        u_g = make_global([-1, 0, 1, 0], u_map, dim_u)
        v_g = make_global([1, 1, 0, 0], v_map, dim_v)
        w_g = make_global([0, 0, 0, 1], w_map, dim_w) # C11
        add_term(u_g, v_g, w_g)
        
        # M7 = (A01 - A11)(B10 + B11)
        u_g = make_global([0, 1, 0, -1], u_map, dim_u)
        v_g = make_global([0, 0, 1, 1], v_map, dim_v)
        w_g = make_global([1, 0, 0, 0], w_map, dim_w) # C00
        add_term(u_g, v_g, w_g)

    # Helper to generate Schoolbook terms for 2x2x1
    def add_schoolbook_2x2x1(u_map, v_map, w_map):
        # Local A: 0,1,2,3
        # Local B: 0,1
        # Local C: 0,1
        
        def make_global(vec, map_fn, dim):
            g = np.zeros(dim)
            for i, val in enumerate(vec):
                if val != 0:
                    g[map_fn(i)] = val
            return g

        # A00 B0 -> C0
        u_g = make_global([1, 0, 0, 0], u_map, dim_u)
        v_g = make_global([1, 0], v_map, dim_v)
        w_g = make_global([1, 0], w_map, dim_w)
        add_term(u_g, v_g, w_g)
        
        # A01 B1 -> C0
        u_g = make_global([0, 1, 0, 0], u_map, dim_u)
        v_g = make_global([0, 1], v_map, dim_v)
        w_g = make_global([1, 0], w_map, dim_w)
        add_term(u_g, v_g, w_g)
        
        # A10 B0 -> C1
        u_g = make_global([0, 0, 1, 0], u_map, dim_u)
        v_g = make_global([1, 0], v_map, dim_v)
        w_g = make_global([0, 1], w_map, dim_w)
        add_term(u_g, v_g, w_g)
        
        # A11 B1 -> C1
        u_g = make_global([0, 0, 0, 1], u_map, dim_u)
        v_g = make_global([0, 1], v_map, dim_v)
        w_g = make_global([0, 1], w_map, dim_w)
        add_term(u_g, v_g, w_g)

    # Define mappings for Block 1: A1 B1
    # A1 is A[:, 0:2]
    # Local A indices 0,1,2,3 map to global U indices 0,1,2,3
    u_map_1 = lambda i: i
    
    # B1 is B[0:2, :]
    # Local B indices (j, k) -> j*2+k? No, Strassen uses 0..3 for 2x2.
    # We call Strassen on 2x2 blocks of B.
    
    # We need to handle the 7 columns of B1.
    # Cols 0,1: Strassen
    # Cols 2,3: Strassen
    # Cols 4,5: Strassen
    # Col 6: Schoolbook
    
    # For Strassen on cols offset by `col_offset`:
    # Local B index `l` (0..3) corresponds to:
    # l=0 -> (0, col_offset)
    # l=1 -> (0, col_offset+1)
    # l=2 -> (1, col_offset)
    # l=3 -> (1, col_offset+1)
    # Global V index: row * 7 + col
    
    def v_map_strassen(col_offset):
        def map_fn(l):
            row = l // 2
            col = l % 2
            return row * 7 + (col_offset + col)
        return map_fn

    # For Schoolbook on col `col_offset`:
    # Local B index `l` (0,1) corresponds to:
    # l=0 -> (0, col_offset)
    # l=1 -> (1, col_offset)
    def v_map_schoolbook(col_offset):
        def map_fn(l):
            row = l
            return row * 7 + col_offset
        return map_fn

    # W mapping for C
    # Local C index `l` (0..3 for Strassen, 0..1 for Schoolbook)
    # Strassen C indices:
    # 0 -> (0, 0)
    # 1 -> (0, 1)
    # 2 -> (1, 0)
    # 3 -> (1, 1)
    # Global W index: col * 2 + row
    
    def w_map_strassen(col_offset):
        def map_fn(l):
            row = l // 2
            col = l % 2
            return (col_offset + col) * 2 + row
        return map_fn

    def w_map_schoolbook(col_offset):
        def map_fn(l):
            row = l
            return col_offset * 2 + row
        return map_fn

    # Execute Block 1: A1 B1
    # Cols 0,1
    add_strassen(u_map_1, v_map_strassen(0), w_map_strassen(0))
    # Cols 2,3
    add_strassen(u_map_1, v_map_strassen(2), w_map_strassen(2))
    # Cols 4,5
    add_strassen(u_map_1, v_map_strassen(4), w_map_strassen(4))
    # Col 6
    add_schoolbook_2x2x1(u_map_1, v_map_schoolbook(6), w_map_schoolbook(6))

    # Define mappings for Block 2: A2 B2
    # A2 is A[:, 2:4]
    # Local A indices 0,1,2,3 map to global U indices:
    # 0 -> A[0,2] = 2
    # 1 -> A[0,3] = 3
    # 2 -> A[1,2] = 6
    # 3 -> A[1,3] = 7
    u_map_2 = lambda i: [2, 3, 6, 7][i]
    
    # B2 is B[2:4, :]
    # Local B indices (j, k) -> j*2+k?
    # Strassen local 0..3:
    # 0 -> (0, col) -> Global Row 2, Col col
    # 1 -> (0, col+1) -> Global Row 2, Col col+1
    # 2 -> (1, col) -> Global Row 3, Col col
    # 3 -> (1, col+1) -> Global Row 3, Col col+1
    
    def v_map_strassen_2(col_offset):
        def map_fn(l):
            local_row = l // 2
            col = l % 2
            global_row = local_row + 2
            return global_row * 7 + (col_offset + col)
        return map_fn

    def v_map_schoolbook_2(col_offset):
        def map_fn(l):
            local_row = l
            global_row = local_row + 2
            return global_row * 7 + col_offset
        return map_fn

    # Execute Block 2: A2 B2
    # Cols 0,1
    add_strassen(u_map_2, v_map_strassen_2(0), w_map_strassen(0))
    # Cols 2,3
    add_strassen(u_map_2, v_map_strassen_2(2), w_map_strassen(2))
    # Cols 4,5
    add_strassen(u_map_2, v_map_strassen_2(4), w_map_strassen(4))
    # Col 6
    add_schoolbook_2x2x1(u_map_2, v_map_schoolbook_2(6), w_map_schoolbook(6))

    # Construct final matrices
    rank = len(terms_U)
    
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    for i in range(rank):
        U[:, i] = terms_U[i]
        V[:, i] = terms_V[i]
        W[:, i] = terms_W[i]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
