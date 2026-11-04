import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank is expected to be 88
    # We will build the columns of U, V, W
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        """
        u: list of length n*m (12)
        v: list of length m*p (32)
        w: list of length p*n (24)
        """
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # Helper to create sparse vectors
    def make_u(indices, values):
        vec = [0.0] * (n * m)
        for idx, val in zip(indices, values):
            vec[idx] += val
        return vec

    def make_v(indices, values):
        vec = [0.0] * (m * p)
        for idx, val in zip(indices, values):
            vec[idx] += val
        return vec

    def make_w(indices, values):
        vec = [0.0] * (p * n)
        for idx, val in zip(indices, values):
            vec[idx] += val
        return vec

    # Strassen decomposition for 2x2 * 2x2
    # Inputs: A (2x2), B (2x2)
    # A indices in U: (i, j) -> i*m + j
    # B indices in V: (j, k) -> j*p + k
    # C indices in W: (k, i) -> k*n + i
    # Note: The local 2x2 block corresponds to specific global indices.
    
    # We will iterate over the 4 column blocks of B
    for c in range(4):
        k0 = 2 * c
        k1 = 2 * c + 1
        
        # --- Top Part (i=0,1) ---
        # A_top is rows 0,1 of A.
        # Split A_top into A_t1 (cols 0,1) and A_t2 (cols 2,3)
        
        # B_c is cols k0, k1 of B.
        # Split B_c into B_1c (rows 0,1) and B_2c (rows 2,3)
        
        # Product 1: A_t1 * B_1c
        # A_t1 indices: (0,0), (0,1), (1,0), (1,1) -> U idx: 0,1,4,5
        # B_1c indices: (0,k0), (0,k1), (1,k0), (1,k1) -> V idx: 0*p+k0, 0*p+k1, 1*p+k0, 1*p+k1
        # C indices: (k0,0), (k1,0), (k0,1), (k1,1) -> W idx: k0*n+0, k1*n+0, k0*n+1, k1*n+1
        
        # Strassen terms for A_t1 * B_1c
        # M1 = (A00+A11)(B00+B11)
        # A00: U[0], A11: U[5]
        # B00: V[k0], B11: V[p+k1] (since j=1, k=k1 -> 1*p+k1)
        u_m1 = make_u([0, 5], [1, 1])
        v_m1 = make_v([k0, p + k1], [1, 1])
        w_m1 = make_w([k0*n+0, k1*n+0, k0*n+1, k1*n+1], [1, 1, -1, -1]) 
        # Wait, C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's define the W contributions for each Strassen multiplier
        # M1 contributes: C00(+1), C11(+1)
        # M2 contributes: C10(+1), C11(-1)
        # M3 contributes: C01(+1), C11(+1)
        # M4 contributes: C00(+1), C10(+1)
        # M5 contributes: C00(-1), C01(+1)
        # M6 contributes: C11(+1)
        # M7 contributes: C00(+1)
        
        # Re-verify Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B11+B01)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # W indices for C:
        # C00 -> (k0, 0) -> k0*n + 0
        # C01 -> (k1, 0) -> k1*n + 0
        # C10 -> (k0, 1) -> k0*n + 1
        # C11 -> (k1, 1) -> k1*n + 1
        
        w_idx_c00 = k0*n + 0
        w_idx_c01 = k1*n + 0
        w_idx_c10 = k0*n + 1
        w_idx_c11 = k1*n + 1
        
        # M1
        add_term(u_m1, v_m1, make_w([w_idx_c00, w_idx_c11], [1, 1]))
        
        # M2 = (A10+A11)B00
        u_m2 = make_u([4, 5], [1, 1]) # A10 is (1,0)->4, A11 is (1,1)->5
        v_m2 = make_v([k0], [1])      # B00 is (0,k0)->k0
        add_term(u_m2, v_m2, make_w([w_idx_c10, w_idx_c11], [1, -1]))
        
        # M3 = A00(B10-B11)
        u_m3 = make_u([0], [1])       # A00
        v_m3 = make_v([p+k0, p+k1], [1, -1]) # B10 is (1,k0)->p+k0, B11 is (1,k1)->p+k1
        add_term(u_m3, v_m3, make_w([w_idx_c01, w_idx_c11], [1, 1]))
        
        # M4 = A11(B01-B00)
        u_m4 = make_u([5], [1])       # A11
        v_m4 = make_v([k1, k0], [1, -1]) # B01 is (0,k1)->k1, B00 is (0,k0)->k0
        add_term(u_m4, v_m4, make_w([w_idx_c00, w_idx_c10], [1, 1]))
        
        # M5 = (A00+A01)B11
        u_m5 = make_u([0, 1], [1, 1]) # A00, A01
        v_m5 = make_v([p+k1], [1])    # B11
        add_term(u_m5, v_m5, make_w([w_idx_c00, w_idx_c01], [-1, 1]))
        
        # M6 = (A10-A00)(B00+B10)
        u_m6 = make_u([4, 0], [1, -1]) # A10, -A00
        v_m6 = make_v([k0, p+k0], [1, 1]) # B00, B10
        add_term(u_m6, v_m6, make_w([w_idx_c11], [1]))
        
        # M7 = (A01-A11)(B11+B01)
        u_m7 = make_u([1, 5], [1, -1]) # A01, -A11
        v_m7 = make_v([p+k1, k1], [1, 1]) # B11, B01
        add_term(u_m7, v_m7, make_w([w_idx_c00], [1]))
        
        # Product 2: A_t2 * B_2c
        # A_t2 indices: (0,2), (0,3), (1,2), (1,3) -> U idx: 2,3,6,7
        # B_2c indices: (2,k0), (2,k1), (3,k0), (3,k1) -> V idx: 2*p+k0, 2*p+k1, 3*p+k0, 3*p+k1
        
        # M1' = (A00'+A11')(B00'+B11')
        # A00' is (0,2)->2, A11' is (1,3)->7
        # B00' is (2,k0)->2p+k0, B11' is (3,k1)->3p+k1
        u_m1p = make_u([2, 7], [1, 1])
        v_m1p = make_v([2*p+k0, 3*p+k1], [1, 1])
        add_term(u_m1p, v_m1p, make_w([w_idx_c00, w_idx_c11], [1, 1]))
        
        # M2' = (A10'+A11')B00'
        # A10' is (1,2)->6, A11' is (1,3)->7
        # B00' is (2,k0)->2p+k0
        u_m2p = make_u([6, 7], [1, 1])
        v_m2p = make_v([2*p+k0], [1])
        add_term(u_m2p, v_m2p, make_w([w_idx_c10, w_idx_c11], [1, -1]))
        
        # M3' = A00'(B10'-B11')
        # A00' is (0,2)->2
        # B10' is (3,k0)->3p+k0, B11' is (3,k1)->3p+k1
        u_m3p = make_u([2], [1])
        v_m3p = make_v([3*p+k0, 3*p+k1], [1, -1])
        add_term(u_m3p, v_m3p, make_w([w_idx_c01, w_idx_c11], [1, 1]))
        
        # M4' = A11'(B01'-B00')
        # A11' is (1,3)->7
        # B01' is (2,k1)->2p+k1, B00' is (2,k0)->2p+k0
        u_m4p = make_u([7], [1])
        v_m4p = make_v([2*p+k1, 2*p+k0], [1, -1])
        add_term(u_m4p, v_m4p, make_w([w_idx_c00, w_idx_c10], [1, 1]))
        
        # M5' = (A00'+A01')B11'
        # A00' is (0,2)->2, A01' is (0,3)->3
        # B11' is (3,k1)->3p+k1
        u_m5p = make_u([2, 3], [1, 1])
        v_m5p = make_v([3*p+k1], [1])
        add_term(u_m5p, v_m5p, make_w([w_idx_c00, w_idx_c01], [-1, 1]))
        
        # M6' = (A10'-A00')(B00'+B10')
        # A10' is (1,2)->6, A00' is (0,2)->2
        # B00' is (2,k0)->2p+k0, B10' is (3,k0)->3p+k0
        u_m6p = make_u([6, 2], [1, -1])
        v_m6p = make_v([2*p+k0, 3*p+k0], [1, 1])
        add_term(u_m6p, v_m6p, make_w([w_idx_c11], [1]))
        
        # M7' = (A01'-A11')(B11'+B01')
        # A01' is (0,3)->3, A11' is (1,3)->7
        # B11' is (3,k1)->3p+k1, B01' is (2,k1)->2p+k1
        u_m7p = make_u([3, 7], [1, -1])
        v_m7p = make_v([3*p+k1, 2*p+k1], [1, 1])
        add_term(u_m7p, v_m7p, make_w([w_idx_c00], [1]))

        # --- Bottom Part (i=2) ---
        # A_bot is row 2 of A.
        # Split A_bot into A_b1 (cols 0,1) and A_b2 (cols 2,3)
        # A_b1 indices: (2,0)->8, (2,1)->9
        # A_b2 indices: (2,2)->10, (2,3)->11
        
        # B_1c indices: (0,k0)->k0, (0,k1)->k1, (1,k0)->p+k0, (1,k1)->p+k1
        # B_2c indices: (2,k0)->2p+k0, (2,k1)->2p+k1, (3,k0)->3p+k0, (3,k1)->3p+k1
        
        # C_bot_c indices: (k0,2)->k0*n+2, (k1,2)->k1*n+2
        
        w_idx_bot_c0 = k0*n + 2
        w_idx_bot_c1 = k1*n + 2
        
        # A_b1 * B_1c
        # Terms:
        # (2,0)*(0,k0) -> C(k0,2)
        add_term(make_u([8], [1]), make_v([k0], [1]), make_w([w_idx_bot_c0], [1]))
        # (2,0)*(0,k1) -> C(k1,2)
        add_term(make_u([8], [1]), make_v([k1], [1]), make_w([w_idx_bot_c1], [1]))
        # (2,1)*(1,k0) -> C(k0,2)
        add_term(make_u([9], [1]), make_v([p+k0], [1]), make_w([w_idx_bot_c0], [1]))
        # (2,1)*(1,k1) -> C(k1,2)
        add_term(make_u([9], [1]), make_v([p+k1], [1]), make_w([w_idx_bot_c1], [1]))
        
        # A_b2 * B_2c
        # Terms:
        # (2,2)*(2,k0) -> C(k0,2)
        add_term(make_u([10], [1]), make_v([2*p+k0], [1]), make_w([w_idx_bot_c0], [1]))
        # (2,2)*(2,k1) -> C(k1,2)
        add_term(make_u([10], [1]), make_v([2*p+k1], [1]), make_w([w_idx_bot_c1], [1]))
        # (2,3)*(3,k0) -> C(k0,2)
        add_term(make_u([11], [1]), make_v([3*p+k0], [1]), make_w([w_idx_bot_c0], [1]))
        # (2,3)*(3,k1) -> C(k1,2)
        add_term(make_u([11], [1]), make_v([3*p+k1], [1]), make_w([w_idx_bot_c1], [1]))

    rank = len(U_cols)
    
    # Convert lists of columns to lists of rows
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
