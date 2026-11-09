import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    # U shape: (n*m, rank) = (12, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (24, rank)
    
    # We will build lists of columns for U, V, W
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: list or array of length n*m (12)
        v_vec: list or array of length m*p (32)
        w_vec: list or array of length p*n (24)
        """
        U_cols.append(np.array(u_vec))
        V_cols.append(np.array(v_vec))
        W_cols.append(np.array(w_vec))
        
    def idx_A(i, j):
        return i * m + j
        
    def idx_B(j, k):
        return j * p + k
        
    def idx_C(i, k):
        # Convention: W[k*n+i, r]
        # So index in W vector is k*n + i
        return k * n + i

    # Strassen block for M_2,2,2
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global offsets: i0, j0, k0
    def add_strassen(i0, j0, k0):
        # 7 products
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Define linear forms for A (u) and B (v) for each Mt
        # And contributions to C (w)
        
        # Helper to create sparse vectors
        def make_A(coeffs):
            # coeffs: dict { (i,j): val }
            vec = [0.0] * (n*m)
            for (i,j), val in coeffs.items():
                vec[idx_A(i0+i, j0+j)] = val
            return vec
            
        def make_B(coeffs):
            # coeffs: dict { (j,k): val }
            vec = [0.0] * (m*p)
            for (j,k), val in coeffs.items():
                vec[idx_B(j0+j, k0+k)] = val
            return vec
            
        def make_C(coeffs):
            # coeffs: dict { (i,k): val }
            vec = [0.0] * (p*n)
            for (i,k), val in coeffs.items():
                vec[idx_C(i0+i, k0+k)] = val
            return vec

        # M1
        u1 = make_A({(0,0): 1, (1,1): 1})
        v1 = make_B({(0,0): 1, (1,1): 1})
        # C11 += M1, C22 += M1
        w1 = make_C({(0,0): 1, (1,1): 1})
        add_term(u1, v1, w1)
        
        # M2
        u2 = make_A({(1,0): 1, (1,1): 1})
        v2 = make_B({(0,0): 1})
        # C21 += M2, C22 -= M2
        w2 = make_C({(1,0): 1, (1,1): -1})
        add_term(u2, v2, w2)
        
        # M3
        u3 = make_A({(0,0): 1})
        v3 = make_B({(0,1): 1, (1,1): -1})
        # C12 += M3, C22 += M3
        w3 = make_C({(0,1): 1, (1,1): 1})
        add_term(u3, v3, w3)
        
        # M4
        u4 = make_A({(1,1): 1})
        v4 = make_B({(1,0): 1, (0,0): -1})
        # C11 += M4, C21 += M4
        w4 = make_C({(0,0): 1, (1,0): 1})
        add_term(u4, v4, w4)
        
        # M5
        u5 = make_A({(0,0): 1, (0,1): 1})
        v5 = make_B({(1,1): 1})
        # C11 -= M5, C12 += M5
        w5 = make_C({(0,0): -1, (0,1): 1})
        add_term(u5, v5, w5)
        
        # M6
        u6 = make_A({(1,0): 1, (0,0): -1})
        v6 = make_B({(0,0): 1, (0,1): 1})
        # C22 += M6
        w6 = make_C({(1,1): 1})
        add_term(u6, v6, w6)
        
        # M7
        u7 = make_A({(0,1): 1, (1,1): -1})
        v7 = make_B({(1,0): 1, (1,1): 1})
        # C11 += M7
        w7 = make_C({(0,0): 1})
        add_term(u7, v7, w7)
        
    # Construct M_2,4,8 part (i in 0,1)
    # j in 0,1 and k in 0,1
    add_strassen(0, 0, 0)
    # j in 0,1 and k in 2,3
    add_strassen(0, 0, 2)
    # j in 2,3 and k in 0,1
    add_strassen(0, 2, 0)
    # j in 2,3 and k in 2,3
    add_strassen(0, 2, 2)
    
    # Construct M_1,4,8 part (i in 2)
    # i=2, j in 0..3, k in 0..7
    for j in range(m):
        for k in range(p):
            u_vec = [0.0] * (n*m)
            v_vec = [0.0] * (m*p)
            w_vec = [0.0] * (p*n)
            
            u_vec[idx_A(2, j)] = 1.0
            v_vec[idx_B(j, k)] = 1.0
            w_vec[idx_C(2, k)] = 1.0
            
            add_term(u_vec, v_vec, w_vec)
            
    # Convert lists to matrices
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U, "V": V, "W": W}
