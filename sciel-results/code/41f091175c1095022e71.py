import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # We will build the matrices as lists of lists or numpy arrays
    # Since rank is 53, we can pre-allocate or append.
    # Let's use lists of lists for simplicity, then convert if needed? 
    # The return type expects dictionaries with keys 'U', 'V', 'W'.
    # The values should be iterable of numbers. Lists are fine.
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a rank-1 component
    def add_component(u_vec, v_vec, w_vec):
        # u_vec: list of 4 numbers
        # v_vec: list of 30 numbers
        # w_vec: list of 30 numbers
        U_rows.append(u_vec)
        V_rows.append(v_vec)
        W_rows.append(w_vec)

    # Strassen block for columns k, k+1
    def add_strassen_block(k):
        # k is the start column index
        # U indices: 0,1,2,3 correspond to a00, a01, a10, a11
        # V indices: j*p + col
        # W indices: col*n + i
        
        # M1
        u = [1, 0, 0, 1]
        v = [0]*30
        v[0*p + k] = 1
        v[1*p + (k+1)] = 1
        w = [0]*30
        w[k*n + 0] = 1
        w[(k+1)*n + 1] = 1
        add_component(u, v, w)
        
        # M2
        u = [0, 0, 1, 1]
        v = [0]*30
        v[0*p + k] = 1
        w = [0]*30
        w[k*n + 1] = 1
        w[(k+1)*n + 1] = -1
        add_component(u, v, w)
        
        # M3
        u = [1, 0, 0, 0]
        v = [0]*30
        v[0*p + (k+1)] = 1
        v[1*p + (k+1)] = -1
        w = [0]*30
        w[(k+1)*n + 0] = 1
        w[(k+1)*n + 1] = 1
        add_component(u, v, w)
        
        # M4
        u = [0, 0, 0, 1]
        v = [0]*30
        v[0*p + k] = -1
        v[1*p + k] = 1
        w = [0]*30
        w[k*n + 0] = 1
        w[k*n + 1] = 1
        add_component(u, v, w)
        
        # M5
        u = [1, 1, 0, 0]
        v = [0]*30
        v[1*p + (k+1)] = 1
        w = [0]*30
        w[(k+1)*n + 0] = 1
        w[k*n + 0] = -1
        add_component(u, v, w)
        
        # M6
        u = [-1, 0, 1, 0]
        v = [0]*30
        v[0*p + k] = 1
        v[0*p + (k+1)] = 1
        w = [0]*30
        w[(k+1)*n + 1] = 1
        add_component(u, v, w)
        
        # M7
        u = [0, 1, 0, -1]
        v = [0]*30
        v[1*p + k] = 1
        v[1*p + (k+1)] = 1
        w = [0]*30
        w[k*n + 0] = 1
        add_component(u, v, w)

    # Schoolbook block for column k
    def add_schoolbook_block(k):
        # c0k = a00 b0k + a01 b1k
        # c1k = a10 b0k + a11 b1k
        
        # 1. a00 b0k -> c0k
        u = [1, 0, 0, 0]
        v = [0]*30
        v[0*p + k] = 1
        w = [0]*30
        w[k*n + 0] = 1
        add_component(u, v, w)
        
        # 2. a01 b1k -> c0k
        u = [0, 1, 0, 0]
        v = [0]*30
        v[1*p + k] = 1
        w = [0]*30
        w[k*n + 0] = 1
        add_component(u, v, w)
        
        # 3. a10 b0k -> c1k
        u = [0, 0, 1, 0]
        v = [0]*30
        v[0*p + k] = 1
        w = [0]*30
        w[k*n + 1] = 1
        add_component(u, v, w)
        
        # 4. a11 b1k -> c1k
        u = [0, 0, 0, 1]
        v = [0]*30
        v[1*p + k] = 1
        w = [0]*30
        w[k*n + 1] = 1
        add_component(u, v, w)

    # Loop through columns
    for k in range(0, p, 2):
        if k + 1 < p:
            add_strassen_block(k)
        else:
            add_schoolbook_block(k)

    # Transpose to get shapes (n*m, rank) etc?
    # The interface expects U to be (n*m, rank).
    # My U_rows is a list of rank vectors of size n*m.
    # So U_rows[r] is the r-th column of U.
    # I need to return U such that U[i, r] is the value.
    # So I should transpose U_rows.
    
    import numpy as np
    U = np.array(U_rows).T
    V = np.array(V_rows).T
    W = np.array(W_rows).T
    
    return {"U": U, "V": V, "W": W}
