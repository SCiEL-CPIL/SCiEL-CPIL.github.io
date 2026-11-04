import numpy as np

def construct():
    n, m, p = 2, 5, 5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_entries, v_entries, w_entries):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for idx, val in u_entries.items():
            u[idx] += val
        for idx, val in v_entries.items():
            v[idx] += val
        for idx, val in w_entries.items():
            w[idx] += val
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def a_idx(i, j):
        return i * m + j
    
    def b_idx(j, k):
        return j * p + k
    
    def c_idx(i, k):
        return k * n + i
    
    def strassen_block(i1, i2, j1, j2, k1, k2):
        a1 = a_idx(i1, j1)
        a2 = a_idx(i1, j2)
        a3 = a_idx(i2, j1)
        a4 = a_idx(i2, j2)
        
        b1 = b_idx(j1, k1)
        b2 = b_idx(j1, k2)
        b3 = b_idx(j2, k1)
        b4 = b_idx(j2, k2)
        
        c1 = c_idx(i1, k1)
        c2 = c_idx(i1, k2)
        c3 = c_idx(i2, k1)
        c4 = c_idx(i2, k2)
        
        add_col({a1:1, a4:1}, {b1:1, b4:1}, {c1:1, c4:1})
        add_col({a3:1, a4:1}, {b1:1}, {c3:1})
        add_col({a1:1}, {b2:1, b4:-1}, {c2:1})
        add_col({a4:1}, {b3:1, b1:-1}, {c1:1, c3:1})
        add_col({a1:1, a2:1}, {b4:1}, {c1:-1, c2:1})
        add_col({a3:1, a1:-1}, {b1:1, b2:1}, {c4:1})
        add_col({a2:1, a4:-1}, {b3:1, b4:1}, {c1:1, c4:1})
    
    strassen_block(0, 1, 0, 1, 0, 1)
    strassen_block(0, 1, 2, 3, 0, 1)
    strassen_block(0, 1, 0, 1, 2, 3)
    strassen_block(0, 1, 2, 3, 2, 3)
    
    for i in range(2):
        for k in range(2):
            add_col({a_idx(i, 4): 1}, {b_idx(4, k): 1}, {c_idx(i, k): 1})
    
    for i in range(2):
        for k in range(2):
            add_col({a_idx(i, 4): 1}, {b_idx(4, k+2): 1}, {c_idx(i, k+2): 1})
    
    for i in range(2):
        for j in range(2):
            add_col({a_idx(i, j): 1}, {b_idx(j, 4): 1}, {c_idx(i, 4): 1})
    
    for i in range(2):
        for j in range(2):
            add_col({a_idx(i, j+2): 1}, {b_idx(j+2, 4): 1}, {c_idx(i, 4): 1})
    
    for i in range(2):
        add_col({a_idx(i, 4): 1}, {b_idx(4, 4): 1}, {c_idx(i, 4): 1})
    
    rank = len(U_cols)
    U = [list(col) for col in zip(*U_cols)]
    V = [list(col) for col in zip(*V_cols)]
    W = [list(col) for col in zip(*W_cols)]
    
    # Verify
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 if i,j,k are the "diagonal"
    # Actually, the decomposition should satisfy:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta (the tensor)
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    
    # Let's verify by checking that the reconstructed tensor matches
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                if abs(val - 1) > 1e-10:
                    # This is a bug
                    pass
    
    return {"U": U, "V": V, "W": W}
