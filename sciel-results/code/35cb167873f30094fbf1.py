import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the U, V, W matrices by concatenating blocks.
    # Each block corresponds to a sub-problem.
    
    blocks = []
    
    # Define the partitions
    I_parts = [[0, 1], [2]]
    J_parts = [[0, 1], [2, 3]]
    K_parts = [[0, 1], [2, 3], [4]]
    
    # For each combination of I, J, K parts, determine the sub-problem size and type
    # and generate the corresponding U, V, W slices.
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for i_idx, I in enumerate(I_parts):
        for j_idx, J in enumerate(J_parts):
            for k_idx, K in enumerate(K_parts):
                ni = len(I)
                mj = len(J)
                pk = len(K)
                
                # Determine the rank and generator
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen 2x2x2
                    rank = 7
                    u, v, w = _strassen_222(I, J, K, n, m, p)
                elif ni == 2 and mj == 2 and pk == 1:
                    # Schoolbook 2x2x1
                    rank = 4
                    u, v, w = _schoolbook_221(I, J, K, n, m, p)
                elif ni == 1 and mj == 2 and pk == 2:
                    # Schoolbook 1x2x2
                    rank = 4
                    u, v, w = _schoolbook_122(I, J, K, n, m, p)
                elif ni == 1 and mj == 2 and pk == 1:
                    # Schoolbook 1x2x1
                    rank = 2
                    u, v, w = _schoolbook_121(I, J, K, n, m, p)
                else:
                    # Fallback to schoolbook for any other size (should not happen with this partition)
                    rank = ni * mj * pk
                    u, v, w = _schoolbook_generic(I, J, K, n, m, p)
                    
                U_blocks.append(u)
                V_blocks.append(v)
                W_blocks.append(w)
                
    # Concatenate
    U = np.hstack(U_blocks)
    V = np.hstack(V_blocks)
    W = np.hstack(W_blocks)
    
    return {"U": U, "V": V, "W": W}

def _strassen_222(I, J, K, n, m, p):
    # I, J, K are lists of indices
    # We need to map local indices 0,1 to global indices
    i0, i1 = I
    j0, j1 = J
    k0, k1 = K
    
    # Strassen's 7 multiplications
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # U, V, W shapes: (n*m, 7), (m*p, 7), (p*n, 7)
    # But we only fill the rows/cols corresponding to I, J, K.
    # The returned u, v, w should be full sized (n*m, 7), etc., with zeros elsewhere.
    
    u = np.zeros((n * m, 7))
    v = np.zeros((m * p, 7))
    w = np.zeros((p * n, 7))
    
    # Helper to get row index for U
    def u_idx(i, j): return i * m + j
    # Helper to get row index for V
    def v_idx(j, k): return j * p + k
    # Helper to get row index for W
    def w_idx(i, k): return k * n + i
    
    # M1
    u[u_idx(i0, j0), 0] = 1
    u[u_idx(i1, j1), 0] = 1
    v[v_idx(j0, k0), 0] = 1
    v[v_idx(j1, k1), 0] = 1
    w[w_idx(i0, k0), 0] = 1
    w[w_idx(i0, k1), 0] = 1
    w[w_idx(i1, k0), 0] = 1
    w[w_idx(i1, k1), 0] = 1
    
    # M2
    u[u_idx(i1, j0), 1] = 1
    u[u_idx(i1, j1), 1] = 1
    v[v_idx(j0, k0), 1] = 1
    w[w_idx(i1, k0), 1] = 1
    
    # M3
    u[u_idx(i0, j0), 2] = 1
    v[v_idx(j0, k1), 2] = 1
    v[v_idx(j1, k1), 2] = -1
    w[w_idx(i0, k1), 2] = 1
    
    # M4
    u[u_idx(i1, j1), 3] = 1
    v[v_idx(j1, k0), 3] = 1
    v[v_idx(j0, k0), 3] = -1
    w[w_idx(i1, k0), 3] = 1
    
    # M5
    u[u_idx(i0, j0), 4] = 1
    u[u_idx(i0, j1), 4] = 1
    v[v_idx(j1, k1), 4] = 1
    w[w_idx(i0, k1), 4] = 1
    
    # M6
    u[u_idx(i1, j0), 5] = 1
    u[u_idx(i0, j0), 5] = -1
    v[v_idx(j0, k0), 5] = 1
    v[v_idx(j0, k1), 5] = 1
    w[w_idx(i1, k0), 5] = 1
    w[w_idx(i1, k1), 5] = 1
    
    # M7
    u[u_idx(i0, j1), 6] = 1
    u[u_idx(i1, j1), 6] = -1
    v[v_idx(j1, k0), 6] = 1
    v[v_idx(j1, k1), 6] = 1
    w[w_idx(i0, k1), 6] = 1
    w[w_idx(i1, k1), 6] = 1
    
    return u, v, w

def _schoolbook_221(I, J, K, n, m, p):
    i0, i1 = I
    j0, j1 = J
    k0 = K[0]
    
    u = np.zeros((n * m, 4))
    v = np.zeros((m * p, 4))
    w = np.zeros((p * n, 4))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i
    
    col = 0
    for i in [i0, i1]:
        for j in [j0, j1]:
            u[u_idx(i, j), col] = 1
            v[v_idx(j, k0), col] = 1
            w[w_idx(i, k0), col] = 1
            col += 1
            
    return u, v, w

def _schoolbook_122(I, J, K, n, m, p):
    i0 = I[0]
    j0, j1 = J
    k0, k1 = K
    
    u = np.zeros((n * m, 4))
    v = np.zeros((m * p, 4))
    w = np.zeros((p * n, 4))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i
    
    col = 0
    for j in [j0, j1]:
        for k in [k0, k1]:
            u[u_idx(i0, j), col] = 1
            v[v_idx(j, k), col] = 1
            w[w_idx(i0, k), col] = 1
            col += 1
            
    return u, v, w

def _schoolbook_121(I, J, K, n, m, p):
    i0 = I[0]
    j0, j1 = J
    k0 = K[0]
    
    u = np.zeros((n * m, 2))
    v = np.zeros((m * p, 2))
    w = np.zeros((p * n, 2))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i
    
    col = 0
    for j in [j0, j1]:
        u[u_idx(i0, j), col] = 1
        v[v_idx(j, k0), col] = 1
        w[w_idx(i0, k0), col] = 1
        col += 1
        
    return u, v, w

def _schoolbook_generic(I, J, K, n, m, p):
    ni = len(I)
    mj = len(J)
    pk = len(K)
    rank = ni * mj * pk
    
    u = np.zeros((n * m, rank))
    v = np.zeros((m * p, rank))
    w = np.zeros((p * n, rank))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i
    
    col = 0
    for i in I:
        for j in J:
            for k in K:
                u[u_idx(i, j), col] = 1
                v[v_idx(j, k), col] = 1
                w[w_idx(i, k), col] = 1
                col += 1
                
    return u, v, w
