import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def get_strassen_2x2x2():
    """
    Returns U, V, W matrices (4x7) for the rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    Indices: a=i*2+j, b=j*2+k, c=k*2+i for i,j,k in {0,1}.
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    
    # u vectors (coefficients for A):
    u1 = [1, 0, 0, 1] # A00 + A11
    u2 = [0, 0, 1, 1] # A10 + A11
    u3 = [1, 0, 0, 0] # A00
    u4 = [0, 0, 0, 1] # A11
    u5 = [1, 1, 0, 0] # A00 + A01
    u6 = [-1, 0, 1, 0] # A10 - A00
    u7 = [0, 1, 0, -1] # A01 - A11
    
    # v vectors (coefficients for B):
    v1 = [1, 0, 0, 1] # B00 + B11
    v2 = [1, 0, 0, 0] # B00
    v3 = [0, 1, 0, -1] # B01 - B11
    v4 = [0, 0, 1, -1] # B10 - B00
    v5 = [0, 0, 0, 1] # B11
    v6 = [1, 1, 0, 0] # B00 + B01
    v7 = [0, 0, 1, 1] # B10 + B11
    
    # w vectors (coefficients for C):
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    w1 = [1, 0, 0, 1] # M1: +C00, +C11
    w2 = [0, 0, 1, 0] # M2: +C10
    w3 = [0, 1, 0, 0] # M3: +C01
    w4 = [-1, 0, 1, 0] # M4: -C00, +C10
    w5 = [0, 1, 0, 1] # M5: +C01, +C11
    w6 = [1, 0, 0, -1] # M6: +C00, -C11
    w7 = [-1, 0, 0, 1] # M7: -C00, +C11
    
    U = np.array([u1, u2, u3, u4, u5, u6, u7]).T
    V = np.array([v1, v2, v3, v4, v5, v6, v7]).T
    W = np.array([w1, w2, w3, w4, w5, w6, w7]).T
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Dimensions
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    # 1. Get Strassen decomposition for 2x2x2
    U_2, V_2, W_2 = get_strassen_2x2x2()
    
    # 2. Construct 4x4x4 decomposition using Kronecker product
    # This handles the multiplication of 4x4 A by 4x4 B_part (first 4 cols of B)
    # Rank is 7 * 7 = 49
    U_4 = []
    V_4 = []
    W_4 = []
    
    for r in range(7):
        for s in range(7):
            # u is length 16 (4x4)
            u = np.kron(U_2[:, r], U_2[:, s])
            
            # v is length 16 (4x4), needs to be mapped to length 28 (4x7)
            # Index in kron: j*4 + k. Target index: j*7 + k.
            v_kron = np.kron(V_2[:, r], V_2[:, s])
            v_full = np.zeros(dim_v)
            for j in range(4):
                for k in range(4):
                    v_full[j * p + k] = v_kron[j * 4 + k]
            
            # w is length 16 (4x4), needs to be mapped to length 28 (7x4)
            # Index in kron: k*4 + i. Target index: k*4 + i.
            # Since k in 0..3, indices 0..15 are direct. 16..27 are 0.
            w_kron = np.kron(W_2[:, r], W_2[:, s])
            w_full = np.zeros(dim_w)
            w_full[:16] = w_kron
            
            U_4.append(u)
            V_4.append(v_full)
            W_4.append(w_full)
            
    # 3. Construct schoolbook decomposition for the remaining 3 columns (k=4,5,6)
    U_3 = []
    V_3 = []
    W_3 = []
    
    for i in range(n):
        for j in range(m):
            for k in range(4, p): # k = 4, 5, 6
                u = np.zeros(dim_u)
                v = np.zeros(dim_v)
                w = np.zeros(dim_w)
                
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                u[a] = 1
                v[b] = 1
                w[c] = 1
                
                U_3.append(u)
                V_3.append(v)
                W_3.append(w)
                
    # Combine
    # Ensure all lists are non-empty and contain arrays of same shape
    # U_4 has 49 items, U_3 has 4*4*3 = 48 items. Total 97.
    # Each item is a numpy array of length 16, 28, 28 respectively.
    
    U_all = np.array(U_4 + U_3).T
    V_all = np.array(V_4 + V_3).T
    W_all = np.array(W_4 + W_3).T
    
    rank = U_all.shape[1]
    
    # Verification
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
                
    T_decomp = np.einsum('ar,br,cr->abc', U_all, V_all, W_all)
    
    if not np.array_equal(T_target, T_decomp):
        # Fallback to schoolbook if Strassen construction is flawed
        # Schoolbook rank: 4*4*7 = 112
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {
        "U": U_all.tolist(),
        "V": V_all.tolist(),
        "W": W_all.tolist()
    }
