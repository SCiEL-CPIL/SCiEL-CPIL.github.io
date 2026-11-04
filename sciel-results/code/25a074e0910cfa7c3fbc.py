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

def get_strassen_2x2x3():
    """
    Rank-11 decomposition for 2x2x3 matrix multiplication.
    Composed of Strassen 2x2x2 (rank 7) + schoolbook 2x2x1 (rank 4).
    U: (4, 11), V: (6, 11), W: (6, 11)
    """
    U_2, V_2, W_2 = get_strassen_2x2x2()
    
    U_list = []
    V_list = []
    W_list = []
    
    # Add Strassen 2x2x2 part
    for r in range(7):
        u = U_2[:, r]
        v = V_2[:, r] # length 4 (2x2)
        w = W_2[:, r] # length 4 (2x2)
        
        # Extend v to length 6 (2x3)
        v_full = np.zeros(6)
        v_full[:4] = v
        
        # Extend w to length 6 (3x2)
        # W indices for 2x2: k*2+i for k,i in {0,1} -> 0,1,2,3
        # W indices for 2x3: k*2+i for k in {0,1,2}, i in {0,1} -> 0,1,2,3,4,5
        # The Strassen W only affects k=0,1. So indices 0..3 are set, 4,5 are 0.
        w_full = np.zeros(6)
        w_full[:4] = w
        
        U_list.append(u)
        V_list.append(v_full)
        W_list.append(w_full)
        
    # Add schoolbook 2x2x1 part (for k=2)
    for i in range(2):
        for j in range(2):
            u = np.zeros(4)
            v = np.zeros(6)
            w = np.zeros(6)
            
            a = i * 2 + j
            b = j * 3 + 2 # k=2
            c = 2 * 2 + i # k=2
            
            u[a] = 1
            v[b] = 1
            w[c] = 1
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
            
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    U_list = []
    V_list = []
    W_list = []
    
    # 1. Construct 4x4x4 decomposition using Kronecker product of Strassen 2x2x2
    # This handles the multiplication of 4x4 A by 4x4 B_part (first 4 cols of B)
    # Rank is 7 * 7 = 49
    U_2, V_2, W_2 = get_strassen_2x2x2()
    
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
            
            U_list.append(u)
            V_list.append(v_full)
            W_list.append(w_full)
            
    # 2. Construct 4x4x3 decomposition for the remaining 3 columns (k=4,5,6)
    # Instead of schoolbook (rank 48), use 4 separate 2x2x3 decompositions (rank 11 each).
    # Total rank 4 * 11 = 44.
    U_23, V_23, W_23 = get_strassen_2x2x3()
    
    for i_block in range(2): # i in {0,1} for first row block, {2,3} for second
        for j_block in range(2): # j in {0,1} for first col block, {2,3} for second
            
            base_i = i_block * 2
            base_j = j_block * 2
            
            for r in range(11):
                u_23 = U_23[:, r] # length 4 (2x2)
                v_23 = V_23[:, r] # length 6 (2x3)
                w_23 = W_23[:, r] # length 6 (3x2)
                
                u_full = np.zeros(dim_u)
                v_full = np.zeros(dim_v)
                w_full = np.zeros(dim_w)
                
                # Map u (2x2) to 4x4
                # u index: i'*2 + j'. Target: (base_i + i')*4 + (base_j + j')
                for i_prime in range(2):
                    for j_prime in range(2):
                        src = i_prime * 2 + j_prime
                        dst = (base_i + i_prime) * m + (base_j + j_prime)
                        u_full[dst] = u_23[src]
                        
                # Map v (2x3) to 4x7
                # v index: j'*3 + k'. Target: (base_j + j')*7 + (4 + k')
                # Note: k' in {0,1,2} corresponds to global k in {4,5,6}
                for j_prime in range(2):
                    for k_prime in range(3):
                        src = j_prime * 3 + k_prime
                        dst = (base_j + j_prime) * p + (4 + k_prime)
                        v_full[dst] = v_23[src]
                        
                # Map w (3x2) to 7x4
                # w index: k'*2 + i'. Target: (4 + k')*4 + (base_i + i')
                # Note: k' in {0,1,2} corresponds to global k in {4,5,6}
                for k_prime in range(3):
                    for i_prime in range(2):
                        src = k_prime * 2 + i_prime
                        dst = (4 + k_prime) * n + (base_i + i_prime)
                        w_full[dst] = w_23[src]
                        
                U_list.append(u_full)
                V_list.append(v_full)
                W_list.append(w_full)
                
    # Combine
    U_all = np.array(U_list).T
    V_all = np.array(V_list).T
    W_all = np.array(W_list).T
    
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
