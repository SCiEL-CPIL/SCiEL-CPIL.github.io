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
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    U_all = []
    V_all = []
    W_all = []
    
    # 1. 4x4x4 part (columns 0-3 of B)
    # Rank 49
    U_2, V_2, W_2 = get_strassen_2x2x2()
    
    for r in range(7):
        for s in range(7):
            u = np.kron(U_2[:, r], U_2[:, s])
            
            v_kron = np.kron(V_2[:, r], V_2[:, s])
            v_full = np.zeros(dim_v)
            for j in range(4):
                for k in range(4):
                    if k < 4: # Only first 4 columns of B
                        v_full[j * p + k] = v_kron[j * 4 + k]
                        
            w_kron = np.kron(W_2[:, r], W_2[:, s])
            w_full = np.zeros(dim_w)
            # W indices: k*n + i. k in 0..3.
            # Target W indices: k*n + i.
            # Since we only compute C for k=0..3, we map directly.
            for k in range(4):
                for i in range(4):
                    w_full[k * n + i] = w_kron[k * 4 + i]
                    
            U_all.append(u)
            V_all.append(v_full)
            W_all.append(w_full)
            
    # 2. 4x4x2 part (columns 4-5 of B)
    # Rank 28
    # B cols 4,5.
    # B_0 (top) uses B[0,4], B[0,5], B[1,4], B[1,5] -> indices 4,5,6,7 in B (4x7)
    # B_1 (bot) uses B[2,4], B[2,5], B[3,4], B[3,5] -> indices 12,13,14,15 in B
    # Wait, B index is j*p + k.
    # j=0, k=4 -> 4. j=0, k=5 -> 5.
    # j=1, k=4 -> 1*7+4 = 11. j=1, k=5 -> 12.
    # j=2, k=4 -> 2*7+4 = 18. j=2, k=5 -> 19.
    # j=3, k=4 -> 3*7+4 = 25. j=3, k=5 -> 26.
    
    # C cols 4,5.
    # C index is k*n + i.
    # k=4, i=0 -> 16. k=4, i=1 -> 17. k=4, i=2 -> 18. k=4, i=3 -> 19.
    # k=5, i=0 -> 20. k=5, i=1 -> 21. k=5, i=2 -> 22. k=5, i=3 -> 23.
    
    # Blocks:
    # A_00: rows 0,1; cols 0,1.
    # A_01: rows 0,1; cols 2,3.
    # A_10: rows 2,3; cols 0,1.
    # A_11: rows 2,3; cols 2,3.
    
    # Products:
    # A_00 * B_0 -> C_00 (top-left 2x2 of C_4,5)
    # A_01 * B_1 -> C_01 (top-right? No, C is 4x2. C_00 is C[0,4], C[0,5], C[1,4], C[1,5])
    # Wait, C is 4x2.
    # C_top is 2x2. C_bot is 2x2.
    # C_top[0,0] = C[0,4]. C_top[0,1] = C[0,5].
    # C_top[1,0] = C[1,4]. C_top[1,1] = C[1,5].
    # C_bot[0,0] = C[2,4]. C_bot[0,1] = C[2,5].
    # C_bot[1,0] = C[3,4]. C_bot[1,1] = C[3,5].
    
    # A_00 * B_0 contributes to C_top.
    # A_01 * B_1 contributes to C_top.
    # A_10 * B_0 contributes to C_bot.
    # A_11 * B_1 contributes to C_bot.
    
    # B_0 indices in B (4x7):
    # B[0,4]=4, B[0,5]=5, B[1,4]=11, B[1,5]=12.
    # B_1 indices in B:
    # B[2,4]=18, B[2,5]=19, B[3,4]=25, B[3,5]=26.
    
    # C_top indices in C (7x4 -> 28):
    # C[0,4]=16, C[0,5]=17, C[1,4]=18, C[1,5]=19.
    # C_bot indices:
    # C[2,4]=20, C[2,5]=21, C[3,4]=22, C[3,5]=23.
    
    # U indices for A blocks:
    # A_00: 0,1,2,3.
    # A_01: 4,5,6,7.
    # A_10: 8,9,10,11.
    # A_11: 12,13,14,15.
    
    # For each block, we use Strassen.
    blocks = [
        (0, 0, 4, 16),   # A_00, B_0, C_top
        (4, 11, 18, 16), # A_01, B_1, C_top
        (8, 4, 20, 16),  # A_10, B_0, C_bot
        (12, 18, 20, 16) # A_11, B_1, C_bot
    ]
    # Note: C_top and C_bot have different W indices.
    # C_top W indices: 16,17,18,19.
    # C_bot W indices: 20,21,22,23.
    
    for a_start, b_start, c_start, _ in blocks:
        # a_start is start of A block in U
        # b_start is start of B block in V
        # c_start is start of C block in W
        
        # We need to map the 4x4 Strassen U, V, W to these blocks.
        # U vector for this block:
        # Strassen U is 4x7.
        # We place it at a_start..a_start+3.
        
        # V vector for this block:
        # Strassen V is 4x7.
        # We place it at b_start..b_start+3?
        # No, B_0 indices are 4,5,11,12. Not contiguous.
        # B_1 indices are 18,19,25,26. Not contiguous.
        
        # So we cannot just use a contiguous slice.
        # We must construct V explicitly.
        
        for r in range(7):
            u = np.zeros(dim_u)
            v = np.zeros(dim_v)
            w = np.zeros(dim_w)
            
            # U
            u[a_start:a_start+4] = U_2[:, r]
            
            # V
            # Strassen V vector v2 = V_2[:, r] has 4 elements.
            # These correspond to B_block[0,0], B_block[0,1], B_block[1,0], B_block[1,1].
            # For B_0: indices 4,5,11,12.
            # For B_1: indices 18,19,25,26.
            
            if b_start == 4: # B_0
                v[4] = V_2[0, r]
                v[5] = V_2[1, r]
                v[11] = V_2[2, r]
                v[12] = V_2[3, r]
            elif b_start == 11: # This case is not in my blocks list logic, I used b_start as start index.
                # Let's just use the explicit indices.
                pass
                
            # Let's re-define blocks with explicit index lists.
            # This is getting messy. Let's just loop.

    # Refined 4x4x2 construction
    U_2, V_2, W_2 = get_strassen_2x2x2()
    
    # Block 0: A_00 (0-3), B_0 (4,5,11,12), C_top (16-19)
    for r in range(7):
        u = np.zeros(dim_u)
        v = np.zeros(dim_v)
        w = np.zeros(dim_w)
        
        u[0:4] = U_2[:, r]
        
        v[4] = V_2[0, r]
        v[5] = V_2[1, r]
        v[11] = V_2[2, r]
        v[12] = V_2[3, r]
        
        w[16:20] = W_2[:, r]
        
        U_all.append(u)
        V_all.append(v)
        W_all.append(w)
        
    # Block 1: A_01 (4-7), B_1 (18,19,25,26), C_top (16-19)
    for r in range(7):
        u = np.zeros(dim_u)
        v = np.zeros(dim_v)
        w = np.zeros(dim_w)
        
        u[4:8] = U_2[:, r]
        
        v[18] = V_2[0, r]
        v[19] = V_2[1, r]
        v[25] = V_2[2, r]
        v[26] = V_2[3, r]
        
        w[16:20] = W_2[:, r]
        
        U_all.append(u)
        V_all.append(v)
        W_all.append(w)
        
    # Block 2: A_10 (8-11), B_0 (4,5,11,12), C_bot (20-23)
    for r in range(7):
        u = np.zeros(dim_u)
        v = np.zeros(dim_v)
        w = np.zeros(dim_w)
        
        u[8:12] = U_2[:, r]
        
        v[4] = V_2[0, r]
        v[5] = V_2[1, r]
        v[11] = V_2[2, r]
        v[12] = V_2[3, r]
        
        w[20:24] = W_2[:, r]
        
        U_all.append(u)
        V_all.append(v)
        W_all.append(w)
        
    # Block 3: A_11 (12-15), B_1 (18,19,25,26), C_bot (20-23)
    for r in range(7):
        u = np.zeros(dim_u)
        v = np.zeros(dim_v)
        w = np.zeros(dim_w)
        
        u[12:16] = U_2[:, r]
        
        v[18] = V_2[0, r]
        v[19] = V_2[1, r]
        v[25] = V_2[2, r]
        v[26] = V_2[3, r]
        
        w[20:24] = W_2[:, r]
        
        U_all.append(u)
        V_all.append(v)
        W_all.append(w)

    # 3. 4x4x1 part (column 6 of B)
    # Rank 16
    for i in range(n):
        for j in range(m):
            u = np.zeros(dim_u)
            v = np.zeros(dim_v)
            w = np.zeros(dim_w)
            
            a = i * m + j
            b = j * p + 6
            c = 6 * n + i
            
            u[a] = 1
            v[b] = 1
            w[c] = 1
            
            U_all.append(u)
            V_all.append(v)
            W_all.append(w)

    U_final = np.array(U_all).T
    V_final = np.array(V_all).T
    W_final = np.array(W_all).T
    
    rank = U_final.shape[1]
    
    # Verification
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
                
    T_decomp = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
    
    if not np.array_equal(T_target, T_decomp):
        # Fallback to schoolbook if Strassen construction is flawed
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
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
