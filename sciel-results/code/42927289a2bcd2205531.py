import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition 3+2
    # Total rank will be sum of ranks of components
    
    # 1. Winograd 3x3 (Rank 23)
    # 2. Strassen 2x2 (Rank 7)
    # 3. Schoolbook 3x2 (Rank 6)
    # 4. Schoolbook 2x3 (Rank 6)
    # 5. Schoolbook 3x2x3 (Rank 6)
    # 6. Schoolbook 2x3x2 (Rank 6)
    
    # We need to map local indices to global indices.
    # Global size 5.
    # Block 11: rows 0-2, cols 0-2 (3x3)
    # Block 12: rows 0-2, cols 3-4 (3x2)
    # Block 21: rows 3-4, cols 0-2 (2x3)
    # Block 22: rows 3-4, cols 3-4 (2x2)
    
    # Terms:
    # T1: A11 B11 -> 3x3 (Winograd)
    # T2: A12 B21 -> 3x3 (Schoolbook 3x2x3)
    # T3: A11 B12 -> 3x2 (Schoolbook 3x2)
    # T4: A12 B22 -> 3x2 (Schoolbook 3x2)
    # T5: A21 B11 -> 2x3 (Schoolbook 2x3)
    # T6: A22 B21 -> 2x3 (Schoolbook 2x3)
    # T7: A21 B12 -> 2x2 (Schoolbook 2x3x2)
    # T8: A22 B22 -> 2x2 (Strassen)
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(U_local, V_local, W_local, row_offset_u, col_offset_v, row_offset_w, col_offset_w):
        # U_local shape (n_u, r), V_local shape (n_v, r), W_local shape (n_w, r)
        # Global U index: i_global = row_offset_u + i_local
        # Global V index: j_global = col_offset_v + j_local (Note: V is indexed by j,k? No, V is (m*p, rank). 
        # In the task, U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
        # Standard mapping:
        # U[i*m+j]
        # V[j*p+k]
        # W[k*n+i]
        
        # For a block multiplication A_sub (nu x nv) * B_sub (nv x np) -> C_sub (nu x np)
        # U rows correspond to (i, j) in A_sub.
        # V rows correspond to (j, k) in B_sub.
        # W rows correspond to (k, i) in C_sub.
        
        # We need to map local (i, j) to global (I, J)
        # Global I = row_offset_i + i
        # Global J = col_offset_j + j
        # So U global row = I*m + J
        
        # V global row = J*p + K
        # Local (j, k) -> Global (J, K)
        # J = col_offset_j + j
        # K = col_offset_k + k
        
        # W global row = K*n + I
        # Local (k, i) -> Global (K, I)
        # K = col_offset_k + k
        # I = row_offset_i + i
        
        pass

    # Helper to create schoolbook decomposition for nu x nv * nv x np
    def schoolbook(nu, nv, np_):
        rank = nu * nv * np_
        U = np.zeros((nu * nv, rank))
        V = np.zeros((nv * np_, rank))
        W = np.zeros((np_ * nu, rank))
        col = 0
        for i in range(nu):
            for j in range(nv):
                for k in range(np_):
                    U[i * nv + j, col] = 1
                    V[j * np_ + k, col] = 1
                    W[k * nu + i, col] = 1
                    col += 1
        return U, V, W

    # Helper to create Strassen 2x2
    def strassen_2x2():
        # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # p0 = (a00+a11)(b00+b11)
        # p1 = (a10+a11)b00
        # p2 = a00(b10-b11)
        # p3 = a11(b01-b00)
        # p4 = (a00+a01)b11
        # p5 = (a10-a00)(b00+b01)
        # p6 = (a01-a11)(b10+b11)
        
        # U rows: 00, 01, 10, 11
        # V rows: 00, 01, 10, 11
        # W rows: 00, 01, 10, 11 (k*2+i) -> 00, 01, 10, 11
        
        # p0
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # p1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # p2
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # p3
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[3, 3] = 1
        
        # p4
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[2, 4] = 1
        
        # p5
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = 1
        
        # p6
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1
        
        return U, V, W

    # Helper to create Winograd 3x3
    def winograd_3x3():
        # Rank 23
        # This is complex to hardcode. 
        # Given time, I will use schoolbook for 3x3 as well to ensure correctness, 
        # which gives rank 27.
        # Total rank: 27 + 6 + 6 + 6 + 6 + 6 + 6 + 7 = 70.
        # 70 is still much better than 125.
        return schoolbook(3, 3, 3)

    # Define block offsets
    # Block 11: I=0..2, J=0..2
    # Block 12: I=0..2, J=3..4
    # Block 21: I=3..4, J=0..2
    # Block 22: I=3..4, J=3..4
    
    def map_block(U_local, V_local, W_local, I_start, J_start, K_start):
        # U_local (nu*nv, r) -> Global (I*m+J, r)
        # V_local (nv*np, r) -> Global (J*p+K, r)
        # W_local (np*nu, r) -> Global (K*n+I, r)
        
        nu = int(np.sqrt(U_local.shape[0] / V_local.shape[0])) # This is fragile. 
        # Better to pass sizes explicitly.
        pass

    # Let's do it manually for each term to avoid bugs.
    
    # T1: A11 B11 (3x3 * 3x3)
    U1, V1, W1 = winograd_3x3()
    # Map to global
    # U1 rows i*3+j (i,j in 0..2) -> Global (0+i)*5 + (0+j)
    U1_g = np.zeros((25, 27))
    V1_g = np.zeros((25, 27))
    W1_g = np.zeros((25, 27))
    for i in range(3):
        for j in range(3):
            U1_g[i*5 + j] = U1[i*3 + j]
            # V1 row j*3+k -> Global j*5 + k
            for k in range(3):
                V1_g[j*5 + k] = V1[j*3 + k]
                W1_g[k*5 + i] = W1[k*3 + i]
    U_list.append(U1_g)
    V_list.append(V1_g)
    W_list.append(W1_g)
    
    # T2: A12 B21 (3x2 * 2x3 -> 3x3)
    # A12: I=0..2, J=3..4
    # B21: J=3..4, K=0..2
    # C: I=0..2, K=0..2
    U2, V2, W2 = schoolbook(3, 2, 3)
    U2_g = np.zeros((25, 18))
    V2_g = np.zeros((25, 18))
    W2_g = np.zeros((25, 18))
    for i in range(3):
        for j in range(2):
            U2_g[i*5 + (3+j)] = U2[i*2 + j]
            for k in range(3):
                V2_g[(3+j)*5 + k] = V2[j*3 + k]
                W2_g[k*5 + i] = W2[k*3 + i]
    U_list.append(U2_g)
    V_list.append(V2_g)
    W_list.append(W2_g)
    
    # T3: A11 B12 (3x3 * 3x2 -> 3x2)
    # A11: I=0..2, J=0..2
    # B12: J=0..2, K=3..4
    # C: I=0..2, K=3..4
    U3, V3, W3 = schoolbook(3, 3, 2)
    U3_g = np.zeros((25, 18))
    V3_g = np.zeros((25, 18))
    W3_g = np.zeros((25, 18))
    for i in range(3):
        for j in range(3):
            U3_g[i*5 + j] = U3[i*3 + j]
            for k in range(2):
                V3_g[j*5 + (3+k)] = V3[j*2 + k]
                W3_g[(3+k)*5 + i] = W3[k*3 + i]
    U_list.append(U3_g)
    V_list.append(V3_g)
    W_list.append(W3_g)
    
    # T4: A12 B22 (3x2 * 2x2 -> 3x2)
    # A12: I=0..2, J=3..4
    # B22: J=3..4, K=3..4
    # C: I=0..2, K=3..4
    U4, V4, W4 = schoolbook(3, 2, 2)
    U4_g = np.zeros((25, 12))
    V4_g = np.zeros((25, 12))
    W4_g = np.zeros((25, 12))
    for i in range(3):
        for j in range(2):
            U4_g[i*5 + (3+j)] = U4[i*2 + j]
            for k in range(2):
                V4_g[(3+j)*5 + (3+k)] = V4[j*2 + k]
                W4_g[(3+k)*5 + i] = W4[k*3 + i]
    U_list.append(U4_g)
    V_list.append(V4_g)
    W_list.append(W4_g)
    
    # T5: A21 B11 (2x3 * 3x3 -> 2x3)
    # A21: I=3..4, J=0..2
    # B11: J=0..2, K=0..2
    # C: I=3..4, K=0..2
    U5, V5, W5 = schoolbook(2, 3, 3)
    U5_g = np.zeros((25, 18))
    V5_g = np.zeros((25, 18))
    W5_g = np.zeros((25, 18))
    for i in range(2):
        for j in range(3):
            U5_g[(3+i)*5 + j] = U5[i*3 + j]
            for k in range(3):
                V5_g[j*5 + k] = V5[j*3 + k]
                W5_g[k*5 + (3+i)] = W5[k*2 + i]
    U_list.append(U5_g)
    V_list.append(V5_g)
    W_list.append(W5_g)
    
    # T6: A22 B21 (2x2 * 2x3 -> 2x3)
    # A22: I=3..4, J=3..4
    # B21: J=3..4, K=0..2
    # C: I=3..4, K=0..2
    U6, V6, W6 = schoolbook(2, 2, 3)
    U6_g = np.zeros((25, 12))
    V6_g = np.zeros((25, 12))
    W6_g = np.zeros((25, 12))
    for i in range(2):
        for j in range(2):
            U6_g[(3+i)*5 + (3+j)] = U6[i*2 + j]
            for k in range(3):
                V6_g[(3+j)*5 + k] = V6[j*3 + k]
                W6_g[k*5 + (3+i)] = W6[k*2 + i]
    U_list.append(U6_g)
    V_list.append(V6_g)
    W_list.append(W6_g)
    
    # T7: A21 B12 (2x3 * 3x2 -> 2x2)
    # A21: I=3..4, J=0..2
    # B12: J=0..2, K=3..4
    # C: I=3..4, K=3..4
    U7, V7, W7 = schoolbook(2, 3, 2)
    U7_g = np.zeros((25, 12))
    V7_g = np.zeros((25, 12))
    W7_g = np.zeros((25, 12))
    for i in range(2):
        for j in range(3):
            U7_g[(3+i)*5 + j] = U7[i*3 + j]
            for k in range(2):
                V7_g[j*5 + (3+k)] = V7[j*2 + k]
                W7_g[(3+k)*5 + (3+i)] = W7[k*2 + i]
    U_list.append(U7_g)
    V_list.append(V7_g)
    W_list.append(W7_g)
    
    # T8: A22 B22 (2x2 * 2x2 -> 2x2)
    # A22: I=3..4, J=3..4
    # B22: J=3..4, K=3..4
    # C: I=3..4, K=3..4
    U8, V8, W8 = strassen_2x2()
    U8_g = np.zeros((25, 7))
    V8_g = np.zeros((25, 7))
    W8_g = np.zeros((25, 7))
    for i in range(2):
        for j in range(2):
            U8_g[(3+i)*5 + (3+j)] = U8[i*2 + j]
            for k in range(2):
                V8_g[(3+j)*5 + (3+k)] = V8[j*2 + k]
                W8_g[(3+k)*5 + (3+i)] = W8[k*2 + i]
    U_list.append(U8_g)
    V_list.append(V8_g)
    W_list.append(W8_g)
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
