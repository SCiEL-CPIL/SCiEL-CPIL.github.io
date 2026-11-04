import numpy as np

def get_M_n1p(n, p):
    """Returns U, V, W for M_{n,1,p} with rank n*p."""
    rank = n * p
    U = np.zeros((n * 1, rank))
    V = np.zeros((1 * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for k in range(p):
            U[i, col] = 1.0
            V[k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    return U, V, W

def get_M_nm1(n, m):
    """Returns U, V, W for M_{n,m,1} with rank n*m."""
    rank = n * m
    U = np.zeros((n * m, rank))
    V = np.zeros((m * 1, rank))
    W = np.zeros((1 * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j, col] = 1.0
            W[i, col] = 1.0
            col += 1
    return U, V, W

def get_M_222():
    """Returns U, V, W for M_{2,2,2} using Strassen (Rank 7)."""
    # Strassen matrices for 2x2
    # U is 4x7, V is 4x7, W is 4x7
    # Indices for U: i*2+j
    # Indices for V: j*2+k
    # Indices for W: k*2+i
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen formulas:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # We need to map these to tensor decomposition.
    # Each multiplication mr corresponds to a term U_r * V_r * W_r.
    # U_r has non-zeros at A indices involved in the first factor of mr.
    # V_r has non-zeros at B indices involved in the second factor of mr.
    # W_r has non-zeros at C indices that receive the contribution from mr.
    
    # m1: (a00+a11)(b00+b11) -> C00 (+1), C11 (+1)
    U[0, 0] = 1; U[3, 0] = 1 # a00, a11
    V[0, 0] = 1; V[3, 0] = 1 # b00, b11
    W[0, 0] = 1; W[3, 0] = 1 # c00, c11 (indices: 0*2+0=0, 1*2+1=3)
    
    # m2: (a10+a11)b00 -> C10 (+1), C11 (-1)
    U[2, 1] = 1; U[3, 1] = 1 # a10, a11
    V[0, 1] = 1 # b00
    W[2, 1] = 1 # c10 (1*2+0=2)
    W[3, 1] = -1 # c11
    
    # m3: a00(b01-b11) -> C01 (+1), C11 (+1)
    U[0, 2] = 1 # a00
    V[1, 2] = 1; V[3, 2] = -1 # b01, b11
    W[1, 2] = 1 # c01 (0*2+1=1)
    W[3, 2] = 1 # c11
    
    # m4: a11(b10-b00) -> C00 (+1), C10 (+1)
    U[3, 3] = 1 # a11
    V[2, 3] = 1; V[0, 3] = -1 # b10, b00
    W[0, 3] = 1 # c00
    W[2, 3] = 1 # c10
    
    # m5: (a00+a01)b11 -> C00 (-1), C01 (+1)
    U[0, 4] = 1; U[1, 4] = 1 # a00, a01
    V[3, 4] = 1 # b11
    W[0, 4] = -1 # c00
    W[1, 4] = 1 # c01
    
    # m6: (a10-a00)(b00+b01) -> C11 (+1)
    U[2, 5] = 1; U[0, 5] = -1 # a10, a00
    V[0, 5] = 1; V[1, 5] = 1 # b00, b01
    W[3, 5] = 1 # c11
    
    # m7: (a01-a11)(b10+b11) -> C00 (+1)
    U[1, 6] = 1; U[3, 6] = -1 # a01, a11
    V[2, 6] = 1; V[3, 6] = 1 # b10, b11
    W[0, 6] = 1 # c00
    
    return U, V, W

def construct():
    n, m, p = 2, 3, 7
    total_rank = 39
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # 1. M_{2,1,7} at offset i=0, j=0, k=0
    U1, V1, W1 = get_M_n1p(2, 7)
    U[0:2, col:col+14] = U1
    V[0:7, col:col+14] = V1 # V index j*p+k. j=0. So rows 0..6
    W[0:14, col:col+14] = W1 # W index k*n+i. k=0..6, i=0..1
    col += 14
    
    # 2. M_{2,2,2} at offset i=0, j=1, k=0
    U2, V2, W2 = get_M_222()
    # U rows: i*2+j. i=0..1, j=1..2 (relative to block). Global j=1,2.
    # Block U is 4x7. Rows 0,1,2,3 correspond to (0,1),(0,2),(1,1),(1,2) relative?
    # No, standard M_{2,2,2} U indices are i*2+j for i=0,1 j=0,1.
    # We need to map block j=0,1 to global j=1,2.
    # Block i=0,1 to global i=0,1.
    # So global row for block row r (0..3):
    # r=0 (i=0,j=0) -> global i=0, j=1 -> idx 0*2+1=1
    # r=1 (i=0,j=1) -> global i=0, j=2 -> idx 0*2+2=2
    # r=2 (i=1,j=0) -> global i=1, j=1 -> idx 1*2+1=3
    # r=3 (i=1,j=1) -> global i=1, j=2 -> idx 1*2+2=4
    u_map = [1, 2, 3, 4]
    # V block indices j*p+k. Block p=2. j=0,1. k=0,1.
    # Global j=1,2. Global k=0,1.
    # Block row r (0..3) for V:
    # r=0 (j=0,k=0) -> global j=1, k=0 -> idx 1*7+0=7
    # r=1 (j=0,k=1) -> global j=1, k=1 -> idx 1*7+1=8
    # r=2 (j=1,k=0) -> global j=2, k=0 -> idx 2*7+0=14
    # r=3 (j=1,k=1) -> global j=2, k=1 -> idx 2*7+1=15
    v_map = [7, 8, 14, 15]
    # W block indices k*n+i. Block n=2. k=0,1. i=0,1.
    # Global k=0,1. Global i=0,1.
    # Block row r (0..3) for W:
    # r=0 (k=0,i=0) -> global k=0, i=0 -> idx 0*2+0=0
    # r=1 (k=0,i=1) -> global k=0, i=1 -> idx 0*2+1=1
    # r=2 (k=1,i=0) -> global k=1, i=0 -> idx 1*2+0=2
    # r=3 (k=1,i=1) -> global k=1, i=1 -> idx 1*2+1=3
    w_map = [0, 1, 2, 3]
    
    for r in range(4):
        U[u_map[r], col:col+7] = U2[r, :]
        V[v_map[r], col:col+7] = V2[r, :]
        W[w_map[r], col:col+7] = W2[r, :]
    col += 7
    
    # 3. M_{2,2,2} at offset i=0, j=1, k=2
    U3, V3, W3 = get_M_222()
    # Same U map
    # V: global j=1,2. k=2,3.
    # r=0 (j=0,k=0) -> g_j=1, g_k=2 -> 1*7+2=9
    # r=1 (j=0,k=1) -> g_j=1, g_k=3 -> 1*7+3=10
    # r=2 (j=1,k=0) -> g_j=2, g_k=2 -> 2*7+2=16
    # r=3 (j=1,k=1) -> g_j=2, g_k=3 -> 2*7+3=17
    v_map3 = [9, 10, 16, 17]
    # W: global k=2,3. i=0,1.
    # r=0 (k=0,i=0) -> g_k=2, g_i=0 -> 2*2+0=4
    # r=1 (k=0,i=1) -> g_k=2, g_i=1 -> 2*2+1=5
    # r=2 (k=1,i=0) -> g_k=3, g_i=0 -> 3*2+0=6
    # r=3 (k=1,i=1) -> g_k=3, g_i=1 -> 3*2+1=7
    w_map3 = [4, 5, 6, 7]
    
    for r in range(4):
        U[u_map[r], col:col+7] = U3[r, :]
        V[v_map3[r], col:col+7] = V3[r, :]
        W[w_map3[r], col:col+7] = W3[r, :]
    col += 7
    
    # 4. M_{2,2,1} at offset i=0, j=1, k=4
    U4, V4, W4 = get_M_nm1(2, 2)
    # U map same: [1, 2, 3, 4]
    # V: global j=1,2. k=4. p=1.
    # r=0 (j=0) -> g_j=1, g_k=4 -> 1*7+4=11
    # r=1 (j=1) -> g_j=2, g_k=4 -> 2*7+4=18
    v_map4 = [11, 18]
    # W: global k=4. i=0,1. n=2.
    # r=0 (i=0) -> g_k=4, g_i=0 -> 4*2+0=8
    # r=1 (i=1) -> g_k=4, g_i=1 -> 4*2+1=9
    w_map4 = [8, 9]
    
    for r in range(4):
        U[u_map[r], col:col+4] = U4[r, :]
        V[v_map4[r//2], col:col+4] = V4[r%2, :] # Wait, V4 is 2x4. Rows j=0,1.
        # My loop r is 0..3 for U. But V4 only has 2 rows.
        # U4 is 4x4. Rows (0,0),(0,1),(1,0),(1,1).
        # V4 is 2x4. Rows j=0,1.
        # W4 is 2x4. Rows i=0,1.
        
        # Correct mapping:
        # r=0: U row 0 (i=0,j=0). V row 0 (j=0). W row 0 (i=0).
        # r=1: U row 1 (i=0,j=1). V row 1 (j=1). W row 0 (i=0).
        # r=2: U row 2 (i=1,j=0). V row 0 (j=0). W row 1 (i=1).
        # r=3: U row 3 (i=1,j=1). V row 1 (j=1). W row 1 (i=1).
        
        # Let's just do it explicitly or fix the loop.
    # Redo block 4 explicitly to be safe
    U[1, col:col+4] = U4[0, :] # i=0,j=1
    U[2, col:col+4] = U4[1, :] # i=0,j=2
    U[3, col:col+4] = U4[2, :] # i=1,j=1
    U[4, col:col+4] = U4[3, :] # i=1,j=2
    
    V[11, col:col+4] = V4[0, :] # j=1
    V[18, col:col+4] = V4[1, :] # j=2
    
    W[8, col:col+4] = W4[0, :] # i=0
    W[9, col:col+4] = W4[1, :] # i=1
    
    col += 4
    
    # 5. M_{2,2,2} at offset i=0, j=1, k=5
    U5, V5, W5 = get_M_222()
    # V: global j=1,2. k=5,6.
    # r=0 -> 1*7+5=12
    # r=1 -> 1*7+6=13
    # r=2 -> 2*7+5=19
    # r=3 -> 2*7+6=20
    v_map5 = [12, 13, 19, 20]
    # W: global k=5,6. i=0,1.
    # r=0 -> 5*2+0=10
    # r=1 -> 5*2+1=11
    # r=2 -> 6*2+0=12
    # r=3 -> 6*2+1=13
    w_map5 = [10, 11, 12, 13]
    
    for r in range(4):
        U[u_map[r], col:col+7] = U5[r, :]
        V[v_map5[r], col:col+7] = V5[r, :]
        W[w_map5[r], col:col+7] = W5[r, :]
    col += 7
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
