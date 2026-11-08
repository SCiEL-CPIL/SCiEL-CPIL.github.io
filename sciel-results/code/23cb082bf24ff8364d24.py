import numpy as np

def combine(U1, V1, W1, U2, V2, W2):
    """
    Combine two decompositions.
    U1, V1, W1 are lists of lists.
    """
    U = U1 + U2
    V = V1 + V2
    W = W1 + W2
    return U, V, W

def decomp_1_m_p(m, p):
    """
    Decomposition for 1 x m x p.
    Rank = m * p.
    U: (m, rank), V: (m*p, rank), W: (p, rank)
    Note: In the main problem, U is (n*m, rank). Here n=1, so U is (m, rank).
    V is (m*p, rank).
    W is (p*1, rank) = (p, rank).
    """
    rank = m * p
    U = [[0] * rank for _ in range(m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            U[j][col] = 1
            V[j * p + k][col] = 1
            W[k][col] = 1
            col += 1
    return U, V, W

def decomp_2_2_1():
    """
    Decomposition for 2 x 2 x 1.
    Rank = 4.
    """
    return decomp_1_m_p(2, 1) # Wait, 1 x 2 x 1 is different.
    # 2 x 2 x 1 is C = A B where A is 2x2, B is 2x1.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}. k is always 0.
    # This is just 4 dot products of length 2.
    rank = 4
    U = [[0] * rank for _ in range(2 * 2)] # n*m = 4
    V = [[0] * rank for _ in range(2 * 1)] # m*p = 2
    W = [[0] * rank for _ in range(1 * 2)] # p*n = 2
    
    col = 0
    for i in range(2):
        for j in range(2):
            U[i * 2 + j][col] = 1
            V[j * 1 + 0][col] = 1
            W[0 * 2 + i][col] = 1
            col += 1
    return U, V, W

def decomp_strassen_2_2_2():
    """
    Strassen decomposition for 2 x 2 x 2.
    Rank = 7.
    """
    # Standard Strassen
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # Indices:
    # U[i*m+j] -> i in 0..1, j in 0..1
    # V[j*p+k] -> j in 0..1, k in 0..1
    # W[k*n+i] -> k in 0..1, i in 0..1
    
    # M1
    # A11, A22, B11, B22
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1; W[1][1] = 1; W[2][2] = 1 # C11, C22? No.
    
    # Let's use a known table.
    # Col 0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][3] = 1 # C11, C22?
    # C11 = M1... C22 = M1...
    # W[k*n+i].
    # C11 is i=0, k=0 -> W[0].
    # C22 is i=1, k=1 -> W[1*2+1] = W[3].
    W[0][0] = 1; W[3][0] = 1
    
    # Col 1: M2
    # A21, A22, B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1 # C21 (i=1, k=0 -> W[0*2+1]=W[1]?)
    # C21 is i=1, k=0. W[k*n+i] = W[0*2+1] = W[1].
    W[1][1] = 1
    
    # Col 2: M3
    # A11, B12, -B22
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1 # C12 (i=0, k=1 -> W[1*2+0]=W[2]?)
    # C12 is i=0, k=1. W[1*2+0] = W[2].
    W[2][2] = 1
    W[3][2] = 1 # C22 (i=1, k=1 -> W[3])
    
    # Col 3: M4
    # A22, B21, -B11
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1 # C11
    W[1][3] = 1 # C21
    
    # Col 4: M5
    # A11, A12, B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C11
    W[2][4] = 1 # C12
    
    # Col 5: M6
    # A21, -A11, B11, B12
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1 # C22
    
    # Col 6: M7
    # A12, -A22, B21, B22
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1 # C11
    W[3][6] = 1 # C22
    
    return U, V, W

def decomp_2_2_4():
    """
    2 x 2 x 4.
    Split k into k1 in {0,1} and k2 in {2,3}.
    Sum of two 2 x 2 x 2.
    """
    U1, V1, W1 = decomp_strassen_2_2_2()
    U2, V2, W2 = decomp_strassen_2_2_2()
    
    # Shift indices for the second part
    # V index: j*p+k. p=2.
    # For k in {2,3}, k_local = k-2.
    # V2 has indices 0..3 (j*2+k_local).
    # We need to map V2[j*2+k_local] to V[j*2 + (k_local+2)].
    V2_shifted = [[0] * 7 for _ in range(2 * 4)]
    for j in range(2):
        for k_loc in range(2):
            V2_shifted[j * 4 + k_loc + 2] = V2[j * 2 + k_loc]
            
    # W index: k*n+i. n=2.
    # For k in {2,3}, k_local = k-2.
    # W2 has indices 0..3 (k_local*2+i).
    # We need to map W2[k_local*2+i] to W[(k_local+2)*2+i].
    W2_shifted = [[0] * 7 for _ in range(4 * 2)]
    for k_loc in range(2):
        for i in range(2):
            W2_shifted[(k_loc + 2) * 2 + i] = W2[k_loc * 2 + i]
            
    U, V, W = combine(U1, V1, W1, U2, V2_shifted, W2_shifted)
    return U, V, W

def decomp_2_2_3():
    """
    2 x 2 x 3.
    Split k into k1 in {0,1} and k2 in {2}.
    Sum of 2 x 2 x 2 and 2 x 2 x 1.
    """
    U1, V1, W1 = decomp_strassen_2_2_2()
    U2, V2, W2 = decomp_2_2_1()
    
    # Shift indices for the second part (2x2x1)
    # V index: j*p+k. p=1.
    # V2 has indices 0..1 (j*1+0).
    # We need to map to V[j*3 + 2].
    V2_shifted = [[0] * 4 for _ in range(2 * 3)]
    for j in range(2):
        V2_shifted[j * 3 + 2] = V2[j]
        
    # W index: k*n+i. n=2.
    # W2 has indices 0..1 (0*2+i).
    # We need to map to W[2*2+i].
    W2_shifted = [[0] * 4 for _ in range(3 * 2)]
    for i in range(2):
        W2_shifted[2 * 2 + i] = W2[i]
        
    U, V, W = combine(U1, V1, W1, U2, V2_shifted, W2_shifted)
    return U, V, W

def decomp_3_2_4():
    """
    3 x 2 x 4.
    Split i into i1=0 and i2 in {1,2}.
    Sum of 1 x 2 x 4 and 2 x 2 x 4.
    """
    U1, V1, W1 = decomp_1_m_p(2, 4)
    U2, V2, W2 = decomp_2_2_4()
    
    # Shift indices for the second part (2x2x4)
    # U index: i*m+j. m=2.
    # U2 has indices 0..3 (i_local*2+j).
    # We need to map to U[(i_local+1)*2+j].
    U2_shifted = [[0] * 14 for _ in range(3 * 2)]
    for i_loc in range(2):
        for j in range(2):
            U2_shifted[(i_loc + 1) * 2 + j] = U2[i_loc * 2 + j]
            
    # V index: j*p+k. p=4.
    # V2 is already for p=4. No shift needed for V.
    
    # W index: k*n+i. n=2.
    # W2 has indices 0..7 (k*2+i_local).
    # We need to map to W[k*3 + (i_local+1)].
    W2_shifted = [[0] * 14 for _ in range(4 * 3)]
    for k in range(4):
        for i_loc in range(2):
            W2_shifted[k * 3 + i_loc + 1] = W2[k * 2 + i_loc]
            
    U, V, W = combine(U1, V1, W1, U2_shifted, V2, W2_shifted)
    return U, V, W

def decomp_3_2_3():
    """
    3 x 2 x 3.
    Split i into i1=0 and i2 in {1,2}.
    Sum of 1 x 2 x 3 and 2 x 2 x 3.
    """
    U1, V1, W1 = decomp_1_m_p(2, 3)
    U2, V2, W2 = decomp_2_2_3()
    
    # Shift indices for the second part (2x2x3)
    # U index: i*m+j. m=2.
    # U2 has indices 0..3 (i_local*2+j).
    # We need to map to U[(i_local+1)*2+j].
    U2_shifted = [[0] * 11 for _ in range(3 * 2)]
    for i_loc in range(2):
        for j in range(2):
            U2_shifted[(i_loc + 1) * 2 + j] = U2[i_loc * 2 + j]
            
    # V index: j*p+k. p=3.
    # V2 is already for p=3. No shift needed for V.
    
    # W index: k*n+i. n=2.
    # W2 has indices 0..5 (k*2+i_local).
    # We need to map to W[k*3 + (i_local+1)].
    W2_shifted = [[0] * 11 for _ in range(3 * 3)]
    for k in range(3):
        for i_loc in range(2):
            W2_shifted[k * 3 + i_loc + 1] = W2[k * 2 + i_loc]
            
    U, V, W = combine(U1, V1, W1, U2_shifted, V2, W2_shifted)
    return U, V, W

def decomp_3_4_4():
    """
    3 x 4 x 4.
    Split j into j1 in {0,1} and j2 in {2,3}.
    Sum of two 3 x 2 x 4.
    """
    U1, V1, W1 = decomp_3_2_4()
    U2, V2, W2 = decomp_3_2_4()
    
    # Shift indices for the second part
    # U index: i*m+j. m=4.
    # U2 has indices 0..5 (i*2+j_local).
    # We need to map to U[i*4 + (j_local+2)].
    U2_shifted = [[0] * 22 for _ in range(3 * 4)]
    for i in range(3):
        for j_loc in range(2):
            U2_shifted[i * 4 + j_loc + 2] = U2[i * 2 + j_loc]
            
    # V index: j*p+k. p=4.
    # V2 has indices 0..7 (j_local*4+k).
    # We need to map to V[(j_local+2)*4+k].
    V2_shifted = [[0] * 22 for _ in range(4 * 4)]
    for j_loc in range(2):
        for k in range(4):
            V2_shifted[(j_loc + 2) * 4 + k] = V2[j_loc * 4 + k]
            
    # W index: k*n+i. n=3.
    # W2 is already for n=3. No shift needed for W.
            
    U, V, W = combine(U1, V1, W1, U2_shifted, V2_shifted, W2)
    return U, V, W

def decomp_3_4_3():
    """
    3 x 4 x 3.
    Split j into j1 in {0,1} and j2 in {2,3}.
    Sum of two 3 x 2 x 3.
    """
    U1, V1, W1 = decomp_3_2_3()
    U2, V2, W2 = decomp_3_2_3()
    
    # Shift indices for the second part
    # U index: i*m+j. m=4.
    # U2 has indices 0..5 (i*2+j_local).
    # We need to map to U[i*4 + (j_local+2)].
    U2_shifted = [[0] * 17 for _ in range(3 * 4)]
    for i in range(3):
        for j_loc in range(2):
            U2_shifted[i * 4 + j_loc + 2] = U2[i * 2 + j_loc]
            
    # V index: j*p+k. p=3.
    # V2 has indices 0..5 (j_local*3+k).
    # We need to map to V[(j_local+2)*3+k].
    V2_shifted = [[0] * 17 for _ in range(4 * 3)]
    for j_loc in range(2):
        for k in range(3):
            V2_shifted[(j_loc + 2) * 3 + k] = V2[j_loc * 3 + k]
            
    # W index: k*n+i. n=3.
    # W2 is already for n=3. No shift needed for W.
            
    U, V, W = combine(U1, V1, W1, U2_shifted, V2_shifted, W2)
    return U, V, W

def construct():
    n, m, p = 3, 4, 7
    U1, V1, W1 = decomp_3_4_4()
    U2, V2, W2 = decomp_3_4_3()
    
    # Shift indices for the second part (3x4x3)
    # U index: i*m+j. m=4.
    # U2 has indices 0..11 (i*4+j).
    # No shift needed for U.
    
    # V index: j*p+k. p=7.
    # V2 has indices 0..11 (j*3+k).
    # We need to map to V[j*7 + (k+4)].
    V2_shifted = [[0] * 34 for _ in range(4 * 7)]
    for j in range(4):
        for k in range(3):
            V2_shifted[j * 7 + k + 4] = V2[j * 3 + k]
            
    # W index: k*n+i. n=3.
    # W2 has indices 0..8 (k*3+i).
    # We need to map to W[(k+4)*3+i].
    W2_shifted = [[0] * 34 for _ in range(7 * 3)]
    for k in range(3):
        for i in range(3):
            W2_shifted[(k + 4) * 3 + i] = W2[k * 3 + i]
            
    U, V, W = combine(U1, V1, W1, U2, V2_shifted, W2_shifted)
    
    rank = len(U[0])
    return {"U": U, "V": V, "W": W}
