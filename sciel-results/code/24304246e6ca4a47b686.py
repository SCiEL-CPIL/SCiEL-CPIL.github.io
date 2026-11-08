import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Decomposition for 3x2x2 (Rank 11)
    # Split into 2x2x2 (Strassen, Rank 7) and 1x2x2 (Schoolbook, Rank 4)
    
    # Strassen 2x2x2
    # U_S, V_S, W_S are 2x7
    U_S = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0]
    ])
    V_S = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0]
    ])
    # Wait, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The factor matrices U, V, W correspond to the inputs.
    # U rows: A11, A21 (for 2x2 A)
    # V rows: B11, B21 (for 2x2 B) -- Wait, V is m x rank. m=2.
    # W rows: C11, C21 (for 2x2 C) -- Wait, W is p x rank. p=2.
    
    # Let's use the explicit coefficients for Strassen.
    # Columns correspond to M1..M7.
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, -1], # A11 coeffs
        [0, 1, 0, 1, 0, 1, 1]    # A21 coeffs (Wait, A is 2x2. Rows are A11, A21? No, A11, A12, A21, A22?)
    ])
    # The task U is n*m x rank. For 2x2x2, n=2, m=2. U is 4x7?
    # No, in my sub-decomposition, I am treating the "3x2x2" problem.
    # For the "2x2x2" part, n=2, m=2, p=2.
    # U size: 2*2=4? No, the factor U in the tensor decomposition T(i,j,k) = sum U[i,c]V[j,c]W[k,c]
    # has U of size n x rank, V of size m x rank, W of size p x rank.
    # But the task output requires U of size n*m x rank.
    # This implies the "U" in the task is actually the flattened version of the "A" factors?
    # No, the tensor decomposition is for the multiplication map.
    # T(i,j,k) is the coefficient.
    # The factors are U (n x r), V (m x r), W (p x r).
    # The task asks for U (n*m x r), V (m*p x r), W (p*n x r).
    # This is different.
    # Let's re-read the seed program.
    # U[i*m+j, c] = 1. This suggests U is indexed by (i,j).
    # V[j*p+k, c] = 1. V is indexed by (j,k).
    # W[k*n+i, c] = 1. W is indexed by (k,i).
    # This is NOT the standard tensor factorization U(n,r) V(m,r) W(p,r).
    # This is a factorization of the 3-way tensor T(i,j,k) into a sum of rank-1 tensors?
    # T(i,j,k) = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # Yes, this is the standard tensor rank definition.
    # The "U" in the task is the flattened U tensor (n*m x r).
    # The "V" in the task is the flattened V tensor (m*p x r).
    # The "W" in the task is the flattened W tensor (p*n x r).
    
    # So for 2x2x2 Strassen:
    # U_flat is 4x7. Indices (i,j).
    # V_flat is 4x7. Indices (j,k).
    # W_flat is 4x7. Indices (k,i).
    
    # Let's construct Strassen flat matrices.
    # M1: (A11+A22)(B11+B22)
    # U part: A11=1, A22=1. Others 0.
    # V part: B11=1, B22=1. Others 0.
    # W part: C11=1, C22=1. Others 0.
    
    U_S_flat = np.zeros((4, 7))
    V_S_flat = np.zeros((4, 7))
    W_S_flat = np.zeros((4, 7))
    
    # M1
    U_S_flat[0, 0] = 1 # A11 (i=0,j=0)
    U_S_flat[3, 0] = 1 # A22 (i=1,j=1)
    V_S_flat[0, 0] = 1 # B11 (j=0,k=0)
    V_S_flat[3, 0] = 1 # B22 (j=1,k=1)
    W_S_flat[0, 0] = 1 # C11 (k=0,i=0)
    W_S_flat[3, 0] = 1 # C22 (k=1,i=1)
    
    # M2: (A21+A22)B11
    U_S_flat[2, 1] = 1 # A21 (i=1,j=0)
    U_S_flat[3, 1] = 1 # A22
    V_S_flat[0, 1] = 1 # B11
    W_S_flat[0, 1] = 1 # C21 (k=0,i=1)
    
    # M3: A11(B21-B22)
    U_S_flat[0, 2] = 1 # A11
    V_S_flat[1, 2] = 1 # B21 (j=0,k=1)
    V_S_flat[3, 2] = -1 # B22 (j=1,k=1)
    W_S_flat[1, 2] = 1 # C12 (k=1,i=0)
    
    # M4: A22(B12-B11)
    U_S_flat[3, 3] = 1 # A22
    V_S_flat[2, 3] = 1 # B12 (j=1,k=0)
    V_S_flat[0, 3] = -1 # B11
    W_S_flat[0, 3] = 1 # C11
    W_S_flat[3, 3] = 1 # C22? No, M4 contributes to C11 and C22?
    # C11 = M1 + M4 - M5 + M7
    # C22 = M1 - M2 + M3 + M6
    # So M4 contributes to C11.
    W_S_flat[0, 3] = 1
    
    # M5: (A11+A12)B22
    U_S_flat[0, 4] = 1 # A11
    U_S_flat[1, 4] = 1 # A12 (i=0,j=1)
    V_S_flat[3, 4] = 1 # B22
    W_S_flat[0, 4] = -1 # C11
    W_S_flat[3, 4] = -1 # C22? No.
    # C11 = ... - M5
    # C22 = ...
    # Wait, M5 is (A11+A12)B22.
    # It contributes to C12? No.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # So M5 contributes to C11 (-1) and C12 (+1).
    W_S_flat[0, 4] = -1
    W_S_flat[1, 4] = 1
    
    # M6: (A21-A11)(B11+B12)
    U_S_flat[2, 5] = 1 # A21
    U_S_flat[0, 5] = -1 # A11
    V_S_flat[0, 5] = 1 # B11
    V_S_flat[2, 5] = 1 # B12
    W_S_flat[3, 5] = 1 # C22
    
    # M7: (A12-A22)(B21+B22)
    U_S_flat[1, 6] = 1 # A12
    U_S_flat[3, 6] = -1 # A22
    V_S_flat[1, 6] = 1 # B21
    V_S_flat[3, 6] = 1 # B22
    W_S_flat[0, 6] = 1 # C11
    
    # Now 1x2x2 Schoolbook (Rank 4)
    # n=1, m=2, p=2.
    # U_flat (2x4), V_flat (4x4), W_flat (2x4).
    # Terms: A1 B11, A1 B12, A2 B21, A2 B22.
    U_S1_flat = np.zeros((2, 4))
    V_S1_flat = np.zeros((4, 4))
    W_S1_flat = np.zeros((2, 4))
    
    U_S1_flat[0, 0] = 1 # A1
    V_S1_flat[0, 0] = 1 # B11
    W_S1_flat[0, 0] = 1 # C11
    
    U_S1_flat[0, 1] = 1 # A1
    V_S1_flat[1, 1] = 1 # B12
    W_S1_flat[1, 1] = 1 # C12
    
    U_S1_flat[1, 2] = 1 # A2
    V_S1_flat[2, 2] = 1 # B21
    W_S1_flat[0, 2] = 1 # C21
    
    U_S1_flat[1, 3] = 1 # A2
    V_S1_flat[3, 3] = 1 # B22
    W_S1_flat[1, 3] = 1 # C22

    # Combine for 3x2x2
    # U_322 (6x11) -- Wait, n=3, m=2. U is 3*2=6 rows.
    # Rows 0,1 from U_S_flat (A11, A12)
    # Rows 2,3 from U_S_flat (A21, A22)
    # Rows 4,5 from U_S1_flat (A1, A2) -- Wait, U_S1_flat has 2 rows (A1, A2).
    # The "3" dimension is split into 2 (from Strassen) and 1 (from Schoolbook).
    # So U_322 rows 0-3 correspond to the 2x2 block.
    # U_322 rows 4-5 correspond to the 1x2 block.
    # But U_S1_flat has 2 rows (m=2).
    # So U_322 is 6x11.
    
    U_322 = np.zeros((6, 11))
    V_322 = np.zeros((4, 11)) # m=2, p=2 -> 4 rows
    W_322 = np.zeros((4, 11)) # p=2, n=3? No, W is p*n. p=2, n=3. 6 rows.
    
    # Wait, W for 3x2x2 is p*n = 2*3 = 6 rows.
    # W_S_flat is 4x7 (p=2, n=2).
    # W_S1_flat is 2x4 (p=2, n=1).
    
    # Fill U_322
    U_322[0:4, 0:7] = U_S_flat
    U_322[4:6, 7:11] = U_S1_flat
    
    # Fill V_322
    V_322[0:4, 0:7] = V_S_flat
    V_322[0:4, 7:11] = V_S1_flat
    
    # Fill W_322
    # W_S_flat indices (k,i) for i in 0..1.
    # W_S1_flat indices (k,i) for i in 0..0.
    # W_322 indices (k, i_total) for i_total in 0..2.
    # i_total 0,1 come from Strassen. i_total 2 comes from Schoolbook.
    W_322[0:4, 0:7] = W_S_flat # Rows 0-3 (k=0,1; i=0,1)
    # W_S1_flat has 2 rows (k=0,1; i=0).
    # We need to place them in rows corresponding to i=2.
    # W_322 row index = k*3 + i.
    # For i=2, rows are 2, 5.
    W_322[2, 7:11] = W_S1_flat[0] # k=0, i=2
    W_322[5, 7:11] = W_S1_flat[1] # k=1, i=2

    # 2. Decomposition for 1x2x3 (Rank 6)
    # n=1, m=2, p=3.
    # U_123 (2x6), V_123 (6x6), W_123 (3x6).
    U_123 = np.zeros((2, 6))
    V_123 = np.zeros((6, 6))
    W_123 = np.zeros((3, 6))
    
    # Terms: A1 B11, A1 B12, A1 B13, A2 B21, A2 B22, A2 B23
    U_123[0, 0] = 1; V_123[0, 0] = 1; W_123[0, 0] = 1
    U_123[0, 1] = 1; V_123[1, 1] = 1; W_123[1, 1] = 1
    U_123[0, 2] = 1; V_123[2, 2] = 1; W_123[2, 2] = 1
    U_123[1, 3] = 1; V_123[3, 3] = 1; W_123[0, 3] = 1
    U_123[1, 4] = 1; V_123[4, 4] = 1; W_123[1, 4] = 1
    U_123[1, 5] = 1; V_123[5, 5] = 1; W_123[2, 5] = 1

    # 3. Tensor Product
    # U_final = U_322 (kron) U_123
    # U_322 is 6x11. U_123 is 2x6.
    # Result 12x66.
    # V_final = V_322 (kron) V_123
    # V_322 is 4x11. V_123 is 6x6.
    # Result 24x66.
    # W_final = W_322 (kron) W_123
    # W_322 is 6x11. W_123 is 3x6.
    # Result 18x66.
    
    U_final = np.kron(U_322, U_123)
    V_final = np.kron(V_322, V_123)
    W_final = np.kron(W_322, W_123)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
