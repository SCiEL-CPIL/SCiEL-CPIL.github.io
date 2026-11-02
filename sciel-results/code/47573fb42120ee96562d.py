import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # The decomposition is M_{2,2,3} (o) M_{2,2,2}
    # M_{2,2,3} has rank 12 (schoolbook)
    # M_{2,2,2} has rank 7 (Strassen)
    # Total rank = 12 * 7 = 84
    
    # 1. Construct M_{2,2,3} schoolbook decomposition
    # U1 shape: (2*2, 12) = (4, 12)
    # V1 shape: (2*3, 12) = (6, 12)
    # W1 shape: (3*2, 12) = (6, 12)
    
    n1, m1, p1 = 2, 2, 3
    rank1 = n1 * m1 * p1
    U1 = np.zeros((n1 * m1, rank1))
    V1 = np.zeros((m1 * p1, rank1))
    W1 = np.zeros((p1 * n1, rank1))
    
    col = 0
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                U1[i1 * m1 + j1, col] = 1
                V1[j1 * p1 + k1, col] = 1
                W1[k1 * n1 + i1, col] = 1
                col += 1
                
    # 2. Construct M_{2,2,2} Strassen decomposition
    # U2 shape: (2*2, 7) = (4, 7)
    # V2 shape: (2*2, 7) = (4, 7)
    # W2 shape: (2*2, 7) = (4, 7)
    
    n2, m2, p2 = 2, 2, 2
    rank2 = 7
    U2 = np.zeros((n2 * m2, rank2))
    V2 = np.zeros((m2 * p2, rank2))
    W2 = np.zeros((p2 * n2, rank2))
    
    # Strassen's 7 multiplications
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Indices:
    # A: i, j (0..1, 0..1) -> idx = i*2 + j
    # B: j, k (0..1, 0..1) -> idx = j*2 + k
    # C: i, k (0..1, 0..1) -> idx = i*2 + k
    
    # M1: (A11+A22)(B11+B22) -> C11 += M1-M4-M5+M7, C12 += M3+M5, C21 += M2+M4, C22 += M1-M2+M6-M7
    # Note: Standard Strassen output reconstruction:
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 - M7
    
    # Let's populate U2, V2, W2
    # Each column c corresponds to one of the 7 products.
    
    # M1: A11+A22, B11+B22
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = 1; W2[2, 0] = 1; W2[3, 0] = 1 # Wait, W is output.
    # W index is k*n + i.
    # For C11 (i=0, k=0): idx 0. C12 (i=0, k=1): idx 1. C21 (i=1, k=0): idx 2. C22 (i=1, k=1): idx 3.
    # M1 contributes to C11, C12, C21, C22 with coeffs 1, 1, 1, 1?
    # No, M1 is a scalar.
    # C11 += M1
    # C12 += M1? No.
    # Let's use the explicit formulas.
    
    # Reset W2
    W2 = np.zeros((4, 7))
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1 # C11
    W2[1, 0] = 1 # C12? No, C12 = M3+M5.
    # Let's re-verify Strassen coefficients.
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 - M7
    
    # Col 0 (M1):
    # U: A11+A22 (idx 0, 3)
    # V: B11+B22 (idx 0, 3)
    # W: C11 (idx 0), C22 (idx 3)
    W2[0, 0] = 1
    W2[3, 0] = 1
    
    # Col 1 (M2): (A21+A22)B11
    # U: A21+A22 (idx 2, 3)
    # V: B11 (idx 0)
    # W: C21 (idx 2), C22 (idx 3)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    W2[3, 1] = -1
    
    # Col 2 (M3): A11(B12-B22)
    # U: A11 (idx 0)
    # V: B12-B22 (idx 1, 3)
    # W: C12 (idx 1)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # Col 3 (M4): A22(B21-B11)
    # U: A22 (idx 3)
    # V: B21-B11 (idx 2, 0)
    # W: C11 (idx 0), C21 (idx 2)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = -1
    W2[2, 3] = 1
    
    # Col 4 (M5): (A11+A12)B22
    # U: A11+A12 (idx 0, 1)
    # V: B22 (idx 3)
    # W: C11 (idx 0), C12 (idx 1)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1
    W2[1, 4] = 1
    
    # Col 5 (M6): (A21-A11)(B11+B12)
    # U: A21-A11 (idx 2, 0)
    # V: B11+B12 (idx 0, 1)
    # W: C22 (idx 3)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Col 6 (M7): (A12-A22)(B21+B22)
    # U: A12-A22 (idx 1, 3)
    # V: B21+B22 (idx 2, 3)
    # W: C11 (idx 0), C22 (idx 3)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    W2[3, 6] = -1

    # 3. Kronecker Product
    # U = U1 (o) U2 ?
    # The tensor product T1 (o) T2 has components T1_abc T2_ijk.
    # The resulting U, V, W are formed by the Kronecker product of the factor matrices.
    # U_total = U1 (o) U2
    # V_total = V1 (o) V2
    # W_total = W1 (o) W2
    
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
