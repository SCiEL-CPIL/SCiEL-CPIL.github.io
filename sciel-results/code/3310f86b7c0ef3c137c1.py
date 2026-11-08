import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = A11(B11+B22)
    U2[0, 0] = 1 # A11
    V2[0, 0] = 1 # B11
    V2[3, 0] = 1 # B22
    W2[0, 0] = 1 # C11
    
    # M2 = (A11+A22)B22
    U2[0, 1] = 1 # A11
    U2[3, 1] = 1 # A22
    V2[3, 1] = 1 # B22
    W2[3, 1] = 1 # C22
    
    # M3 = (A11+A12)B21
    U2[0, 2] = 1 # A11
    U2[1, 2] = 1 # A12
    V2[2, 2] = 1 # B21
    W2[2, 2] = 1 # C21
    
    # M4 = (A21+A22)B11
    U2[2, 3] = 1 # A21
    U2[3, 3] = 1 # A22
    V2[0, 3] = 1 # B11
    W2[3, 3] = 1 # C22
    
    # M5 = A11(B12-B22)
    U2[0, 4] = 1 # A11
    V2[1, 4] = 1 # B12
    V2[3, 4] = -1 # -B22
    W2[0, 4] = 1 # C11
    
    # M6 = (A21-A11)(B11+B12)
    U2[2, 5] = 1 # A21
    U2[0, 5] = -1 # -A11
    V2[0, 5] = 1 # B11
    V2[1, 5] = 1 # B12
    W2[3, 5] = 1 # C22
    
    # M7 = (A12-A22)(B21+B22)
    U2[1, 6] = 1 # A12
    U2[3, 6] = -1 # -A22
    V2[2, 6] = 1 # B21
    V2[3, 6] = 1 # B22
    W2[0, 6] = 1 # C11
    
    # Kronecker product for 4x4x4
    # U4: 16x49, V4: 16x49, W4: 16x49
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Add 4 terms for the 5th column (k=4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Term j: U[i*4+j, r]=1, V[j*5+4, r]=1, W[4*4+i, r]=1
    # We need to add 4 columns to U4, V4, W4
    # Current rank 49. New rank 53.
    
    U_final = np.zeros((16, 53))
    V_final = np.zeros((20, 53)) # m*p = 4*5 = 20
    W_final = np.zeros((20, 53)) # p*n = 5*4 = 20
    
    # Copy Strassen part
    U_final[:, :49] = U4
    V_final[:, :49] = V4
    W_final[:, :49] = W4
    
    # Add 4 terms
    for j in range(4):
        r = 49 + j
        for i in range(4):
            U_final[i * 4 + j, r] = 1
        V_final[j * 5 + 4, r] = 1
        for i in range(4):
            W_final[4 * 4 + i, r] = 1
            
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
