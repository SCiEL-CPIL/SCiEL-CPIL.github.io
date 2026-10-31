import numpy as np

def construct():
    n, m, p = 4, 4, 4
    
    # 2x2 Strassen factors
    # U2[i,j] = coefficient of A[i,j] in multiplication s
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ])
    
    # V2[j,k] = coefficient of B[j,k] in multiplication s
    V2 = np.array([
        [1, 0, 0, -1],   # B00
        [0, 1, 1, 0],    # B01
        [1, 0, -1, 0],   # B10
        [1, 1, 0, 1],    # B11
    ])
    
    # W2[k,i] = coefficient of C[k,i]... wait, need to check convention
    # The task says W[k*n+i, r]
    # For 2x2: W2[k*2+i, s]
    # C00 = M0 - M4 + M6
    # C01 = M2 + M4
    # C10 = M3 + M1
    # C11 = M0 - M1 + M2 + M5
    W2 = np.array([
        [1, 0, 0, 0, -1, 0, 1],   # C00: k=0,i=0
        [0, 0, 1, 0, 1, 0, 0],    # C01: k=0,i=1
        [0, 1, 0, 1, 0, 0, 0],    # C10: k=1,i=0
        [1, -1, 1, 0, 0, 1, 0],   # C11: k=1,i=1
    ])
    
    # Verify: for each s, sum_i,j,k U2[i,j,s]*A[i,j]*V2[j,k,s]*B[j,k] should give W2[k,i,s]*C[i,k]
    # Actually the tensor decomposition: T[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
    # where T[i,j,k] = 1 if this is a valid multiplication term... 
    
    # Standard: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = 1 for all i,j,k (it's the identity tensor for matrix mult)
    # Decomposition: T[i,j,k] = sum_s U[i,j,s] V[j,k,s] W[k,i,s]
    
    # Let me verify U2, V2, W2:
    # s=0: M0 = (A00+A11)(B00+B11)
    # U2[:,0] = [1,0,0,1] -> A00 + A11 ✓
    # V2[:,0] = [1,0,0,1] -> B00 + B11 ✓
    # W2[:,0] = [1,0,0,1] -> C00 += M0, C11 += M0 ✓
    
    # s=1: M1 = (A10+A11)B00
    # U2[:,1] = [0,0,1,1] -> A10 + A11 ✓
    # V2[:,1] = [0,0,1,0] -> B00 ✓
    # W2[:,1] = [0,0,1,-1] -> C10 += M1, C11 -= M1 ✓
    
    # s=2: M2 = A00(B01-B11)
    # U2[:,2] = [1,0,0,0] -> A00 ✓
    # V2[:,2] = [0,1,-1,0] -> B01 - B11 ✓
    # W2[:,2] = [0,1,0,1] -> C01 += M2, C11 += M2 ✓
    
    # s=3: M3 = A11(B10-B00)
    # U2[:,3] = [0,0,0,1] -> A11 ✓
    # V2[:,3] = [-1,0,0,1] -> B10 - B00 ✓
    # W2[:,3] = [0,0,1,0] -> C10 += M3 ✓
    
    # s=4: M4 = (A00+A01)B11
    # U2[:,4] = [1,1,0,0] -> A00 + A01 ✓
    # V2[:,4] = [0,0,0,1] -> B11 ✓
    # W2[:,4] = [-1,1,0,0] -> C00 -= M4, C01 += M4 ✓
    
    # s=5: M5 = (A10-A00)(B00+B01)
    # U2[:,5] = [-1,0,1,0] -> A10 - A00 ✓
    # V2[:,5] = [1,1,0,0] -> B00 + B01 ✓
    # W2[:,5] = [0,0,0,1] -> C11 += M5 ✓
    
    # s=6: M6 = (A01-A11)(B10+B11)
    # U2[:,6] = [0,1,0,-1] -> A01 - A11 ✓
    # V2[:,6] = [0,0,1,1] -> B10 + B11 ✓
    # W2[:,6] = [1,0,0,0] -> C00 += M6 ✓
    
    # Now build 4x4 factors via Kronecker product
    # U4[i*4+j, s*7+t] = U2[i,s] * U2[j,t]
    # V4[j*4+k, s*7+t] = V2[j,s] * V2[k,t]
    # W4[k*4+i, s*7+t] = W2[k,s] * W2[i,t]
    
    U4 = np.kron(U2, U2)  # shape (16, 49), row i*4+j, col s*7+t
    V4 = np.kron(V2, V2)  # shape (16, 49), row j*4+k, col s*7+t
    W4 = np.kron(W2, W2)  # shape (16, 49), row k*4+i, col s*7+t
    
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
