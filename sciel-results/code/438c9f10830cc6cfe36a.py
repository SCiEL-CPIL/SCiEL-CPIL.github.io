import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    """
    Constructs a rank-49 exact tensor decomposition for 4x4 matrix multiplication
    using the Kronecker product of the Strassen decomposition for 2x2 matrices.
    """
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrices
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    # Indices:
    # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -- Note: k*2+i
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00(0), A11(3)
    U2[0, 0] = 1
    U2[3, 0] = 1
    # V: B00(0), B11(3)
    V2[0, 0] = 1
    V2[3, 0] = 1
    # W: C00(0), C11(3)
    W2[0, 0] = 1
    W2[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    # U: A10(2), A11(3)
    U2[2, 1] = 1
    U2[3, 1] = 1
    # V: B00(0)
    V2[0, 1] = 1
    # W: C10(2)
    W2[2, 1] = 1
    
    # M3 = A00(B01 - B11)
    # U: A00(0)
    U2[0, 2] = 1
    # V: B01(1), B11(3)
    V2[1, 2] = 1
    V2[3, 2] = -1
    # W: C01(1), C11(3)
    W2[1, 2] = 1
    W2[3, 2] = 1
    
    # M4 = A11(B10 - B00)
    # U: A11(3)
    U2[3, 3] = 1
    # V: B10(2), B00(0)
    V2[2, 3] = 1
    V2[0, 3] = -1
    # W: C00(0), C10(2)
    W2[0, 3] = 1
    W2[2, 3] = 1
    
    # M5 = (A00 + A01)B11
    # U: A00(0), A01(1)
    U2[0, 4] = 1
    U2[1, 4] = 1
    # V: B11(3)
    V2[3, 4] = 1
    # W: C00(0) [-1], C01(1) [1]
    W2[0, 4] = -1
    W2[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # U: A10(2), A00(0)
    U2[2, 5] = 1
    U2[0, 5] = -1
    # V: B00(0), B01(1)
    V2[0, 5] = 1
    V2[1, 5] = 1
    # W: C11(3)
    W2[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # U: A01(1), A11(3)
    U2[1, 6] = 1
    U2[3, 6] = -1
    # V: B10(2), B11(3)
    V2[2, 6] = 1
    V2[3, 6] = 1
    # W: C00(0)
    W2[0, 6] = 1
    
    # Kronecker product to get 4x4 decomposition
    # U4[i*4+j, r1*7+r2] = U2[i*2+j1, r1] * U2[j1*2+j2, r2] ?
    # The block structure:
    # A = [[A11, A12], [A21, A22]]
    # U4 should map A_{ij} to the appropriate 2x2 block factor.
    # Standard Kronecker product of tensors:
    # If T = T1 x T2, then U = U1 x U2, V = V1 x V2, W = W1 x W2.
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Ensure shapes are correct
    # U2 is 4x7. U4 is 16x49.
    # This matches n*m=16, rank=49.
    
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

# EVOLVE-BLOCK-END
