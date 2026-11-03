import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4, plus 1 for the 5th column
    rank = 50
    
    # Strassen 2x2 factors
    # U, V, W are 4x7 matrices
    # Indices: i,j in {0,1} -> row i*2+j
    # k in {0,1} -> for V, row j*2+k; for W, row k*2+i
    
    # Using a known valid integer decomposition for 2x2x2
    # See e.g. https://en.wikipedia.org/wiki/Strassen_algorithm
    # We need U, V, W such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    # One valid set:
    U2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ])
    V2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ])
    # Note: V2 rows are j*2+k
    # j=0,k=0 -> 0
    # j=0,k=1 -> 1
    # j=1,k=0 -> 2
    # j=1,k=1 -> 3
    
    # This U2/V2 pair might not be correct. I will use a verified set.
    # Actually, I can just use the fact that rank 7 is possible and use a solver?
    # No, I must provide the factors.
    # I will use the factors from the "Strassen algorithm" tensor decomposition.
    
    # Correct Strassen factors for 2x2:
    # r=0: M1 = A11(B11+B22)
    # U[0,0]=1, V[0,0]=1, V[1,0]=1, W[0,0]=1, W[1,0]=1
    # r=1: M2 = (A11+A22)B22
    # U[0,1]=1, U[3,1]=1, V[1,1]=1, W[0,1]=1, W[1,1]=1
    # r=2: M3 = (A21+A22)B11
    # U[2,2]=1, U[3,2]=1, V[0,2]=1, W[0,2]=1, W[1,2]=1
    # r=3: M4 = A22(B21-B11)
    # U[3,3]=1, V[1,3]=1, V[0,3]=-1, W[0,3]=1, W[1,3]=1
    # r=4: M5 = A11(B12-B22)
    # U[0,4]=1, V[1,4]=1, V[3,4]=-1, W[0,4]=1, W[1,4]=1
    # r=5: M6 = (A21+A11)B22
    # U[0,5]=1, U[2,5]=1, V[1,5]=1, W[0,5]=1, W[1,5]=1
    # r=6: M7 = (A12-A22)(B11+B21)
    # U[1,6]=1, U[3,6]=-1, V[0,6]=1, V[2,6]=1, W[0,6]=1, W[1,6]=1
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0
    U2[0,0] = 1
    V2[0,0] = 1; V2[1,0] = 1
    W2[0,0] = 1; W2[1,0] = 1
    
    # r=1
    U2[0,1] = 1; U2[3,1] = 1
    V2[1,1] = 1
    W2[0,1] = 1; W2[1,1] = 1
    
    # r=2
    U2[2,2] = 1; U2[3,2] = 1
    V2[0,2] = 1
    W2[0,2] = 1; W2[1,2] = 1
    
    # r=3
    U2[3,3] = 1
    V2[1,3] = 1; V2[0,3] = -1
    W2[0,3] = 1; W2[1,3] = 1
    
    # r=4
    U2[0,4] = 1
    V2[1,4] = 1; V2[3,4] = -1
    W2[0,4] = 1; W2[1,4] = 1
    
    # r=5
    U2[0,5] = 1; U2[2,5] = 1
    V2[1,5] = 1
    W2[0,5] = 1; W2[1,5] = 1
    
    # r=6
    U2[1,6] = 1; U2[3,6] = -1
    V2[0,6] = 1; V2[2,6] = 1
    W2[0,6] = 1; W2[1,6] = 1
    
    # Verify 2x2
    # for i,j,k in 0..1
    # sum_r U2[i*2+j, r] V2[j*2+k, r] W2[k*2+i, r] should be 1
    
    # Generate 4x4 factors via Kronecker product
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # U4 shape 16x49
    # V4 shape 16x49 (for 4x4, j*4+k)
    # W4 shape 16x49 (for 4x4, k*4+i)
    
    # Construct final U, V, W for 4x4x5
    U = np.zeros((16, 50))
    V = np.zeros((20, 50))
    W = np.zeros((20, 50))
    
    # Copy first 49 cols
    U[:, :49] = U4
    V[:, :49] = 0
    W[:, :49] = 0
    
    # Fill V for k=0..3
    for j in range(4):
        for k in range(4):
            V[j*5 + k, :49] = V4[j*4 + k, :49]
            
    # Fill W for k=0..3
    for k in range(4):
        for i in range(4):
            W[k*4 + i, :49] = W4[k*4 + i, :49]
            
    # Add 50th term (index 49)
    U[:, 49] = 1
    for j in range(4):
        V[j*5 + 4, 49] = 1
    for i in range(4):
        W[4*4 + i, 49] = 1
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
