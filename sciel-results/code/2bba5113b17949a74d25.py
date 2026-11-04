import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Strassen decomposition for 2x2 matrix multiplication
    # Rank 7
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # Strassen factors (one standard form)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # U factors (Input A)
    # A11=0, A12=1, A21=2, A22=3 (row-major 2x2)
    # M1: A11+A22 -> U[0,0]=1, U[3,0]=1
    # M2: A21+A22 -> U[2,1]=1, U[3,1]=1
    # M3: A11       -> U[0,2]=1
    # M4: A22       -> U[3,3]=1
    # M5: A11+A12   -> U[0,4]=1, U[1,4]=1
    # M6: A21-A11   -> U[2,5]=1, U[0,5]=-1
    # M7: A12-A22   -> U[1,6]=1, U[3,6]=-1
    
    U2[0, 0] = 1; U2[3, 0] = 1
    U2[2, 1] = 1; U2[3, 1] = 1
    U2[0, 2] = 1
    U2[3, 3] = 1
    U2[0, 4] = 1; U2[1, 4] = 1
    U2[2, 5] = 1; U2[0, 5] = -1
    U2[1, 6] = 1; U2[3, 6] = -1
    
    # V factors (Input B)
    # B11=0, B12=1, B21=2, B22=3
    # M1: B11+B22 -> V[0,0]=1, V[3,0]=1
    # M2: B11       -> V[0,1]=1
    # M3: B12-B22   -> V[1,2]=1, V[3,2]=-1
    # M4: B21-B11   -> V[2,3]=1, V[0,3]=-1
    # M5: B22       -> V[3,4]=1
    # M6: B11+B12   -> V[0,5]=1, V[1,5]=1
    # M7: B21+B22   -> V[2,6]=1, V[3,6]=1
    
    V2[0, 0] = 1; V2[3, 0] = 1
    V2[0, 1] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    V2[2, 3] = 1; V2[0, 3] = -1
    V2[3, 4] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # W factors (Output C)
    # C11=0, C12=1, C21=2, C22=3
    # C11 = M1 + M4 - M5 + M7 -> W[0,0]=1, W[0,3]=1, W[0,4]=-1, W[0,6]=1
    # C12 = M3 + M5           -> W[1,2]=1, W[1,4]=1
    # C21 = M2 + M4           -> W[2,1]=1, W[2,3]=1
    # C22 = M1 - M2 + M3 + M6 -> W[3,0]=1, W[3,1]=-1, W[3,2]=1, W[3,5]=1
    
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    W2[1, 2] = 1; W2[1, 4] = 1
    W2[2, 1] = 1; W2[2, 3] = 1
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1
    
    # Kronecker product to get 4x4 decomposition
    # U4 = U2 (x) U2
    # V4 = V2 (x) V2
    # W4 = W2 (x) W2
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Convert to lists of lists with integer/half-integer entries
    # The entries are integers, so we can cast directly.
    
    def to_list(mat):
        return [[int(x) if x == int(x) else float(x) for x in row] for row in mat]
    
    U_list = to_list(U4)
    V_list = to_list(V4)
    W_list = to_list(W4)
    
    # Verify time
    if time.time() > deadline:
        # Should not happen, but safety
        pass
        
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
