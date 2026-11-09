import numpy as np

def construct():
    n, m, p = 3, 4, 4
    # Ranks
    r1 = 27 # 3x3x3
    r2 = 9  # 3x3x1
    r3 = 1  # 3x1x3
    r4 = 3  # 3x1x1
    R = r1 + r2 + r3 + r4
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    # 1. 3x3x3
    # A[0:3, 0:3], B[0:3, 0:3] -> C[0:3, 0:3]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*4 + j, col] = 1
                V[j*4 + k, col] = 1
                W[k*3 + i, col] = 1
                col += 1
                
    # 2. 3x3x1
    # A[0:3, 0:3], B[0:3, 3] -> C[0:3, 3]
    for i in range(3):
        for j in range(3):
            # B index is j*4 + 3
            # C index is 3*3 + i = 9 + i
            U[i*4 + j, col] = 1
            V[j*4 + 3, col] = 1
            W[9 + i, col] = 1
            col += 1
            
    # 3. 3x1x3
    # A[0:3, 3], B[3, 0:3] -> C[0:3, 0:3]
    # Outer product: C[i,k] += A[i,3] * B[3,k]
    # Rank 1.
    # U[i*4+3] = 1 for all i? No.
    # Term: P = A[i,3] * B[3,k].
    # This is not a single product P.
    # C[i,k] = A[i,3] B[3,k].
    # We need to decompose this tensor.
    # T_{i,k} = u_i v_k.
    # Rank 1 decomposition:
    # U_{i, col} = 1? No.
    # If we use 1 rank:
    # U_{i, col} must depend on i?
    # Let's check: Sum_t U_{i,t} V_{k,t} W_{t, i,k}.
    # If t=1: U_{i,1} V_{k,1} W_{1, i,k} = A[i,3] B[3,k].
    # We can set U_{i,1} = 1, V_{k,1} = 1, W_{1, i,k} = A[i,3] B[3,k]?
    # No, W must be constant.
    # The tensor is $T_{i,k} = 1 \cdot 1$? No, the inputs are $A_{i,3}$ and $B_{3,k}$.
    # The bilinear form is $\sum_{i,k} A_{i,3} B_{3,k} C_{i,k}$.
    # This is equivalent to $\sum_{i,k} (A_{i,3}) (B_{3,k}) C_{i,k}$.
    # This is a rank 1 tensor in the space of (i,k) indices?
    # Yes, $T_{i,k} = 1$.
    # So $U_{i, col} = 1$ for all $i \in 0..2$.
    # $V_{k, col} = 1$ for all $k \in 0..2$. (Note: B index is $3*4+k = 12+k$).
    # $W_{col, k*3+i} = 1$.
    
    U[0*4+3, col] = 1
    U[1*4+3, col] = 1
    U[2*4+3, col] = 1
    V[3*4+0, col] = 1
    V[3*4+1, col] = 1
    V[3*4+2, col] = 1
    W[0*3+0, col] = 1
    W[1*3+0, col] = 1
    W[2*3+0, col] = 1
    W[0*3+1, col] = 1
    W[1*3+1, col] = 1
    W[2*3+1, col] = 1
    W[0*3+2, col] = 1
    W[1*3+2, col] = 1
    W[2*3+2, col] = 1
    col += 1
    
    # 4. 3x1x1
    # A[0:3, 3], B[3, 3] -> C[0:3, 3]
    # C[i, 3] += A[i, 3] * B[3, 3]
    # Rank 3.
    for i in range(3):
        U[i*4+3, col] = 1
        V[3*4+3, col] = 1
        W[9+i, col] = 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
