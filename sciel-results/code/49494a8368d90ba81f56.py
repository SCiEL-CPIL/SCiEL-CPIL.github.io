import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 7 decomposition for 2x2x2 matrix multiplication
    # Indices for 2x2 matrices: 1: (1,1), 2: (1,2), 3: (2,1), 4: (2,2)
    # u, v, w are 4x7 matrices
    
    u = np.zeros((4, 7))
    v = np.zeros((4, 7))
    w = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22) -> C11, C22
    u[:, 0] = [1, 0, 0, 1]
    v[:, 0] = [1, 0, 0, 1]
    w[:, 0] = [1, 0, 0, 1]
    
    # M2 = (A21+A22)B11 -> C21, -C22
    u[:, 1] = [0, 0, 1, 1]
    v[:, 1] = [1, 0, 0, 0]
    w[:, 1] = [0, 0, 1, -1]
    
    # M3 = A11(B12-B22) -> C12, C22
    u[:, 2] = [1, 0, 0, 0]
    v[:, 2] = [0, 1, 0, -1]
    w[:, 2] = [0, 1, 0, 1]
    
    # M4 = A22(B21-B11) -> C11, C21
    u[:, 3] = [0, 0, 0, 1]
    v[:, 3] = [0, 0, 1, -1]
    w[:, 3] = [1, 0, 1, 0]
    
    # M5 = (A11+A12)B22 -> C12, -C11
    u[:, 4] = [1, 1, 0, 0]
    v[:, 4] = [0, 0, 0, 1]
    w[:, 4] = [-1, 1, 0, 0]
    
    # M6 = (A21-A11)(B11+B12) -> C22
    u[:, 5] = [-1, 0, 1, 0]
    v[:, 5] = [1, 1, 0, 0]
    w[:, 5] = [0, 0, 0, 1]
    
    # M7 = (A12-A22)(B21+B22) -> C11
    u[:, 6] = [0, 1, 0, -1]
    v[:, 6] = [0, 0, 1, 1]
    w[:, 6] = [1, 0, 0, 0]
    
    # Kronecker product to get 4x4x4 decomposition
    # Rank 49
    # U_444[i1*2+i2, r1*7+r2] = u[i1, r1] * u[i2, r2]
    # Note: i1, i2 in 0..1. r1, r2 in 0..6.
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    U_444[i, r] = u[i1, r1] * u[i2, r2]
                    V_444[i, r] = v[i1, r1] * v[i2, r2]
                    W_444[i, r] = w[i1, r1] * w[i2, r2]
    
    # Now we have rank 49 for 4x4x4 (indices 0..3 for i,j,k)
    # We need to extend to 4x5x5.
    # n=4, m=5, p=5.
    # U size 4x51, V size 5x51, W size 5x51.
    
    rank = 51
    U = np.zeros((n, rank))
    V = np.zeros((m, rank))
    W = np.zeros((p, rank))
    
    # Copy 4x4x4 part
    # U_444 is 4x49? No, 16x49.
    # Wait, U_444 constructed above is for the flattened 4x4 matrix?
    # No, the tensor indices are i, j, k.
    # In the 4x4x4 case, i is 0..3.
    # My U_444 construction above:
    # i = i1*2 + i2. This is the row index of A (4x4).
    # So U_444 is 4x49?
    # No, in the loop:
    # for i1 in range(2): for i2 in range(2): i = i1*2 + i2.
    # So i is 0..3.
    # So U_444 is 4x49.
    # But I initialized U_444 as (16, 49).
    # Let's re-verify the Kronecker product logic.
    
    # The tensor T[i,j,k] for 4x4x4.
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
    # If we use Kronecker, T[i,j,k] = T2[i1,j1,k1] T2[i2,j2,k2].
    # T2 is 2x2x2.
    # U2[i1, r1], V2[j1, r1], W2[k1, r1].
    # Then U4[i, r1*7+r2] = U2[i1, r1] U2[i2, r2].
    # This is correct.
    # So U_444 should be 4x49.
    
    U_444 = np.zeros((4, 49))
    V_444 = np.zeros((4, 49))
    W_444 = np.zeros((4, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    U_444[i, r] = u[i1, r1] * u[i2, r2]
                    V_444[i, r] = v[i1, r1] * v[i2, r2]
                    W_444[i, r] = w[i1, r1] * w[i2, r2]
    
    # Fill U, V, W for the first 49 columns
    U[:4, :49] = U_444
    V[:4, :49] = V_444
    W[:4, :49] = W_444
    
    # Add term 1: j=5 (index 4)
    # U1 = 1, V1 = e5, W1 = 1
    U[:, 49] = 1.0
    V[4, 49] = 1.0
    W[:, 49] = 1.0
    
    # Add term 2: k=5 (index 4), j<=4
    # U2 = 1, V2 = [1,1,1,1,0], W2 = e5
    U[:, 50] = 1.0
    V[:4, 50] = 1.0
    V[4, 50] = 0.0
    W[4, 50] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
