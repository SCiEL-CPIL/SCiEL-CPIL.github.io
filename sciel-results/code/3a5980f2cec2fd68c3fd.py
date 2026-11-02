def construct():
    import numpy as np
    import time
    start = time.time()
    
    n, m, p = 2, 4, 6
    
    # Strassen steps for 2x2 * 2x2 -> 2x2
    # Each step: (A_coeffs, B_coeffs, W_coeffs)
    # A_coeffs: dict (i, j) -> coeff, for X[i,j]
    # B_coeffs: dict (j, k) -> coeff, for Y[j,k]  
    # W_coeffs: dict (i, k) -> coeff, for Z[i,k]
    
    strassen_steps = [
        # M1 = (X00+X11)*(Y00+Y11)
        {
            'A': {(0,0):1, (1,1):1},
            'B': {(0,0):1, (1,1):1},
            'W': {(0,0):1, (1,1):-1}  # Z00 += M1, Z11 -= M1
        },
        # M2 = (X10+X11)*Y00
        {
            'A': {(1,0):1, (1,1):1},
            'B': {(0,0):1},
            'W': {(1,0):1, (1,1):-1}  # Z10 += M2, Z11 -= M2
        },
        # M3 = X00*(Y01-Y11)
        {
            'A': {(0,0):1},
            'B': {(0,1):1, (1,1):-1},
            'W': {(0,1):1, (1,1):1}  # Z01 += M3, Z11 += M3
        },
        # M4 = X11*(Y10-Y00)
        {
            'A': {(1,1):1},
            'B': {(1,0):1, (0,0):-1},
            'W': {(0,0):1, (1,0):1}  # Z00 += M4, Z10 += M4
        },
        # M5 = (X00+X01)*Y11
        {
            'A': {(0,0):1, (0,1):1},
            'B': {(1,1):1},
            'W': {(0,0):-1, (0,1):1}  # Z00 -= M5, Z01 += M5
        },
        # M6 = (X10-X00)*(Y00+Y01)
        {
            'A': {(1,0):1, (0,0):-1},
            'B': {(0,0):1, (0,1):1},
            'W': {(1,1):1}  # Z11 += M6
        },
        # M7 = (X01-X11)*(Y10+Y11)
        {
            'A': {(0,1):1, (1,1):-1},
            'B': {(1,0):1, (1,1):1},
            'W': {(0,0):1, (1,1):1}  # Z00 += M7, Z11 += M7
        },
    ]
    
    # Verify Strassen: for each Z[i,k], sum of contributions should give correct result
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Now build the full 2x4x6 decomposition
    # Split A into A1=A[:,0:2], A2=A[:,2:4]
    # Split B into B1=B[0:2,:], B2=B[2:4,:]
    # For each column pair c in 0,1,2:
    #   C[:, 2c:2c+2] = A1*B1[:,2c:2c+2] + A2*B2[:,2c:2c+2]
    
    U = np.zeros((n*m, 42))
    V = np.zeros((m*p, 42))
    W = np.zeros((p*n, 42))
    
    col = 0
    for c in range(3):  # column pair
        k_base = 2 * c
        for block in range(2):  # 0: A1*B1, 1: A2*B2
            # X = A[:, 2*block:2*block+2] (2x2)
            # Y = B[2*block:2*block+2, k_base:k_base+2] (2x2)
            # Z = 2x2 result at C[:, k_base:k_base+2]
            
            for step in strassen_steps:
                A_coeffs = step['A']
                B_coeffs = step['B']
                W_coeffs = step['W']
                
                # Update U: A[i, 2*block + j_local]
                for (i, j_local), coeff in A_coeffs.items():
                    j_global = 2 * block + j_local
                    U[i * m + j_global, col] += coeff
                
                # Update V: B[2*block + j_local, k_base + k_local]
                for (j_local, k_local), coeff in B_coeffs.items():
                    j_global = 2 * block + j_local
                    k_global = k_base + k_local
                    V[j_global * p + k_global, col] += coeff
                
                # Update W: C[i, k_base + k_local]
                for (i, k_local), coeff in W_coeffs.items():
                    k_global = k_base + k_local
                    W[k_global * n + i, col] += coeff
                
                col += 1
    
    # Verify
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    if not np.allclose(T, 1.0, atol=1e-10):
        # Find where it fails
        diff = T - 1.0
        max_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        raise ValueError(f"Verification failed at {max_idx}, value={T[max_idx]}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
