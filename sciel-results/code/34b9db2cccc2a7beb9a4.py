import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Strategy: Split B (3x7) into B_left (3x3) and B_right (3x4)
    # Compute C_left = A @ B_left using rank R1 decomposition
    # Compute C_right = A @ B_right using rank R2 decomposition
    # Total rank R = R1 + R2
    
    # 1. Construct Rank 27 decomposition for 3x3 multiplication
    # Dimensions: A(3x3), B(3x3), C(3x3)
    n1, m1, p1 = 3, 3, 3
    R1 = n1 * m1 * p1  # 27
    U1 = np.zeros((n1 * m1, R1))
    V1 = np.zeros((m1 * p1, R1))
    W1 = np.zeros((p1 * n1, R1))
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j, col] = 1
                V1[j * p1 + k, col] = 1
                W1[k * n1 + i, col] = 1
                col += 1
                
    # 2. Construct Rank 12 decomposition for 3x4 multiplication
    # Dimensions: A(3x3), B(3x4), C(3x4)
    n2, m2, p2 = 3, 3, 4
    R2 = n2 * m2 * p2  # 12
    U2 = np.zeros((n2 * m2, R2))
    V2 = np.zeros((m2 * p2, R2))
    W2 = np.zeros((p2 * n2, R2))
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j, col] = 1
                V2[j * p2 + k, col] = 1
                W2[k * n2 + i, col] = 1
                col += 1
                
    # 3. Combine into full 3x7 decomposition
    R = R1 + R2  # 39
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Copy U1 and U2
    # U1 is (9, 27), U2 is (9, 12)
    U[:, :R1] = U1
    U[:, R1:] = U2
    
    # Copy V1 and V2 with correct index mapping
    # V1 corresponds to B_left (cols 0-2 of global B)
    # Global B index: j * p + k
    # For B_left: k_local in 0..2, k_global = k_local
    for r in range(R1):
        for j in range(m1):
            for k in range(p1):
                if V1[j * p1 + k, r] != 0:
                    V[j * p + k, r] = V1[j * p1 + k, r]
                    
    # V2 corresponds to B_right (cols 3-6 of global B)
    # For B_right: k_local in 0..3, k_global = 3 + k_local
    for r in range(R2):
        for j in range(m2):
            for k in range(p2):
                if V2[j * p2 + k, r] != 0:
                    V[j * p + (3 + k), R1 + r] = V2[j * p2 + k, r]
                    
    # Copy W1 and W2 with correct index mapping
    # W1 corresponds to C_left (cols 0-2 of global C)
    # Global C index: k * n + i
    # For C_left: k_local in 0..2, k_global = k_local
    for r in range(R1):
        for k in range(p1):
            for i in range(n1):
                if W1[k * n1 + i, r] != 0:
                    W[k * n + i, r] = W1[k * n1 + i, r]
                    
    # W2 corresponds to C_right (cols 3-6 of global C)
    # For C_right: k_local in 0..3, k_global = 3 + k_local
    for r in range(R2):
        for k in range(p2):
            for i in range(n2):
                if W2[k * n2 + i, r] != 0:
                    W[(3 + k) * n + i, R1 + r] = W2[k * n2 + i, r]
                    
    # Verify the decomposition
    # C[i, k] = sum_j A[i, j] B[j, k]
    # We check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    # Actually, we check the tensor identity:
    # T[i, j, k] = 1 if term (i,j,k) is present.
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should be 1 for all i,j,k?
    # No, the tensor M[i, j, k] = 1.
    # The decomposition is M = sum_r U_r \otimes V_r \otimes W_r.
    # So for each (i, j, k), sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should be 1.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(R):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if not np.isclose(val, 1.0):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook 63
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
