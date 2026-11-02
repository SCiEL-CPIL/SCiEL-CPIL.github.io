import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    
    rank = 18
    
    U = np.zeros((4, rank))
    V = np.zeros((12, rank))
    W = np.zeros((12, rank))
    
    # Strassen-like decomposition
    # For 2x2 matrices: C = AB
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block k in 0..2 (since p=6, we have 3 blocks of size 2)
    for k in range(p // 2):
        base = k * 6
        
        # U (A factor): A is 2x2, so 4 entries
        # A[0,0] -> index 0
        # A[0,1] -> index 1
        # A[1,0] -> index 2
        # A[1,1] -> index 3
        #
        # M1 uses A00, A11
        # M2 uses A10, A11
        # M3 uses A00
        # M4 uses A11
        # M5 uses A00, A10
        # M6 uses A11, A00
        # M7 uses A01, A11
        
        # U[:, 0]: A00 = 1
        U[0, base + 0] = 1
        # U[:, 1]: A11 = 1
        U[1, base + 1] = 1
        # U[:, 2]: A10 = 1
        U[2, base + 2] = 1
        # U[:, 3]: A00 = 1
        U[0, base + 3] = 1
        # U[:, 4]: A00 = 1, A10 = 1
        U[0, base + 4] = 1
        U[2, base + 4] = 1
        # U[:, 5]: A11 = 1, A00 = -1
        U[1, base + 5] = 1
        U[0, base + 5] = -1
    
        # V (B factor): B is 2x6, for block k, B is 2x2
        # B[m, col] -> index m * 6 + k * 2 + col (for col in 0,1)
        #
        # For block k, B entries are:
        # B[0, k*2] -> index 0 * 6 + k*2 + 0 = k*2
        # B[0, k*2+1] -> index 0 * 6 + k*2 + 1 = k*2+1
        # B[1, k*2] -> index 1 * 6 + k*2 + 0 = 6+k*2
        # B[1, k*2+1] -> index 1 * 6 + k*2 + 1 = 6+k*2+1
        
        # M1: (B00 + B11)
        V[k*2, base + 0] = 1
        V[6+k*2+1, base + 0] = 1
        # M2: B00
        V[k*2, base + 2] = 1
        # M3: (B10 - B11)
        V[6+k*2, base + 3] = 1
        V[6+k*2+1, base + 3] = -1
        # M4: (B01 - B00)
        V[k*2+1, base + 1] = 1
        V[k*2, base + 1] = -1
        # M5: B11
        V[6+k*2+1, base + 4] = 1
        # M6: (B00 + B10)
        V[k*2, base + 5] = 1
        V[6+k*2, base + 5] = 1
        # M7: (B01 + B11)
        V[k*2+1, base + 3] = 1
        V[6+k*2+1, base + 3] = 1
        
        # W (C factor): C is 2x6, for block k, C is 2x2
        # C[row, col] -> index row * 6 + k * 2 + col (for col in 0,1)
        #
        # C[0, k*2] -> index 0 * 6 + k*2 + 0 = k*2
        # C[0, k*2+1] -> index 0 * 6 + k*2 + 1 = k*2+1
        # C[1, k*2] -> index 1 * 6 + k*2 + 0 = 6+k*2
        # C[1, k*2+1] -> index 1 * 6 + k*2 + 1 = 6+k*2+1
        
        # C00 = M1 + M4 - M5 + M7
        W[k*2, base + 0] = 1   # M1
        W[k*2, base + 1] = 1   # M4
        W[k*2, base + 4] = -1  # -M5
        W[k*2, base + 3] = 1   # M7
        # C01 = M3 + M5
        W[k*2+1, base + 3] = 1  # M3
        W[k*2+1, base + 4] = 1  # M5
        # C10 = M2 + M4
        W[6+k*2, base + 2] = 1  # M2
        W[6+k*2, base + 1] = 1  # M4
        # C11 = M1 - M2 + M3 + M6
        W[6+k*2+1, base + 0] = 1   # M1
        W[6+k*2+1, base + 2] = -1  # -M2
        W[6+k*2+1, base + 3] = 1   # M3
        W[6+k*2+1, base + 5] = 1   # M6
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Verification
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for mid in range(m):
            for k in range(p):
                a_idx = i * m + mid
                b_idx = mid * p + k
                c_idx = i * p + k
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Reconstruct
    T_approx = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_approx += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    error = np.linalg.norm(T - T_approx)
    print(f"Rank: {rank}")
    print(f"Error: {error}")
    print(f"Valid: {np.allclose(T, T_approx, atol=1e-10)}")
