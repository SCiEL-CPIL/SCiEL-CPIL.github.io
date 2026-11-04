"""Rewritten program: Strassen-based rank-65 decomposition for 4x4x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    start_time = time.time()
    deadline = start_time + 140

    # 1. Build Schoolbook Decomposition (Rank 80) as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # 2. Define Strassen 2x2x2 Factors (Rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # m1
    U2[0, 0] = 1; U2[2, 0] = -1
    V2[2, 0] = 1; V2[3, 0] = -1
    W2[0, 0] = 1; W2[2, 0] = 1
    
    # m2
    U2[0, 1] = 1; U2[1, 1] = 1
    V2[3, 1] = 1
    W2[1, 1] = 1
    
    # m3
    U2[0, 2] = 1
    V2[0, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # m4
    U2[3, 3] = 1
    V2[0, 3] = 1; V2[3, 3] = 1
    W2[0, 3] = 1; W2[3, 3] = 1
    
    # m5
    U2[1, 4] = 1; U2[3, 4] = -1
    V2[0, 4] = 1; V2[2, 4] = 1
    W2[0, 4] = -1; W2[3, 4] = 1
    
    # m6
    U2[0, 5] = 1; U2[2, 5] = 1
    V2[2, 5] = -1; V2[3, 5] = 1
    W2[2, 5] = 1
    
    # m7
    U2[0, 6] = -1; U2[2, 6] = 1
    V2[0, 6] = 1; V2[1, 6] = 1
    W2[0, 6] = 1

    # 3. Construct 4x4x4 Decomposition (Rank 49) via Kronecker product
    U4 = np.kron(U2, U2)  # 16x49
    V4 = np.kron(V2, V2)  # 16x49
    W4 = np.kron(W2, W2)  # 16x49

    # 4. Extend to 4x4x5 (Rank 65)
    U65 = np.zeros((16, 65))
    V65 = np.zeros((20, 65))
    W65 = np.zeros((20, 65))
    
    # Copy Strassen part (first 49 cols)
    U65[:, :49] = U4
    
    # Map V4 (16x49) to V65 (20x49)
    for j in range(4):
        for k in range(4):
            v4_idx = j * 4 + k
            v65_idx = j * 5 + k
            V65[v65_idx, :49] = V4[v4_idx, :49]
            
    # Map W4 (16x49) to W65 (20x49)
    for i in range(4):
        for k in range(4):
            w4_idx = i * 4 + k
            w65_idx = i * 5 + k
            W65[w65_idx, :49] = W4[w4_idx, :49]
            
    # Add Schoolbook part for 5th column (cols 49..64)
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            U65[i * 4 + j, col] = 1
            V65[j * 5 + 4, col] = 1
            W65[i * 5 + 4, col] = 1

    # 5. Verify Decomposition
    def verify(U, V, W):
        T = np.einsum('ar,br,cr->abc', U, V, W)
        Target = np.zeros((16, 20, 20))
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    a = i * 4 + j
                    b = j * 5 + k
                    c = i * 5 + k
                    Target[a, b, c] = 1.0
        return np.allclose(T, Target, atol=1e-9)

    if verify(U65, V65, W65):
        best_U, best_V, best_W = U65, V65, W65

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
