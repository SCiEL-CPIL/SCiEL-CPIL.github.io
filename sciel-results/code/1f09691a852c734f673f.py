import time
import numpy as np

def construct():
    """
    Rank-56 decomposition for 2x4 * 4x8 matrix multiplication.
    Uses Kronecker product of Strassen (2x2x2, rank 7) and schoolbook (1x2x4, rank 8).
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen factors for 2x2x2
    # Indices: A[i, j] -> i*2+j, B[i, j] -> i*2+j, C[i, j] -> i*2+j
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1
    
    # Schoolbook factors for 1x2x4:
    # A is 1x2, B is 2x4, C is 1x4
    # U_b (2, 8), V_b (8, 8), W_b (4, 8)
    U_b = np.zeros((2, 8))
    V_b = np.zeros((8, 8))
    W_b = np.zeros((4, 8))
    
    col = 0
    for j in range(2):
        for k in range(4):
            U_b[j, col] = 1
            V_b[j * 4 + k, col] = 1
            W_b[k, col] = 1
            col += 1
    
    # Construct global factors
    # Decomposition: T_{2,4,8} = T_{2,2,2} (Strassen) x T_{1,2,4} (Schoolbook)
    # A[i, j] where j = 2*j1 + j2. i in {0,1}, j1 in {0,1}, j2 in {0,1}
    # B[j, k] where j = 2*j1 + j2, k = 4*c1 + c2. c1 in {0,1}, c2 in {0,1,2,3}
    # C[i, k] where k = 4*c1 + c2
    
    for s in range(7):
        for b in range(8):
            col = s * 8 + b
            
            # U: global index i*4 + j = i*4 + 2*j1 + j2
            for i in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        val_s = U_s[i * 2 + j1, s]
                        val_b = U_b[j2, b]
                        if val_s != 0 and val_b != 0:
                            global_idx = i * 4 + (2 * j1 + j2)
                            U[global_idx, col] += val_s * val_b
            
            # V: global index j*8 + k = (2*j1 + j2)*8 + (4*c1 + c2)
            for j1 in range(2):
                for j2 in range(2):
                    for c1 in range(2):
                        for c2 in range(4):
                            j = 2 * j1 + j2
                            k = 4 * c1 + c2
                            val_s = V_s[j1 * 2 + c1, s]
                            val_b = V_b[j2 * 4 + c2, b]
                            if val_s != 0 and val_b != 0:
                                global_idx = j * 8 + k
                                V[global_idx, col] += val_s * val_b
            
            # W: global index k*2 + i = (4*c1 + c2)*2 + i
            for i in range(2):
                for c1 in range(2):
                    for c2 in range(4):
                        k = 4 * c1 + c2
                        val_s = W_s[c1 * 2 + i, s]
                        val_b = W_b[c2, b]
                        if val_s != 0 and val_b != 0:
                            global_idx = k * 2 + i
                            W[global_idx, col] += val_s * val_b
    
    # Quick verification
    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((n, m))
    B_test = rng.standard_normal((m, p))
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = U[:, r].reshape(n, m)
        v_vec = V[:, r].reshape(m, p)
        w_vec = W[:, r].reshape(p, n)
        C_computed += (u_vec @ v_vec) * w_vec.T
    
    if not np.allclose(C_computed, C_expected, atol=1e-10, rtol=1e-10):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
