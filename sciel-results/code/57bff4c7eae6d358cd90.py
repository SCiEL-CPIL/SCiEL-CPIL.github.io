import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # The tensor M_{2,4,4} is the direct sum of two M_{2,4,2} tensors.
    # M_{2,4,2} is the direct sum of two M_{2,2,2} tensors.
    # So M_{2,4,4} is the direct sum of four M_{2,2,2} tensors.
    # Rank = 4 * 7 = 28.
    
    # We construct the Strassen decomposition for M_{2,2,2}.
    # U, V, W are 4x7 matrices.
    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10 + A11) * B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A00 * (B01 - B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A11 * (B10 - B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00 + A01) * B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    U_222, V_222, W_222 = get_strassen_2x2x2()
    
    # We need to assemble these into the 2x4x4 structure.
    # U shape: (8, 28)
    # V shape: (16, 28)
    # W shape: (8, 28)
    
    U_full = np.zeros((8, 28))
    V_full = np.zeros((16, 28))
    W_full = np.zeros((8, 28))
    
    # Block 1: A1 * B1 contributes to C1
    # A1 is A[0:2, 0:2] -> U rows 0:4
    # B1 is B[0:2, 0:2] -> V rows 0:4
    # C1 is C[0:2, 0:2] -> W rows 0:4 (k*2+i, k in 0:2, i in 0:2)
    # Columns 0:7
    U_full[0:4, 0:7] = U_222
    V_full[0:4, 0:7] = V_222
    W_full[0:4, 0:7] = W_222
    
    # Block 2: A2 * B3 contributes to C1
    # A2 is A[0:2, 2:4] -> U rows 4:8
    # B3 is B[2:4, 0:2] -> V rows 8:12
    # C1 is C[0:2, 0:2] -> W rows 0:4
    # Columns 7:14
    U_full[4:8, 7:14] = U_222
    V_full[8:12, 7:14] = V_222
    W_full[0:4, 7:14] = W_222
    
    # Block 3: A1 * B2 contributes to C2
    # A1 is A[0:2, 0:2] -> U rows 0:4
    # B2 is B[0:2, 2:4] -> V rows 4:8
    # C2 is C[0:2, 2:4] -> W rows 4:8 (k in 2:4, i in 0:2 -> k*2+i in 4:8)
    # Columns 14:21
    U_full[0:4, 14:21] = U_222
    V_full[4:8, 14:21] = V_222
    W_full[4:8, 14:21] = W_222
    
    # Block 4: A2 * B4 contributes to C2
    # A2 is A[0:2, 2:4] -> U rows 4:8
    # B4 is B[2:4, 2:4] -> V rows 12:16
    # C2 is C[0:2, 2:4] -> W rows 4:8
    # Columns 21:28
    U_full[4:8, 21:28] = U_222
    V_full[12:16, 21:28] = V_222
    W_full[4:8, 21:28] = W_222
    
    # Verification
    T_reconstructed = np.zeros((2, 4, 4))
    for i in range(2):
        for j in range(4):
            for k in range(4):
                val = 0.0
                for r in range(28):
                    val += U_full[i*4+j, r] * V_full[j*4+k, r] * W_full[k*2+i, r]
                T_reconstructed[i, j, k] = val
                
    if not np.allclose(T_reconstructed, 1.0):
        # Fallback to schoolbook rank 32
        rank = 32
        U_fb = np.zeros((8, rank))
        V_fb = np.zeros((16, rank))
        W_fb = np.zeros((8, rank))
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j, col] = 1
                    V_fb[j*4+k, col] = 1
                    W_fb[k*2+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
