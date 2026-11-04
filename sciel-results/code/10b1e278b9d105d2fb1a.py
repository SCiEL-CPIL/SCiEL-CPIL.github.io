import sys
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Build the matrix multiplication tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i * m + j
                idx2 = j * p + k
                idx3 = k * n + i
                T[idx1, idx2, idx3] = 1.0

    def verify(U, V, W):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        T_recon = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-8)

    # Strassen 2x2x2 decomposition (rank 7)
    # Returns U, V, W blocks of shape (4, 7), (4, 7), (4, 7)
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    def strassen_2x2x2():
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
        W_s[2, 1] = 1  # C10 = M2
        
        # M3 = A00(B01-B11)
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1  # C01 = M3
        
        # M4 = A11(B10-B00)
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = -1; W_s[2, 3] = 1 # C00 -= M4, C10 += M4? 
        # Wait, standard Strassen:
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        # Let's re-verify W signs.
        # M1: C00 += 1, C11 += 1. W[0,0]=1, W[3,0]=1. Correct.
        # M2: C10 += 1. W[2,1]=1. Correct.
        # M3: C01 += 1. W[1,2]=1. Correct.
        # M4: C10 += 1, C00 -= 1. W[2,3]=1, W[0,3]=-1. Correct.
        # M5: (A00+A10)B11. C01 += 1, C00 -= 1.
        U_s[0, 4] = 1; U_s[2, 4] = 1
        V_s[3, 4] = 1
        W_s[1, 4] = 1; W_s[0, 4] = -1
        # M6: (A10-A00)(B00+B01). C11 -= 1.
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = -1
        # M7: (A01-A11)(B10+B11). C00 += 1.
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1

        return U_s, V_s, W_s

    # Build 2x2x6 block using 3 Strassen blocks
    # A is 2x2, B is 2x6 (split into 3 2x2 blocks)
    # U: (4, 21), V: (12, 21), W: (12, 21)
    def strassen_2x2x6():
        U_s, V_s, W_s = strassen_2x2x2()
        U_226 = np.zeros((4, 21))
        V_226 = np.zeros((12, 21))
        W_226 = np.zeros((12, 21))
        
        for block_idx in range(3):
            c_offset = block_idx * 7
            # U is same for all blocks
            U_226[:, c_offset:c_offset+7] = U_s
            
            # V: B is 2x6. Block idx corresponds to columns 2*block_idx, 2*block_idx+1
            # V indices in 2x2x2 are j*2+k. Here j in {0,1}, k in {0,1} local.
            # Global k = 2*block_idx + k_local.
            # V_global_index = j*6 + k_global? No, V size is mp=12 for this subproblem?
            # Wait, in the 2x2x6 subproblem, B is 2x6.
            # V indices are (j, k) with j in {0,1}, k in {0..5}.
            # In Strassen 2x2x2, V indices are (j, k_local) with k_local in {0,1}.
            # So V_226[j*6 + (2*block_idx + k_local), c] = V_s[j*2 + k_local, c_local]
            
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    v_idx = j * 6 + k_global
                    v_s_idx = j * 2 + k_local
                    V_226[v_idx, c_offset:c_offset+7] = V_s[v_s_idx, :]
            
            # W: C is 2x6. W indices are (k, i) with k in {0..5}, i in {0,1}.
            # In Strassen 2x2x2, W indices are (k_local, i).
            # W_226[(2*block_idx + k_local)*2 + i, c] = W_s[k_local*2 + i, c_local]
            for k_local in range(2):
                for i in range(2):
                    k_global = 2 * block_idx + k_local
                    w_idx = k_global * 2 + i
                    w_s_idx = k_local * 2 + i
                    W_226[w_idx, c_offset:c_offset+7] = W_s[w_s_idx, :]
        
        return U_226, V_226, W_226

    # Build 2x1x6 block (rank 1)
    def rank1_2x1x6():
        U_216 = np.zeros((2, 1))
        V_216 = np.zeros((6, 1))
        W_216 = np.zeros((12, 1))
        
        U_216[0, 0] = 1
        U_216[1, 0] = 1
        
        for k in range(6):
            V_216[k, 0] = 1
            
        for k in range(6):
            for i in range(2):
                W_216[k*2+i, 0] = 1
                
        return U_216, V_216, W_216

    # Assemble global decomposition
    # n=2, m=5, p=6
    # Blocks:
    # 1. j in {0,1}: 2x2x6. A cols 0,1. B rows 0,1.
    # 2. j in {2,3}: 2x2x6. A cols 2,3. B rows 2,3.
    # 3. j in {4}: 2x1x6. A col 4. B row 4.
    
    U_global = np.zeros((nm, 43))
    V_global = np.zeros((mp, 43))
    W_global = np.zeros((pn, 43))
    
    # Block 1
    U1, V1, W1 = strassen_2x2x6()
    # Map to global
    # U1 indices (i, j_local) -> global (i, j_global)
    # j_local 0,1 -> j_global 0,1
    for i in range(2):
        for j_local in range(2):
            j_global = j_local
            U_global[i*5 + j_global, 0:21] = U1[i*2 + j_local, :]
            
    # V1 indices (j_local, k) -> global (j_global, k)
    # j_local 0,1 -> j_global 0,1
    for j_local in range(2):
        for k in range(6):
            j_global = j_local
            V_global[j_global*6 + k, 0:21] = V1[j_local*6 + k, :]
            
    # W1 indices (k, i) -> global (k, i)
    # Same
    W_global[0:12, 0:21] = W1
    
    # Block 2
    U2, V2, W2 = strassen_2x2x6()
    # j_local 0,1 -> j_global 2,3
    for i in range(2):
        for j_local in range(2):
            j_global = 2 + j_local
            U_global[i*5 + j_global, 21:42] = U2[i*2 + j_local, :]
            
    for j_local in range(2):
        for k in range(6):
            j_global = 2 + j_local
            V_global[j_global*6 + k, 21:42] = V2[j_local*6 + k, :]
            
    W_global[0:12, 21:42] = W2
    
    # Block 3
    U3, V3, W3 = rank1_2x1x6()
    # j_local 0 -> j_global 4
    for i in range(2):
        U_global[i*5 + 4, 42] = U3[i, 0]
        
    for k in range(6):
        V_global[4*6 + k, 42] = V3[k, 0]
        
    W_global[0:12, 42] = W3[0:12, 0]
    
    if verify(U_global, V_global, W_global):
        return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx1 = i * m + j
                    idx2 = j * p + k
                    idx3 = k * n + i
                    U[idx1, col] = 1.0
                    V[idx2, col] = 1.0
                    W[idx3, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
