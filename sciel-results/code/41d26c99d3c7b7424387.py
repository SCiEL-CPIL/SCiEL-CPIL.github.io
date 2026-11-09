import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Target Rank: 117
    # 8 Strassen blocks (rank 7 each) + 61 Schoolbook terms (rank 1 each)
    R_strassen = 8 * 7
    R_schoolbook = 61
    total_rank = R_strassen + R_schoolbook
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Helper to get 2x2 Strassen factors
    # Returns U, V, W for 2x2 * 2x2 -> 2x2. Rank 7.
    # Local indices 0,1.
    def get_strassen_2x2_factors():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        U[0, 0] = 1; U[3, 0] = 1 
        V[0, 0] = 1; V[3, 0] = 1 
        W[0, 0] = 1; W[3, 0] = 1 
        
        # M2 = (A10+A11)B00 -> C10
        U[2, 1] = 1; U[3, 1] = 1 
        V[0, 1] = 1             
        W[2, 1] = 1             
        
        # M3 = A00(B01-B11) -> C01, C11
        U[0, 2] = 1             
        V[1, 2] = 1; V[3, 2] = -1 
        W[1, 2] = 1; W[3, 2] = 1  
        
        # M4 = A11(B10-B00) -> C00, C10
        U[3, 3] = 1             
        V[2, 3] = 1; V[0, 3] = -1 
        W[0, 3] = 1; W[2, 3] = 1  
        
        # M5 = (A00+A01)B11 -> C00, C01
        U[0, 4] = 1; U[1, 4] = 1 
        V[3, 4] = 1             
        W[0, 4] = -1; W[1, 4] = 1 
        
        # M6 = (A10-A11)(B01+B11) -> C11
        U[2, 5] = 1; U[3, 5] = -1 
        V[1, 5] = 1; V[3, 5] = 1  
        W[3, 5] = 1             
        
        # M7 = (A01-A11)(B00+B10) -> C00
        U[1, 6] = 1; U[3, 6] = -1 
        V[0, 6] = 1; V[2, 6] = 1  
        W[0, 6] = 1             
        
        return U, V, W

    U_s, V_s, W_s = get_strassen_2x2_factors()
    
    # Define the 8 cubes in the 4x4x4 core
    # Indices 0,1 and 2,3
    idx_sets = [
        [0, 1],
        [2, 3]
    ]
    
    col_offset = 0
    
    # Add Strassen blocks
    for I_set in idx_sets:
        for J_set in idx_sets:
            for K_set in idx_sets:
                # Map local indices (0,1) to global indices in I_set, J_set, K_set
                # Local i=0 -> I_set[0], Local i=1 -> I_set[1]
                # Local j=0 -> J_set[0], Local j=1 -> J_set[1]
                # Local k=0 -> K_set[0], Local k=1 -> K_set[1]
                
                for r in range(7):
                    # U_local[i, r] -> U_global[i_global * 5 + j_global, col]
                    for i_loc in range(2):
                        for j_loc in range(2):
                            val = U_s[i_loc * 2 + j_loc, r]
                            if val != 0:
                                i_glob = I_set[i_loc]
                                j_glob = J_set[j_loc]
                                U[i_glob * m + j_glob, col_offset + r] += val
                                
                    # V_local[j, r] -> V_global[j_global * 5 + k_global, col]
                    for j_loc in range(2):
                        for k_loc in range(2):
                            val = V_s[j_loc * 2 + k_loc, r]
                            if val != 0:
                                j_glob = J_set[j_loc]
                                k_glob = K_set[k_loc]
                                V[j_glob * p + k_glob, col_offset + r] += val
                                
                    # W_local[k, r] -> W_global[k_global * 5 + i_global, col]
                    for k_loc in range(2):
                        for i_loc in range(2):
                            val = W_s[k_loc * 2 + i_loc, r]
                            if val != 0:
                                k_glob = K_set[k_loc]
                                i_glob = I_set[i_loc]
                                W[k_glob * n + i_glob, col_offset + r] += val
                                
                col_offset += 7
    
    # Add Schoolbook terms for the remaining 61 entries
    # The remaining entries are those where i, j, or k is 4
    # Or more precisely, those NOT covered by the 8 cubes.
    # The 8 cubes cover all (i,j,k) where i,j,k in {0,1,2,3}.
    # So we iterate over all (i,j,k) and skip those in {0,1,2,3}^3.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this point is in the 4x4x4 core
                if i < 4 and j < 4 and k < 4:
                    continue
                
                # This point is in the "shell"
                # Add schoolbook term
                U[i * m + j, col_offset] = 1
                V[j * p + k, col_offset] = 1
                W[k * n + i, col_offset] = 1
                col_offset += 1
                
    # Verification
    # Compute tensor from factors
    def compute_tensor(U, V, W):
        R = U.shape[1]
        T_comp = np.zeros((n, m, p), dtype=np.float64)
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        for r in range(R):
            u_r = U_3d[:, :, r]
            v_r = V_3d[:, :, r]
            w_r = W_3d[:, :, r]
            T_comp += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
        return T_comp

    T_target = np.ones((n, m, p), dtype=np.float64)
    T_calc = compute_tensor(U, V, W)
    
    if not np.allclose(T_calc, T_target, atol=1e-9):
        # Fallback to schoolbook if construction failed
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
