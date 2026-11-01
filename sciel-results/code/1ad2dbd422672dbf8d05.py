import time
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Target Rank
    # 3 blocks of 2x2x2 (Strassen rank 7) + 6 blocks of 2x1x1 (Rank 2)
    # Wait, 2x2x6 is split into 3 2x2x2 blocks.
    # 2x1x6 is split into 6 2x1x1 blocks.
    rank_total = 3 * 7 + 6 * 2
    rank_total = 54
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Helper to get Strassen 2x2x2 matrices
    def get_strassen_222():
        # Returns U, V, W of shape (4, 7)
        # Indices: U: i*2+j, V: j*2+k, W: k*2+i
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: A00, A11 -> idx 0, 3
        U_s[0, 0] = 1; U_s[3, 0] = 1
        # V: B00, B11 -> idx 0, 3
        V_s[0, 0] = 1; V_s[3, 0] = 1
        # W: C00, C11 -> idx 0, 3
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        # Term 2: M2 = (A10+A11)B00
        # U: A10, A11 -> idx 2, 3
        U_s[2, 1] = 1; U_s[3, 1] = 1
        # V: B00 -> idx 0
        V_s[0, 1] = 1
        # W: C10 -> idx 2 (k=0, i=1)
        W_s[2, 1] = 1
        
        # Term 3: M3 = A00(B01-B11)
        # U: A00 -> idx 0
        U_s[0, 2] = 1
        # V: B01, B11 -> idx 1, 3
        V_s[1, 2] = 1; V_s[3, 2] = -1
        # W: C01, C11 -> idx 2, 3
        # C01: k=1, i=0 -> idx 2
        # C11: k=1, i=1 -> idx 3
        W_s[2, 2] = 1; W_s[3, 2] = -1
        
        # Term 4: M4 = A11(B10-B00)
        # U: A11 -> idx 3
        U_s[3, 3] = 1
        # V: B10, B00 -> idx 2, 0
        V_s[2, 3] = 1; V_s[0, 3] = -1
        # W: C00, C10 -> idx 0, 2
        W_s[0, 3] = 1; W_s[2, 3] = 1
        
        # Term 5: M5 = (A00+A10)B11
        # U: A00, A10 -> idx 0, 2
        U_s[0, 4] = 1; U_s[2, 4] = 1
        # V: B11 -> idx 3
        V_s[3, 4] = 1
        # W: C00, C10 -> idx 0, 2
        # C00 gets -M5?
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        # So W5 should be -1 for C00, 0 for C10?
        # Wait, M5 contributes to C01 and C11 in some conventions.
        # Let's check my C formulas again.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 + M6
        
        # If C01 = M3 + M5, then M5 contributes +1 to C01.
        # C01: k=1, i=0 -> idx 2.
        # So W5[2] = 1.
        # Does M5 contribute to C00?
        # In my formula C00 = ... - M5.
        # So W5[0] = -1.
        W_s[0, 4] = -1; W_s[2, 4] = 1
        
        # Term 6: M6 = (A01-A00)(B00+B10)
        # U: A01, A00 -> idx 1, 0
        U_s[1, 5] = 1; U_s[0, 5] = -1
        # V: B00, B10 -> idx 0, 2
        V_s[0, 5] = 1; V_s[2, 5] = 1
        # W: C11 -> idx 3
        # C11 = ... + M6
        W_s[3, 5] = 1
        
        # Term 7: M7 = (A11-A10)(B01+B11)
        # U: A11, A10 -> idx 3, 2
        U_s[3, 6] = 1; U_s[2, 6] = -1
        # V: B01, B11 -> idx 1, 3
        V_s[1, 6] = 1; V_s[3, 6] = 1
        # W: C00 -> idx 0
        # C00 = ... + M7
        W_s[0, 6] = 1
        
        return U_s, V_s, W_s

    # Helper to get 2x1x1 matrices
    def get_trivial_211():
        # Returns U, V, W of shape (2, 2), (1, 2), (2, 2)
        # U: i*1+0 = i
        # V: 0*1+0 = 0
        # W: k*2+i = i (since k=0)
        U_t = np.zeros((2, 2))
        V_t = np.zeros((1, 2))
        W_t = np.zeros((2, 2))
        
        # Term 1: A00 * B00 -> C00
        U_t[0, 0] = 1
        V_t[0, 0] = 1
        W_t[0, 0] = 1
        
        # Term 2: A10 * B00 -> C10
        U_t[1, 1] = 1
        V_t[0, 1] = 1
        W_t[1, 1] = 1
        
        return U_t, V_t, W_t

    U_s, V_s, W_s = get_strassen_222()
    U_t, V_t, W_t = get_trivial_211()
    
    col = 0
    
    # Process 2x2x6 blocks (j=0,1 and j=2,3)
    for block_idx in range(2):
        j_start = block_idx * 2
        for k_block in range(3):
            k_start = k_block * 2
            
            # Add 7 Strassen terms
            for r in range(7):
                # U
                for i in range(2):
                    for j_loc in range(2):
                        val = U_s[i*2+j_loc, r]
                        if val != 0:
                            j_glob = j_start + j_loc
                            U[i*5 + j_glob, col] = val
                            
                # V
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = V_s[j_loc*2+k_loc, r]
                        if val != 0:
                            j_glob = j_start + j_loc
                            k_glob = k_start + k_loc
                            V[j_glob*6 + k_glob, col] = val
                            
                # W
                for k_loc in range(2):
                    for i in range(2):
                        val = W_s[k_loc*2+i, r]
                        if val != 0:
                            k_glob = k_start + k_loc
                            W[k_glob*2 + i, col] = val
                            
                col += 1
                
    # Process 2x1x6 block (j=4)
    for k in range(6):
        # Add 2 trivial terms
        for r in range(2):
            # U
            for i in range(2):
                val = U_t[i, r]
                if val != 0:
                    U[i*5 + 4, col] = val
                    
            # V
            val = V_t[0, r]
            if val != 0:
                V[4*6 + k, col] = val
                
            # W
            for i in range(2):
                val = W_t[i, r]
                if val != 0:
                    W[k*2 + i, col] = val
                    
            col += 1
            
    # Verify
    # Compute residuals
    max_err = 0
    for i in range(n):
        for k in range(p):
            # Sum over j
            # C_ik = sum_j A_ij B_jk
            # Tensor check: sum_r U[i,j,r] V[j,k,r] W[k,i,r] should be 1 for all i,j,k?
            # No, the sum over j is implicit in the matrix multiplication.
            # The tensor decomposition condition is:
            # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{i,j,k}?
            # No, T_{i,j,k} = 1.
            # So sum_r U[i,j,r] V[j,k,r] W[k,i,r] should be 1 for all i,j,k.
            
            for j in range(m):
                s = 0
                for r in range(rank_total):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s != 1:
                    max_err = max(max_err, abs(s - 1))
                    
    if max_err > 1e-6:
        # Fallback to schoolbook if verification fails
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n * m, rank_schoolbook))
        V_sb = np.zeros((m * p, rank_schoolbook))
        W_sb = np.zeros((p * n, rank_schoolbook))
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, c] = 1
                    V_sb[j * p + k, c] = 1
                    W_sb[k * n + i, c] = 1
                    c += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
