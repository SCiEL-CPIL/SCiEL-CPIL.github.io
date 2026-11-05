import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook fallback
    rank_sb = n * m * p  # 96
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb

    def verify(U, V, W, tol=1e-8):
        """Verify the decomposition exactly."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Build Strassen-based decomposition: 88 rank terms
    # Split: i=0,1 (2 rows) use Strassen on 2x2x2 blocks (8 blocks x 7 = 56)
    #        i=2 (1 row) uses schoolbook (1*4*8 = 32)
    # Total = 56 + 32 = 88
    
    U_str = np.zeros((n*m, 88))
    V_str = np.zeros((m*p, 88))
    W_str = np.zeros((p*n, 88))
    
    col = 0
    
    # Strassen's 7 rank terms for 2x2x2 tensor
    # For a 2x2x2 multiplication tensor T[i,j,k] with i,j,k in {0,1}:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen coefficients: for each of the 7 multiplications,
    # the coefficient of U[i,j], V[j,k], W[k,i]
    # U coeff for (i,j) in 2x2, V coeff for (j,k) in 2x2, W coeff for (k,i) in 2x2
    
    # M1: U: A00+A11 -> {(0,0):1,(1,1):1}, V: B00+B11 -> {(0,0):1,(1,1):1}, W: C00-M4+M5-M7 -> {(0,0):1}
    # M2: U: A10+A11 -> {(1,0):1,(1,1):1}, V: B00 -> {(0,0):1}, W: C10+M5 -> {(1,0):1}
    # M3: U: A00 -> {(0,0):1}, V: B10-B11 -> {(1,0):1,(1,1):-1}, W: C01+M4 -> {(0,1):1}
    # M4: U: A11 -> {(1,1):1}, V: B01-B00 -> {(0,1):1,(0,0):-1}, W: C00-M4+M5-M7 -> {(0,0):-1}, C01+M4 -> {(0,1):1}
    # M5: U: A00+A01 -> {(0,0):1,(0,1):1}, V: B11 -> {(1,1):1}, W: C00-M4+M5-M7 -> {(0,0):1}, C10+M5 -> {(1,0):1}
    # M6: U: A10-A00 -> {(1,0):1,(0,0):-1}, V: B00+B10 -> {(0,0):1,(1,0):1}, W: C11-M2+M3+M6 -> {(1,1):1}
    # M7: U: A01-A11 -> {(0,1):1,(1,1):-1}, V: B01+B11 -> {(0,1):1,(1,1):1}, W: C00-M4+M5-M7 -> {(0,0):-1}
    
    # Let me define the 7 Strassen terms more carefully
    # Each term c has: u_coeff[i,j], v_coeff[j,k], w_coeff[k,i] for i,j,k in {0,1}
    
    strassen_terms = []
    
    # M1
    u1 = {(0,0):1, (1,1):1}
    v1 = {(0,0):1, (1,1):1}
    w1 = {(0,0):1}
    strassen_terms.append((u1, v1, w1))
    
    # M2
    u2 = {(1,0):1, (1,1):1}
    v2 = {(0,0):1}
    w2 = {(1,0):1}
    strassen_terms.append((u2, v2, w2))
    
    # M3
    u3 = {(0,0):1}
    v3 = {(1,0):1, (1,1):-1}
    w3 = {(0,1):1}
    strassen_terms.append((u3, v3, w3))
    
    # M4
    u4 = {(1,1):1}
    v4 = {(0,1):1, (0,0):-1}
    w4 = {(0,0):-1, (0,1):1}
    strassen_terms.append((u4, v4, w4))
    
    # M5
    u5 = {(0,0):1, (0,1):1}
    v5 = {(1,1):1}
    w5 = {(0,0):1, (1,0):1}
    strassen_terms.append((u5, v5, w5))
    
    # M6
    u6 = {(1,0):1, (0,0):-1}
    v6 = {(0,0):1, (1,0):1}
    w6 = {(1,1):1}
    strassen_terms.append((u6, v6, w6))
    
    # M7
    u7 = {(0,1):1, (1,1):-1}
    v7 = {(0,1):1, (1,1):1}
    w7 = {(0,0):-1}
    strassen_terms.append((u7, v7, w7))
    
    # Now fill in the Strassen part for i=0,1
    # The 2x2x2 blocks: for i_base in [0], j_base in [0,2], k_base in [0,2,4,6]
    # i ranges over {i_base, i_base+1} = {0,1} (only i_base=0)
    # j ranges over {j_base, j_base+1}
    # k ranges over {k_base, k_base+1}
    
    for j_base in range(0, m, 2):
        for k_base in range(0, p, 2):
            for term_idx in range(7):
                u_coeff, v_coeff, w_coeff = strassen_terms[term_idx]
                for (di, dj), uc in u_coeff.items():
                    i_global = 0 + di  # i_base = 0
                    j_global = j_base + dj
                    U_str[i_global * m + j_global, col] += uc
                for (dj, dk), vc in v_coeff.items():
                    j_global = j_base + dj
                    k_global = k_base + dk
                    V_str[j_global * p + k_global, col] += vc
                for (dk, di), wc in w_coeff.items():
                    k_global = k_base + dk
                    i_global = 0 + di
                    W_str[k_global * n + i_global, col] += wc
                col += 1
    
    # Now fill in the schoolbook part for i=2
    for j in range(m):
        for k in range(p):
            U_str[2 * m + j, col] = 1
            V_str[j * p + k, col] = 1
            W_str[k * n + 2, col] = 1
            col += 1
    
    # Verify the Strassen-based decomposition
    if col == 88 and verify(U_str, V_str, W_str):
        best_result = (U_str.copy(), V_str.copy(), W_str.copy())
        best_rank = 88
    
    # Try alternating minimization to potentially reduce rank further
    lower_bound = max(n*m, m*p, p*n)
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 25:
            break
        if target_rank >= best_rank:
            continue
        
        found = False
        for trial in range(8):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(12345 + trial * 7919 + target_rank * 31)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            U *= np.sqrt(target_rank / (n*m))
            V *= np.sqrt(target_rank / (m*p))
            W *= np.sqrt(target_rank / (p*n))
            
            # Alternating minimization
            r = target_rank
            for it in range(150):
                if time.time() > deadline - 10:
                    break
                
                for i in range(n):
                    for j in range(m):
                        M_mat = np.zeros((p, r))
                        for k in range(p):
                            M_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :], _, _, _ = np.linalg.lstsq(M_mat.T, np.ones(p), rcond=None)
                        except:
                            pass
                
                for j in range(m):
                    for k in range(p):
                        M_mat = np.zeros((n, r))
                        for i in range(n):
                            M_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :], _, _, _ = np.linalg.lstsq(M_mat.T, np.ones(n), rcond=None)
                        except:
                            pass
                
                for k in range(p):
                    for i in range(n):
                        M_mat = np.zeros((m, r))
                        for j in range(m):
                            M_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :], _, _, _ = np.linalg.lstsq(M_mat.T, np.ones(m), rcond=None)
                        except:
                            pass
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            error += (val - 1.0)**2
                if error < 1e-10:
                    break
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        error += (val - 1.0)**2
            
            if error < 1e-8 and verify(U, V, W):
                best_rank = target_rank
                best_result = (U.copy(), V.copy(), W.copy())
                found = True
                break
    
    U_list = best_result[0].tolist()
    V_list = best_result[1].tolist()
    W_list = best_result[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
