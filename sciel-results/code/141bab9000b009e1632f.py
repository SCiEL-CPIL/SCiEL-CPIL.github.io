"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Matrix multiplication tensor decomposition for 3x5x7 using block Strassen/Schoolbook."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Define splits for dimensions
    # n=3 -> [2, 1]
    # m=5 -> [2, 2, 1]
    # p=7 -> [2, 2, 2, 1]
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2, 1]
    
    # Pre-compute start indices
    n_starts = []
    curr = 0
    for s in n_splits:
        n_starts.append(curr)
        curr += s
    m_starts = []
    curr = 0
    for s in m_splits:
        m_starts.append(curr)
        curr += s
    p_starts = []
    curr = 0
    for s in p_splits:
        p_starts.append(curr)
        curr += s
        
    # Strassen decomposition for 2x2x2
    # Returns U (4x7), V (4x7), W (4x7)
    def get_strassen_222():
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1 # C10
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1 # C01
        W[3, 2] = 1 # C11
        
        # M4 = A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1 # C00
        W[1, 3] = 1  # C10
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1 # C00
        W[2, 4] = 1 # C01
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C11
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1 # C00
        
        return U, V, W

    # Verify Strassen locally
    U_s, V_s, W_s = get_strassen_222()
    valid_strassen = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U_s[i*2+j, :] * V_s[j*2+k, :] * W_s[k*2+i, :])
                if abs(val - 1.0) > 1e-9:
                    valid_strassen = False
                    break
            if not valid_strassen: break
        if not valid_strassen: break
    if not valid_strassen:
        # Fallback: use schoolbook for 2x2x2 as well if Strassen fails
        # We will just treat all blocks as schoolbook in the loop below if this is false
        pass

    # List to hold (U_col, V_col, W_col, u_idx, v_idx, w_idx)
    # Actually, we build global matrices directly.
    # We need to know the total rank first to allocate, or use lists.
    # Using lists of lists for flexibility.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Iterate over blocks
    for ni, n_size in enumerate(n_splits):
        for mi, m_size in enumerate(m_splits):
            for pi, p_size in enumerate(p_splits):
                n_start = n_starts[ni]
                m_start = m_starts[mi]
                p_start = p_starts[pi]
                
                if n_size == 2 and m_size == 2 and p_size == 2 and valid_strassen:
                    # Use Strassen
                    U_s, V_s, W_s = get_strassen_222()
                    for c in range(7):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        
                        # Map U
                        for i_l in range(2):
                            for j_l in range(2):
                                i_g = n_start + i_l
                                j_g = m_start + j_l
                                coeff = U_s[i_l*2+j_l, c]
                                if coeff != 0:
                                    u_col[i_g*m + j_g] = coeff
                                    
                        # Map V
                        for j_l in range(2):
                            for k_l in range(2):
                                j_g = m_start + j_l
                                k_g = p_start + k_l
                                coeff = V_s[j_l*2+k_l, c]
                                if coeff != 0:
                                    v_col[j_g*p + k_g] = coeff
                                    
                        # Map W
                        for k_l in range(2):
                            for i_l in range(2):
                                k_g = p_start + k_l
                                i_g = n_start + i_l
                                coeff = W_s[k_l*2+i_l, c]
                                if coeff != 0:
                                    w_col[k_g*n + i_g] = coeff
                                    
                        U_list.append(u_col)
                        V_list.append(v_col)
                        W_list.append(w_col)
                        
                else:
                    # Schoolbook
                    for i_l in range(n_size):
                        for j_l in range(m_size):
                            for k_l in range(p_size):
                                i_g = n_start + i_l
                                j_g = m_start + j_l
                                k_g = p_start + k_l
                                
                                u_col = [0.0] * (n * m)
                                v_col = [0.0] * (m * p)
                                w_col = [0.0] * (p * n)
                                
                                u_col[i_g*m + j_g] = 1.0
                                v_col[j_g*p + k_g] = 1.0
                                w_col[k_g*n + i_g] = 1.0
                                
                                U_list.append(u_col)
                                V_list.append(v_col)
                                W_list.append(w_col)
                                
    # Convert lists to NumPy arrays for verification and return
    rank = len(U_list)
    U = np.array(U_list).T if U_list else np.zeros((n*m, 0))
    V = np.array(V_list).T if V_list else np.zeros((m*p, 0))
    W = np.array(W_list).T if W_list else np.zeros((p*n, 0))
    
    # Verify
    if time.time() < deadline and rank > 0:
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U[i*m+j, :], V[j*p+k, :]), W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            # Fallback to pure schoolbook
            rank = n * m * p
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
