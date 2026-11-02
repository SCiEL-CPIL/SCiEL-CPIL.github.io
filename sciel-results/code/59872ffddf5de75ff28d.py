import numpy as np

def get_strassen_222():
    """
    Returns U, V, W for M_2,2,2 using Strassen's algorithm.
    Shapes: (4, 7), (4, 7), (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    r = 7
    U = np.zeros((4, r))
    V = np.zeros((4, r))
    W = np.zeros((4, r))
    
    # P1 = (A00 + A11)(B00 + B11)
    # U1: 1 at 00, 11
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # P2 = (A10 + A11)B00
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[1, 1] = 1.0 # C10 gets P2? 
    # C10 = P2 + P6. C11 = P1 - P2 + P6 + P7.
    # P2 contributes to C10 (coeff 1) and C11 (coeff -1).
    # So W2 should have 1 at 10 and -1 at 11.
    W[1, 1] = 1.0
    W[3, 1] = -1.0
    
    # P3 = A00(B01 - B11)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    # C01 = P3 + P5. C00 = ... - P5 ...
    # P3 contributes to C01 (coeff 1).
    W[1, 2] = 1.0
    
    # P4 = A11(B10 - B00)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    # C00 = P1 + P4 ...
    # P4 contributes to C00 (coeff 1).
    W[0, 3] = 1.0
    
    # P5 = (A00 + A01)B11
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    # C01 = P3 + P5. C00 = ... - P5 ...
    # P5 contributes to C01 (coeff 1) and C00 (coeff -1).
    W[1, 4] = 1.0
    W[0, 4] = -1.0
    
    # P6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    # C10 = P2 + P6. C11 = ... + P6 ...
    # P6 contributes to C10 (coeff 1) and C11 (coeff 1).
    W[1, 5] = 1.0
    W[3, 5] = 1.0
    
    # P7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    # C11 = ... + P7. C00 = ... - P7.
    # P7 contributes to C11 (coeff 1) and C00 (coeff -1).
    W[3, 6] = 1.0
    W[0, 6] = -1.0
    
    return U, V, W

def get_schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = 3, 4, 7
    # Global dimensions
    N, M, P = n, m, p
    
    # We will accumulate the factors
    # Total rank is 78
    R = 78
    U_global = np.zeros((N * M, R))
    V_global = np.zeros((M * P, R))
    W_global = np.zeros((P * N, R))
    
    # We define a recursive function to add a sub-problem
    # sub_problem(n_sub, m_sub, p_sub, offset_i, offset_j, offset_k)
    # offset_i, offset_j, offset_k are the starting indices in the global space
    
    def add_subproblem(n_s, m_s, p_s, off_i, off_j, off_k, rank_offset):
        # Determine if this is a leaf
        # Base cases:
        # 1. n_s == 1: Schoolbook
        # 2. n_s == 2, m_s == 2, p_s == 2: Strassen
        # 3. n_s == 2, m_s == 2, p_s == 1: Schoolbook
        # Otherwise, split
        
        if n_s == 1:
            U, V, W = get_schoolbook(n_s, m_s, p_s)
            r = U.shape[1]
            for c in range(r):
                u_vec = U[:, c]
                v_vec = V[:, c]
                w_vec = W[:, c]
                
                # Map to global
                # u_vec index: i_s * m_s + j_s
                # global index: (i_s + off_i) * M + (j_s + off_j)
                for idx in range(len(u_vec)):
                    if u_vec[idx] != 0:
                        i_s = idx // m_s
                        j_s = idx % m_s
                        g_i = i_s + off_i
                        g_j = j_s + off_j
                        U_global[g_i * M + g_j, rank_offset + c] = u_vec[idx]
                        
                # v_vec index: j_s * p_s + k_s
                # global index: (j_s + off_j) * P + (k_s + off_k)
                for idx in range(len(v_vec)):
                    if v_vec[idx] != 0:
                        j_s = idx // p_s
                        k_s = idx % p_s
                        g_j = j_s + off_j
                        g_k = k_s + off_k
                        V_global[g_j * P + g_k, rank_offset + c] = v_vec[idx]
                        
                # w_vec index: k_s * n_s + i_s
                # global index: (k_s + off_k) * N + (i_s + off_i)
                for idx in range(len(w_vec)):
                    if w_vec[idx] != 0:
                        k_s = idx // n_s
                        i_s = idx % n_s
                        g_k = k_s + off_k
                        g_i = i_s + off_i
                        W_global[g_k * N + g_i, rank_offset + c] = w_vec[idx]
                        
            return r

        elif n_s == 2 and m_s == 2 and p_s == 2:
            U, V, W = get_strassen_222()
            r = U.shape[1]
            for c in range(r):
                u_vec = U[:, c]
                v_vec = V[:, c]
                w_vec = W[:, c]
                
                for idx in range(len(u_vec)):
                    if u_vec[idx] != 0:
                        i_s = idx // m_s
                        j_s = idx % m_s
                        g_i = i_s + off_i
                        g_j = j_s + off_j
                        U_global[g_i * M + g_j, rank_offset + c] = u_vec[idx]
                        
                for idx in range(len(v_vec)):
                    if v_vec[idx] != 0:
                        j_s = idx // p_s
                        k_s = idx % p_s
                        g_j = j_s + off_j
                        g_k = k_s + off_k
                        V_global[g_j * P + g_k, rank_offset + c] = v_vec[idx]
                        
                for idx in range(len(w_vec)):
                    if w_vec[idx] != 0:
                        k_s = idx // n_s
                        i_s = idx % n_s
                        g_k = k_s + off_k
                        g_i = i_s + off_i
                        W_global[g_k * N + g_i, rank_offset + c] = w_vec[idx]
                        
            return r

        elif n_s == 2 and m_s == 2 and p_s == 1:
            U, V, W = get_schoolbook(n_s, m_s, p_s)
            r = U.shape[1]
            for c in range(r):
                u_vec = U[:, c]
                v_vec = V[:, c]
                w_vec = W[:, c]
                
                for idx in range(len(u_vec)):
                    if u_vec[idx] != 0:
                        i_s = idx // m_s
                        j_s = idx % m_s
                        g_i = i_s + off_i
                        g_j = j_s + off_j
                        U_global[g_i * M + g_j, rank_offset + c] = u_vec[idx]
                        
                for idx in range(len(v_vec)):
                    if v_vec[idx] != 0:
                        j_s = idx // p_s
                        k_s = idx % p_s
                        g_j = j_s + off_j
                        g_k = k_s + off_k
                        V_global[g_j * P + g_k, rank_offset + c] = v_vec[idx]
                        
                for idx in range(len(w_vec)):
                    if w_vec[idx] != 0:
                        k_s = idx // n_s
                        i_s = idx % n_s
                        g_k = k_s + off_k
                        g_i = i_s + off_i
                        W_global[g_k * N + g_i, rank_offset + c] = w_vec[idx]
                        
            return r

        else:
            # Split logic
            # 1. If n_s == 3, split n into 1+2
            if n_s == 3:
                r1 = add_subproblem(1, m_s, p_s, off_i, off_j, off_k, rank_offset)
                r2 = add_subproblem(2, m_s, p_s, off_i + 1, off_j, off_k, rank_offset + r1)
                return r1 + r2
            
            # 2. If m_s == 4, split m into 2+2
            if m_s == 4:
                r1 = add_subproblem(n_s, 2, p_s, off_i, off_j, off_k, rank_offset)
                r2 = add_subproblem(n_s, 2, p_s, off_i, off_j + 2, off_k, rank_offset + r1)
                return r1 + r2
            
            # 3. If p_s == 7, split p into 2+5
            if p_s == 7:
                r1 = add_subproblem(n_s, m_s, 2, off_i, off_j, off_k, rank_offset)
                r2 = add_subproblem(n_s, m_s, 5, off_i, off_j, off_k + 2, rank_offset + r1)
                return r1 + r2
                
            # 4. If p_s == 5, split p into 2+3
            if p_s == 5:
                r1 = add_subproblem(n_s, m_s, 2, off_i, off_j, off_k, rank_offset)
                r2 = add_subproblem(n_s, m_s, 3, off_i, off_j, off_k + 2, rank_offset + r1)
                return r1 + r2
                
            # 5. If p_s == 3, split p into 1+2
            if p_s == 3:
                r1 = add_subproblem(n_s, m_s, 1, off_i, off_j, off_k, rank_offset)
                r2 = add_subproblem(n_s, m_s, 2, off_i, off_j, off_k + 1, rank_offset + r1)
                return r1 + r2
            
            # Fallback: Schoolbook
            U, V, W = get_schoolbook(n_s, m_s, p_s)
            r = U.shape[1]
            for c in range(r):
                u_vec = U[:, c]
                v_vec = V[:, c]
                w_vec = W[:, c]
                
                for idx in range(len(u_vec)):
                    if u_vec[idx] != 0:
                        i_s = idx // m_s
                        j_s = idx % m_s
                        g_i = i_s + off_i
                        g_j = j_s + off_j
                        U_global[g_i * M + g_j, rank_offset + c] = u_vec[idx]
                        
                for idx in range(len(v_vec)):
                    if v_vec[idx] != 0:
                        j_s = idx // p_s
                        k_s = idx % p_s
                        g_j = j_s + off_j
                        g_k = k_s + off_k
                        V_global[g_j * P + g_k, rank_offset + c] = v_vec[idx]
                        
                for idx in range(len(w_vec)):
                    if w_vec[idx] != 0:
                        k_s = idx // n_s
                        i_s = idx % n_s
                        g_k = k_s + off_k
                        g_i = i_s + off_i
                        W_global[g_k * N + g_i, rank_offset + c] = w_vec[idx]
                        
            return r

    total_rank = add_subproblem(3, 4, 7, 0, 0, 0, 0)
    
    # The total rank should be 78
    # If total_rank != R, we have a bug in the split logic or base cases.
    # We allocated R=78.
    
    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
