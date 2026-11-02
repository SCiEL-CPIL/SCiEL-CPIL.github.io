import numpy as np

DIMENSIONS = [3, 4, 7]

def get_strassen_2x2_2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    Indices:
    U row: i*m + j (0..3)
    V row: j*p + k (0..3)
    W row: k*n + i (0..3)
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen terms
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1 # C10 += M2, C11 -= M2
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00 += M7
    
    return U, V, W

def get_trivial(n, m, p):
    """
    Returns U, V, W for nxmxp matrix multiplication using trivial decomposition.
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by splitting m=4 into 2+2.
    # Block 1: n=3, m=2, p=7
    # Block 2: n=3, m=2, p=7
    
    # We need a function to get decomposition for n=3, m=2, p=7.
    # Split n=3 into 2+1.
    # Sub-block 1.1: n=2, m=2, p=7
    # Sub-block 1.2: n=1, m=2, p=7
    
    # We need decomposition for n=2, m=2, p=7.
    # Split p=7 into 3+4.
    # Sub-sub-block 1.1.1: n=2, m=2, p=3
    # Sub-sub-block 1.1.2: n=2, m=2, p=4
    
    # We need decomposition for n=2, m=2, p=3.
    # Split p=3 into 1+2.
    # Sub-sub-sub-block 1.1.1.1: n=2, m=2, p=1 (Trivial)
    # Sub-sub-sub-block 1.1.1.2: n=2, m=2, p=2 (Strassen)
    
    # We need decomposition for n=2, m=2, p=4.
    # Split p=4 into 2+2.
    # Sub-sub-sub-block 1.1.2.1: n=2, m=2, p=2 (Strassen)
    # Sub-sub-sub-block 1.1.2.2: n=2, m=2, p=2 (Strassen)
    
    # We need decomposition for n=1, m=2, p=7.
    # Trivial.
    
    # Let's create a generic composer.
    
    def compose_decompositions(decomps, n_total, m_total, p_total):
        """
        decomps: list of (U, V, W, n_i, m_i, p_i, i_offset, j_offset, k_offset)
        Returns combined U, V, W for n_total x m_total x p_total.
        """
        total_rank = sum(U.shape[1] for U, V, W, *_ in decomps)
        U_total = np.zeros((n_total * m_total, total_rank))
        V_total = np.zeros((m_total * p_total, total_rank))
        W_total = np.zeros((p_total * n_total, total_rank))
        
        col_offset = 0
        for U, V, W, n_i, m_i, p_i, i_off, j_off, k_off in decomps:
            rank_i = U.shape[1]
            
            # Map U
            for r in range(rank_i):
                u_col = U[:, r]
                for row in range(n_i * m_i):
                    val = u_col[row]
                    if val != 0:
                        i_local = row // m_i
                        j_local = row % m_i
                        i_global = i_local + i_off
                        j_global = j_local + j_off
                        U_total[i_global * m_total + j_global, col_offset + r] = val
                        
            # Map V
            for r in range(rank_i):
                v_col = V[:, r]
                for row in range(m_i * p_i):
                    val = v_col[row]
                    if val != 0:
                        j_local = row // p_i
                        k_local = row % p_i
                        j_global = j_local + j_off
                        k_global = k_local + k_off
                        V_total[j_global * p_total + k_global, col_offset + r] = val
                        
            # Map W
            for r in range(rank_i):
                w_col = W[:, r]
                for row in range(p_i * n_i):
                    val = w_col[row]
                    if val != 0:
                        k_local = row // n_i
                        i_local = row % n_i
                        k_global = k_local + k_off
                        i_global = i_local + i_off
                        W_total[k_global * n_total + i_global, col_offset + r] = val
                        
            col_offset += rank_i
            
        return U_total, V_total, W_total

    # Get base decompositions
    U_s, V_s, W_s = get_strassen_2x2_2()
    
    def get_2x2_p1():
        return get_trivial(2, 2, 1)
        
    def get_1x2_p7():
        return get_trivial(1, 2, 7)
        
    # Build 2x2x3
    U_221, V_221, W_221 = get_2x2_p1()
    # U_221 is (4, 4), V_221 is (4, 4), W_221 is (4, 4)
    
    # 2x2x3 = 2x2x1 (k=0) + 2x2x2 (k=1,2)
    # Note: The trivial 2x2x1 covers k=0.
    # The Strassen 2x2x2 covers k=1,2? No, Strassen is for 2x2x2.
    # We need to map the Strassen output to k=1,2.
    # But Strassen assumes p=2. The local k indices are 0,1.
    # So we add offset k=1.
    
    # Let's create the decompositions for 2x2x3
    # Part 1: Trivial 2x2x1 for k=0
    U_p1, V_p1, W_p1 = U_221, V_221, W_221
    # Part 2: Strassen 2x2x2 for k=1,2
    U_p2, V_p2, W_p2 = U_s, V_s, W_s
    
    decomps_223 = [
        (U_p1, V_p1, W_p1, 2, 2, 1, 0, 0, 0),
        (U_p2, V_p2, W_p2, 2, 2, 2, 0, 0, 1)
    ]
    U_223, V_223, W_223 = compose_decompositions(decomps_223, 2, 2, 3)
    
    # Build 2x2x4
    # Part 1: Strassen 2x2x2 for k=0,1
    # Part 2: Strassen 2x2x2 for k=2,3
    decomps_224 = [
        (U_s, V_s, W_s, 2, 2, 2, 0, 0, 0),
        (U_s, V_s, W_s, 2, 2, 2, 0, 0, 2)
    ]
    U_224, V_224, W_224 = compose_decompositions(decomps_224, 2, 2, 4)
    
    # Build 2x2x7
    # Part 1: 2x2x3 for k=0,1,2
    # Part 2: 2x2x4 for k=3,4,5,6
    decomps_227 = [
        (U_223, V_223, W_223, 2, 2, 3, 0, 0, 0),
        (U_224, V_224, W_224, 2, 2, 4, 0, 0, 3)
    ]
    U_227, V_227, W_227 = compose_decompositions(decomps_227, 2, 2, 7)
    
    # Build 1x2x7
    U_127, V_127, W_127 = get_1x2_p7()
    
    # Build 3x2x7
    # Part 1: 2x2x7 for i=0,1
    # Part 2: 1x2x7 for i=2
    decomps_327 = [
        (U_227, V_227, W_227, 2, 2, 7, 0, 0, 0),
        (U_127, V_127, W_127, 1, 2, 7, 2, 0, 0)
    ]
    U_327, V_327, W_327 = compose_decompositions(decomps_327, 3, 2, 7)
    
    # Build 3x4x7
    # Part 1: 3x2x7 for j=0,1
    # Part 2: 3x2x7 for j=2,3
    decomps_347 = [
        (U_327, V_327, W_327, 3, 2, 7, 0, 0, 0),
        (U_327, V_327, W_327, 3, 2, 7, 0, 2, 0)
    ]
    U_final, V_final, W_final = compose_decompositions(decomps_347, 3, 4, 7)
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
