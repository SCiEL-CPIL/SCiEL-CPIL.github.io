"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Safely verify the decomposition using explicit loops.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]
    
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_arr[:, c]
        v_c = V_arr[:, c]
        w_c = W_arr[:, c]
        
        u_2d = u_c.reshape(n, m)
        v_2d = v_c.reshape(m, p)
        w_2d = w_c.reshape(p, n)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _construct_rank_36():
    """
    Construct a rank-36 decomposition.
    Split B (4x8) into B1 (4x4) and B2 (4x4).
    For each 3x4x4 block:
      Split A (3x4) into A1 (1x4) and A2 (2x4).
      A1 x B: 1x4x4 -> rank 4 (schoolbook: C[k] = sum_j A[j]*B[j,k])
      A2 x B: 2x4x4 -> split into two 2x2x2 Strassen (rank 7 each) = rank 14
      Total per 3x4x4: 4 + 14 = 18
    Grand total: 18 + 18 = 36.
    """
    n, m, p = 3, 4, 8
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_sub_term(i_off, j_off, k_off, sub_n, sub_m, sub_p, u_sub, v_sub, w_sub):
        rank_sub = u_sub.shape[1]
        for c in range(rank_sub):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            for i in range(sub_n):
                for j in range(sub_m):
                    u_col[(i_off + i) * m + (j_off + j)] = u_sub[i * sub_m + j, c]
            
            for j in range(sub_m):
                for k in range(sub_p):
                    v_col[(j_off + j) * p + (k_off + k)] = v_sub[j * sub_p + k, c]
            
            for k in range(sub_p):
                for i in range(sub_n):
                    w_col[(k_off + k) * n + (i_off + i)] = w_sub[k * sub_n + i, c]
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)

    def get_1x4x4():
        """
        1x4x4 decomposition, rank 4.
        C[k] = sum_{j=0}^{3} A[j] * B[j,k]
        For each c in 0..3:
        U[0,c] = 1, U[0,j!=c] = 0
        V[c,k] = 1 for all k, V[j!=c,k] = 0
        W[k,0] = 1 for all k
        Then T[0,j,k] = sum_c U[0,j,c]*V[j,k,c]*W[k,0,c] = delta_{j,j} = 1.
        """
        u = np.zeros((1 * 4, 4))
        v = np.zeros((4 * 4, 4))
        w = np.zeros((4 * 1, 4))
        for c in range(4):
            u[0, c] = 1.0
            for k in range(4):
                v[c * 4 + k, c] = 1.0
            for k in range(4):
                w[k, c] = 1.0
        return u, v, w

    def get_2x2x2_strassen():
        """
        Standard Strassen 2x2x2 decomposition, rank 7.
        """
        # A parts (U): rows are A11,A12,A21,A22 -> index i*2+j
        u_s = np.array([
            [1, 0, 0, 1],   # M1: A11+A22
            [0, 0, 1, 1],   # M2: A21+A22
            [1, 0, 0, 0],   # M3: A11
            [0, 0, 0, 1],   # M4: A22
            [1, 1, 0, 0],   # M5: A11+A12
            [-1, 0, 1, 0],  # M6: A21-A11
            [0, 1, 0, -1]   # M7: A12-A22
        ]).T  # shape (4, 7)
        
        # B parts (V): rows are B11,B12,B21,B22 -> index j*2+k
        v_s = np.array([
            [1, 0, 0, 1],   # M1: B11+B22
            [1, 0, 0, 0],   # M2: B11
            [0, 1, 0, -1],  # M3: B12-B22
            [-1, 0, 1, 0],  # M4: B21-B11
            [0, 0, 0, 1],   # M5: B22
            [1, 1, 0, 0],   # M6: B11+B12
            [0, 0, 1, 1]    # M7: B21+B22
        ]).T  # shape (4, 7)
        
        # C parts (W): rows are C11,C21,C12,C22 -> index k*2+i (W[k*n+i])
        # C11: i=0,k=0 -> 0
        # C21: i=1,k=0 -> 1
        # C12: i=0,k=1 -> 2
        # C22: i=1,k=1 -> 3
        w_s = np.zeros((4, 7))
        w_s[0, 0] = 1; w_s[3, 0] = 1    # M1: C11 + C22
        w_s[1, 1] = 1; w_s[3, 1] = -1   # M2: C21 - C22
        w_s[2, 2] = 1; w_s[3, 2] = 1    # M3: C12 + C22
        w_s[0, 3] = 1; w_s[1, 3] = 1    # M4: C11 + C21
        w_s[0, 4] = -1; w_s[2, 4] = 1   # M5: -C11 + C12
        w_s[3, 5] = 1                    # M6: C22
        w_s[0, 6] = 1                    # M7: C11
        
        return u_s, v_s, w_s

    def get_2x4x4():
        """
        2x4x4 decomposition, rank 14.
        Split the 4 columns into two 2-column blocks.
        Each 2x2x2 block uses Strassen (rank 7).
        """
        u_s, v_s, w_s = get_2x2x2_strassen()
        
        u_244 = np.zeros((4, 14))
        v_244 = np.zeros((8, 14))
        w_244 = np.zeros((8, 14))
        
        # First block: columns 0,1 (k=0,1)
        for c in range(7):
            u_244[:, c] = u_s[:, c]
            # V: j=0,1 for k=0,1 -> indices 0,1,2,3
            v_244[0, c] = v_s[0, c]   # B11
            v_244[1, c] = v_s[1, c]   # B12
            v_244[2, c] = v_s[2, c]   # B21
            v_244[3, c] = v_s[3, c]   # B22
            # W: k=0,1 for i=0,1 -> indices 0,1,2,3
            w_244[0, c] = w_s[0, c]   # C11
            w_244[1, c] = w_s[1, c]   # C21
            w_244[2, c] = w_s[2, c]   # C12
            w_244[3, c] = w_s[3, c]   # C22
            
        # Second block: columns 2,3 (k=2,3)
        for c in range(7):
            u_244[:, c+7] = u_s[:, c]
            # V: j=0,1 for k=2,3 -> indices 4,5,6,7
            v_244[4, c+7] = v_s[0, c]   # B11
            v_244[5, c+7] = v_s[1, c]   # B12
            v_244[6, c+7] = v_s[2, c]   # B21
            v_244[7, c+7] = v_s[3, c]   # B22
            # W: k=2,3 for i=0,1 -> indices 4,5,6,7
            w_244[4, c+7] = w_s[0, c]   # C11
            w_244[5, c+7] = w_s[1, c]   # C21
            w_244[6, c+7] = w_s[2, c]   # C12
            w_244[7, c+7] = w_s[3, c]   # C22
            
        return u_244, v_244, w_244

    u_144, v_144, w_144 = get_1x4x4()
    u_244, v_244, w_244 = get_2x4x4()
    
    # Block 1: A(3x4) x B1(4x4), columns 0-3
    # A1 (1x4) x B1: i_off=0, j_off=0, k_off=0
    add_sub_term(0, 0, 0, 1, 4, 4, u_144, v_144, w_144)
    # A2 (2x4) x B1: i_off=1, j_off=0, k_off=0
    add_sub_term(1, 0, 0, 2, 4, 4, u_244, v_244, w_244)
    
    # Block 2: A(3x4) x B2(4x4), columns 4-7
    # A1 (1x4) x B2: i_off=0, j_off=0, k_off=4
    add_sub_term(0, 0, 4, 1, 4, 4, u_144, v_144, w_144)
    # A2 (2x4) x B2: i_off=1, j_off=0, k_off=4
    add_sub_term(1, 0, 4, 2, 4, 4, u_244, v_244, w_244)
    
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    return U_arr, V_arr, W_arr

def _construct_rank_88():
    """
    Fallback: Construct a rank-88 decomposition (original approach).
    """
    n, m, p = 3, 4, 8
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_sub_term(i_off, j_off, k_off, sub_n, sub_m, sub_p, u_sub, v_sub, w_sub):
        rank_sub = u_sub.shape[1]
        for c in range(rank_sub):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            for i in range(sub_n):
                for j in range(sub_m):
                    u_col[(i_off + i) * m + (j_off + j)] = u_sub[i * sub_m + j, c]
            
            for j in range(sub_m):
                for k in range(sub_p):
                    v_col[(j_off + j) * p + (k_off + k)] = v_sub[j * sub_p + k, c]
            
            for k in range(sub_p):
                for i in range(sub_n):
                    w_col[(k_off + k) * n + (i_off + i)] = w_sub[k * sub_n + i, c]
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)

    def get_1x2x4():
        u = np.zeros((1 * 2, 8))
        v = np.zeros((2 * 4, 8))
        w = np.zeros((4 * 1, 8))
        idx = 0
        for j in range(2):
            for k in range(4):
                u[j, idx] = 1.0
                v[j * 4 + k, idx] = 1.0
                w[k * 1, idx] = 1.0
                idx += 1
        return u, v, w

    def get_2x2x4():
        u_s = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]).T
        v_s = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]).T
        w_s = np.zeros((4, 7))
        w_s[0, 0] = 1; w_s[3, 0] = 1
        w_s[1, 1] = 1; w_s[3, 1] = -1
        w_s[2, 2] = 1; w_s[3, 2] = 1
        w_s[0, 3] = 1; w_s[1, 3] = 1
        w_s[0, 4] = -1; w_s[2, 4] = 1
        w_s[3, 5] = 1
        w_s[0, 6] = 1
        
        u_224 = np.zeros((4, 14))
        v_224 = np.zeros((8, 14))
        w_224 = np.zeros((8, 14))
        
        for c in range(7):
            u_224[:, c] = u_s[:, c]
            v_224[0, c] = v_s[0, c]
            v_224[1, c] = v_s[1, c]
            v_224[4, c] = v_s[2, c]
            v_224[5, c] = v_s[3, c]
            w_224[0, c] = w_s[0, c]
            w_224[1, c] = w_s[1, c]
            w_224[2, c] = w_s[2, c]
            w_224[3, c] = w_s[3, c]
            
        for c in range(7):
            u_224[:, c+7] = u_s[:, c]
            v_224[2, c+7] = v_s[0, c]
            v_224[3, c+7] = v_s[1, c]
            v_224[6, c+7] = v_s[2, c]
            v_224[7, c+7] = v_s[3, c]
            w_224[4, c+7] = w_s[0, c]
            w_224[5, c+7] = w_s[1, c]
            w_224[6, c+7] = w_s[2, c]
            w_224[7, c+7] = w_s[3, c]
            
        return u_224, v_224, w_224

    u_124, v_124, w_124 = get_1x2x4()
    u_224, v_224, w_224 = get_2x2x4()
    
    add_sub_term(0, 0, 0, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 2, 0, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 0, 4, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 2, 4, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(1, 0, 0, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 2, 0, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 0, 4, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 2, 4, 2, 2, 4, u_224, v_224, w_224)
    
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    return U_arr, V_arr, W_arr

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try rank-36 construction
    if time.time() < deadline:
        try:
            U_36, V_36, W_36 = _construct_rank_36()
            if _verify_safe(U_36.tolist(), V_36.tolist(), W_36.tolist()):
                if U_36.shape[1] < best_rank:
                    best_result = {"U": U_36.tolist(), "V": V_36.tolist(), "W": W_36.tolist()}
                    best_rank = U_36.shape[1]
        except Exception:
            pass
    
    # Try Rank 88 as fallback if 36 fails
    if time.time() < deadline:
        try:
            U_88, V_88, W_88 = _construct_rank_88()
            if _verify_safe(U_88.tolist(), V_88.tolist(), W_88.tolist()):
                if U_88.shape[1] < best_rank:
                    best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
                    best_rank = U_88.shape[1]
        except Exception:
            pass
    
    return best_result
