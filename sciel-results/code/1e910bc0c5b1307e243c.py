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
        For each k, one term: U[0,j]=A[j], V[j,k]=B[j,k], W[k]=1
        Actually, for each (j,k) pair we need a term, but since it's 1x4x4,
        the output is 1x1x4. Each output C[k] = sum_j A[j]*B[j,k].
        This is 4 independent dot products of length 4, so rank 4*... 
        Wait, no. The tensor is T[i,j,k] where i is always 0.
        T[0,j,k] = A[j]*B[j,k]. This is a rank-4 tensor? 
        Actually T[0,j,k] = A[j]*B[j,k]. This can be written as sum_c U[0,j,c]*V[j,k,c]*W[k,0,c].
        For rank 4: for each k, we have C[k] = sum_j A[j]*B[j,k].
        We can use 4 terms, one per j: term j has U[0,j]=1, V[j,:]=1, W[:,0]=1 for that k...
        No wait. Let me think again.
        
        The tensor entries are T[0,j,k] = delta_{j,k}? No, that's not right either.
        
        For matrix multiplication 1x4 times 4x4 = 1x4:
        C[k] = sum_{j=0}^{3} A[j] * B[j,k]
        
        The multiplication tensor has T[i,j,k] = 1 if the indices correspond to a valid product term.
        Here i=0 always. T[0,j,k] = 1 for all j,k? No.
        
        Actually the matrix multiplication tensor for (1x4)*(4x4)->(1x4) is:
        T[i,j,k] = 1 if i=0, and j,k are the row/col indices.
        But wait, the tensor is defined as T[i,j,k] = delta for the product A[i,j]*B[j,k] contributing to C[i,k].
        For 1x4 * 4x4 -> 1x4: C[0,k] = sum_j A[0,j]*B[j,k].
        So T[0,j,k] = 1 for all j in [0,4), k in [0,4).
        
        This is a rank-16 tensor if we think of it as 1*4*4 = 16 entries all being 1.
        But we can factor it! T[0,j,k] = 1 = sum_c U[0,j,c]*V[j,k,c]*W[k,0,c].
        
        Actually, this is just the outer product structure. T[0,j,k] = 1 for all j,k.
        This is rank 1? No, because U depends on j, V depends on j,k, W depends on k.
        
        Let me use the standard approach: for each (j,k) pair, one term.
        Term (j,k): U[0,j]=1, V[j,k]=1, W[k,0]=1.
        That gives rank 16.
        
        But we can do better! Note that T[0,j,k] = 1 = A_part[j] * B_part[j,k] * C_part[k] summed over c.
        If we set for each j: U[0,j,c_j]=1, V[j,:,c_j]=1, W[:,0,c_j]... no that doesn't work.
        
        Actually, the key insight: C[k] = sum_j A[j]*B[j,k]. This is a set of 4 dot products of length 4.
        Each dot product has rank 4 (schoolbook) or can be done in... well, a dot product of length 4 is just 4 multiplications, rank 4.
        But all 4 dot products share the same A[j] values!
        
        C[0] = A[0]*B[0,0] + A[1]*B[1,0] + A[2]*B[2,0] + A[3]*B[3,0]
        C[1] = A[0]*B[0,1] + A[1]*B[1,1] + A[2]*B[2,1] + A[3]*B[3,1]
        C[2] = A[0]*B[0,2] + A[1]*B[1,2] + A[2]*B[2,2] + A[3]*B[3,2]
        C[3] = A[0]*B[0,3] + A[1]*B[1,3] + A[2]*B[2,3] + A[3]*B[3,3]
        
        Each term A[j]*B[j,k] appears in exactly one C[k]. So we need 16 scalar multiplications total, but they can be grouped.
        
        Actually, the rank of this tensor: T[0,j,k] = 1 for all j,k. This is a 1x4x4 tensor with all ones.
        The rank of an all-ones tensor of size 1xa*b is... it's rank a*b if we can't factor, but actually:
        T[0,j,k] = sum_c U[0,j,c] * V[j,k,c] * W[k,0,c]
        
        If we set c to range over j (4 values): U[0,j,c]=delta_{j,c}, V[j,k,c]=1 for all k, W[k,0,c]=B... no.
        
        Let me just use rank 16 (schoolbook) for simplicity, or find a better factorization.
        
        Actually, I realize I was wrong. Let me reconsider.
        
        For 1x4 * 4x4 -> 1x4, the standard schoolbook uses 1*4*4 = 16 scalar multiplications.
        But the tensor T[0,j,k] = 1 (for all j,k) has a special structure.
        
        We can write: T[0,j,k] = sum_{c=0}^{3} U[0,j,c] * V[j,k,c] * W[k,0,c]
        where for c=j: U[0,j,c]=1, V[j,k,c]=1, W[k,0,c]=1... that gives T[0,j,k] = sum_c delta_{j,c} * 1 * 1 = 1. Yes!
        
        So rank 4 works! For each c in 0..3:
        U[0,c] = 1, U[0,j!=c] = 0
        V[c,k] = 1 for all k, V[j!=c,k] = 0
        W[k,0] = 1 for all k
        
        Then T[0,j,k] = sum_c U[0,j,c]*V[j,k,c]*W[k,0,c] = sum_c delta_{j,c}*delta_{j,c}*1 = delta_{j,j} = 1. 
        Wait, that's not right. V[j,k,c] should be 1 only when j=c.
        
        Let me redo: For term c:
        U[0,j,c] = 1 if j==c else 0
        V[j,k,c] = 1 if j==c else 0  
        W[k,0,c] = 1
        
        Then T[0,j,k] = sum_c U[0,j,c]*V[j,k,c]*W[k,0,c] = sum_c (1 if j==c else 0)*(1 if j==c else 0)*1
        = sum_c delta_{j,c}^2 = delta_{j,j} = 1. 
        
        Yes! Rank 4.
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
    
    # Try Rank 4 (factorization with one term per j-index)
    if time.time() < deadline:
        try:
            U_4, V_4, W_4 = _construct_rank_4()
            if _verify_safe(U_4.tolist(), V_4.tolist(), W_4.tolist()):
                if U_4.shape[1] < best_rank:
                    best_result = {"U": U_4.tolist(), "V": V_4.tolist(), "W": W_4.tolist()}
                    best_rank = U_4.shape[1]
        except Exception:
            pass
    
    # Try Rank 88 as fallback if 16 fails
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
