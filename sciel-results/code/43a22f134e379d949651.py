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

def _construct_rank_88():
    """
    Construct a rank-88 decomposition using Strassen blocks.
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
        # Rank 8 schoolbook for 1x2x4
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
        # Rank 14 using two 2x2x2 Strassen decompositions
        u_s = np.array([
            [1, 0, 1, 0, 1, -1, 0], # A11
            [0, 0, 0, 0, 1, 0, 1],  # A12
            [0, 1, 0, 0, 0, 1, 0],  # A21
            [1, 1, 0, 1, 0, 0, -1]  # A22
        ]).T # Shape (4, 7)
        
        v_s = np.array([
            [1, 1, 0, -1, 0, 1, 0], # B11
            [0, 0, 1, 0, 0, 1, 0],  # B12
            [0, 0, 0, 1, 0, 0, 1],  # B21
            [1, 0, -1, 0, 1, 0, 1]  # B22
        ]).T # Shape (4, 7)
        
        w_s = np.zeros((4, 7))
        # M1: C11 + C22
        w_s[0, 0] = 1; w_s[3, 0] = 1
        # M2: C21 - C22
        w_s[1, 1] = 1; w_s[3, 1] = -1
        # M3: C12 + C22
        w_s[2, 2] = 1; w_s[3, 2] = 1
        # M4: C11 + C21
        w_s[0, 3] = 1; w_s[1, 3] = 1
        # M5: -C11 + C12
        w_s[0, 4] = -1; w_s[2, 4] = 1
        # M6: C22
        w_s[3, 5] = 1
        # M7: C11
        w_s[0, 6] = 1
        
        # Construct 2x2x4
        u_224 = np.zeros((4, 14))
        v_224 = np.zeros((8, 14))
        w_224 = np.zeros((8, 14))
        
        # First half (k=0,1)
        for c in range(7):
            u_224[:, c] = u_s[:, c]
            v_224[0, c] = v_s[0, c] # B11
            v_224[1, c] = v_s[1, c] # B12
            v_224[2, c] = v_s[2, c] # B21
            v_224[3, c] = v_s[3, c] # B22
            w_224[0, c] = w_s[0, c]
            w_224[1, c] = w_s[1, c]
            w_224[2, c] = w_s[2, c]
            w_224[3, c] = w_s[3, c]
            
        # Second half (k=2,3)
        for c in range(7):
            u_224[:, c+7] = u_s[:, c]
            v_224[2, c+7] = v_s[0, c] # B11
            v_224[3, c+7] = v_s[1, c] # B12
            v_224[6, c+7] = v_s[2, c] # B21
            v_224[7, c+7] = v_s[3, c] # B22
            w_224[4, c+7] = w_s[0, c]
            w_224[5, c+7] = w_s[1, c]
            w_224[6, c+7] = w_s[2, c]
            w_224[7, c+7] = w_s[3, c]
            
        return u_224, v_224, w_224

    u_124, v_124, w_124 = get_1x2x4()
    u_224, v_224, w_224 = get_2x2x4()
    
    # A1 (1x4) x B1 (4x4)
    add_sub_term(0, 0, 0, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 2, 0, 1, 2, 4, u_124, v_124, w_124)
    
    # A1 (1x4) x B2 (4x4)
    add_sub_term(0, 0, 4, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 2, 4, 1, 2, 4, u_124, v_124, w_124)
    
    # A2 (2x4) x B1 (4x4)
    add_sub_term(1, 0, 0, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 2, 0, 2, 2, 4, u_224, v_224, w_224)
    
    # A2 (2x4) x B2 (4x4)
    add_sub_term(1, 0, 4, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 2, 4, 2, 2, 4, u_224, v_224, w_224)
    
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    return U_arr, V_arr, W_arr

def _als_decomposition(n, m, p, rank, iterations=50, seed=0):
    """
    Attempt to find a rank decomposition using ALS.
    Returns (U, V, W) or None if it fails.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    for _ in range(iterations):
        # Update U
        # For each i, j: minimize sum_k (1 - sum_c U_ij_c V_jk_c W_ki_c)^2
        # A_ij is p x rank, A_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
        # b_ij is p x 1, b_ij[k] = 1
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                # Solve A u = b in least squares sense
                u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i * m + j, :] = u
        
        # Update V
        # For each j, k: minimize sum_i (1 - sum_c U_ij_c V_jk_c W_ki_c)^2
        # A_jk is n x rank, A_jk[i, c] = U[i*m+j, c] * W[k*n+i, c]
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j * p + k, :] = v
        
        # Update W
        # For each k, i: minimize sum_j (1 - sum_c U_ij_c V_jk_c W_ki_c)^2
        # A_ki is m x rank, A_ki[j, c] = U[i*m+j, c] * V[j*p+k, c]
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k * n + i, :] = w
        
        # Check convergence
        if _verify_safe(U, V, W, n, m, p, tol=1e-5):
            return U, V, W
            
    # Final check with tighter tolerance
    if _verify_safe(U, V, W, n, m, p, tol=1e-6):
        return U, V, W
        
    return None

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
    
    # Try Rank 88 (Current Program Method)
    if time.time() < deadline:
        try:
            U_88, V_88, W_88 = _construct_rank_88()
            if _verify_safe(U_88.tolist(), V_88.tolist(), W_88.tolist()):
                if U_88.shape[1] < best_rank:
                    best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
                    best_rank = U_88.shape[1]
        except Exception:
            pass
            
    # Try to find lower rank using ALS
    if time.time() < deadline:
        for rank in range(80, 88):
            if time.time() > deadline - 20:
                break
            found = False
            for seed in range(10):
                if time.time() > deadline - 10:
                    break
                try:
                    result = _als_decomposition(n, m, p, rank, iterations=50, seed=seed)
                    if result is not None:
                        U, V, W = result
                        if _verify_safe(U.tolist(), V.tolist(), W.tolist()):
                            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            best_rank = rank
                            found = True
                            break
                except Exception:
                    pass
            if found:
                # If we found a rank, we could try lower, but let's stick to this for now
                # or break if we hit a specific target
                pass

    return best_result
