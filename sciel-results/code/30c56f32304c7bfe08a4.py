import numpy as np
import time

n, m, p = 2, 5, 6
TOTAL_TERMS = n * m * p  # 60

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def check_decomposition(U, V, W, tol=1e-9):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def compute_tensor_residuals(U, V, W):
        R = U.shape[1]
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    res[idx] = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                    idx += 1
        return res - 1.0

    def als_decomposition(U_init, V_init, W_init, max_iters=100, tol=1e-10):
        """Alternating least squares to refine decomposition."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        R = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    A_mat = np.zeros((p, R))
                    b_vec = np.ones(p)
                    for k in range(p):
                        A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[u_idx, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    A_mat = np.zeros((n, R))
                    b_vec = np.ones(n)
                    for i in range(n):
                        A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[v_idx, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    A_mat = np.zeros((m, R))
                    b_vec = np.ones(m)
                    for j in range(m):
                        A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[w_idx, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            res = compute_tensor_residuals(U, V, W)
            max_err = np.max(np.abs(res))
            if max_err < tol:
                break
        
        return U, V, W

    def build_strassen_decomposition():
        """
        Build a decomposition using Strassen-style 2x2x2 blocks.
        Rank 54.
        """
        def strassen_2x2_block(i_offset, j_offset, k_offset):
            U_block = np.zeros((4, 7))
            V_block = np.zeros((4, 7))
            W_block = np.zeros((4, 7))
            
            # M1 = (A11+A22)(B11+B22)
            U_block[0, 0] = 1; U_block[3, 0] = 1
            V_block[0, 0] = 1; V_block[3, 0] = 1
            W_block[0, 0] = 1; W_block[1, 0] = 1; W_block[2, 0] = 1; W_block[3, 0] = 1
            
            # M2 = (A21+A22)B11
            U_block[2, 1] = 1; U_block[3, 1] = 1
            V_block[0, 1] = 1
            W_block[0, 1] = -1; W_block[1, 1] = 1
            
            # M3 = A11(B21-B22)
            U_block[0, 2] = 1
            V_block[2, 2] = 1; V_block[3, 2] = -1
            W_block[1, 2] = 1; W_block[3, 2] = -1
            
            # M4 = A22(B12-B11)
            U_block[3, 3] = 1
            V_block[1, 3] = 1; V_block[0, 3] = -1
            W_block[0, 3] = 1; W_block[2, 3] = 1
            
            # M5 = (A11+A12)B22
            U_block[0, 4] = 1; U_block[1, 4] = 1
            V_block[3, 4] = 1
            W_block[2, 4] = 1; W_block[3, 4] = 1
            
            # M6 = (A21-A11)(B11+B12)
            U_block[2, 5] = 1; U_block[0, 5] = -1
            V_block[0, 5] = 1; V_block[1, 5] = 1
            W_block[0, 5] = -1
            
            # M7 = (A12-A22)(B21+B22)
            U_block[1, 6] = 1; U_block[3, 6] = -1
            V_block[2, 6] = 1; V_block[3, 6] = 1
            W_block[3, 6] = -1
            
            U_full = np.zeros((n * m, 7))
            V_full = np.zeros((m * p, 7))
            W_full = np.zeros((p * n, 7))
            
            for a in range(2):
                for b in range(2):
                    u_idx = (i_offset + a) * m + (j_offset + b)
                    U_full[u_idx, :] = U_block[a * 2 + b, :]
            
            for b in range(2):
                for c in range(2):
                    v_idx = (j_offset + b) * p + (k_offset + c)
                    V_full[v_idx, :] = V_block[b * 2 + c, :]
            
            for c in range(2):
                for a in range(2):
                    w_idx = (k_offset + c) * n + (i_offset + a)
                    W_full[w_idx, :] = W_block[c * 2 + a, :]
            
            return U_full, V_full, W_full
        
        U_list = []
        V_list = []
        W_list = []
        
        j_blocks_2 = [(0, 1), (2, 3)]
        k_blocks_2 = [(0, 1), (2, 3), (4, 5)]
        
        for (j0, j1) in j_blocks_2:
            for (k0, k1) in k_blocks_2:
                U_b, V_b, W_b = strassen_2x2_block(0, j0, k0)
                U_list.append(U_b)
                V_list.append(V_b)
                W_list.append(W_b)
        
        # Schoolbook for j=4
        for i in range(2):
            for k in range(6):
                U_t = np.zeros((n * m, 1))
                V_t = np.zeros((m * p, 1))
                W_t = np.zeros((p * n, 1))
                U_t[i * m + 4, 0] = 1
                V_t[4 * p + k, 0] = 1
                W_t[k * n + i, 0] = 1
                U_list.append(U_t)
                V_list.append(V_t)
                W_list.append(W_t)
        
        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        
        rank = U.shape[1]
        res = compute_tensor_residuals(U, V, W)
        max_err = np.max(np.abs(res))
        
        if max_err < 1e-9:
            return U, V, W, rank
        else:
            return None, None, None, None

    # Build initial Strassen decomposition (rank 54)
    U_str, V_str, W_str, rank_str = build_strassen_decomposition()
    if U_str is None:
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U_str, V_str, W_str
    best_rank = rank_str
    
    # Try to reduce rank to 53
    target_rank = 53
    if time.time() < deadline - 20:
        # Try a few random column drops
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            drop_idx = np.random.randint(0, best_rank)
            keep_idx = [i for i in range(best_rank) if i != drop_idx]
            
            U_sub = best_U[:, keep_idx].copy()
            V_sub = best_V[:, keep_idx].copy()
            W_sub = best_W[:, keep_idx].copy()
            
            U_ref, V_ref, W_ref = als_decomposition(U_sub, V_sub, W_sub, max_iters=50)
            
            res = compute_tensor_residuals(U_ref, V_ref, W_ref)
            err = np.max(np.abs(res))
            
            if err < 1e-9:
                best_U, best_V, best_W = U_ref, V_ref, W_ref
                best_rank = target_rank
                break
    
    # Final check
    res = compute_tensor_residuals(best_U, best_V, best_W)
    max_err = np.max(np.abs(res))
    
    if max_err < 1e-9:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
