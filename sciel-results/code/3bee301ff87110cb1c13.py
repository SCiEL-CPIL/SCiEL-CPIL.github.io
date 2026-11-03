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

def _construct_rank_48():
    """
    Construct a rank-48 decomposition.
    Split A (3x4) into three 1x4 rows.
    Each 1x4x8 subproblem is split into 1x2x4 + 1x2x4 (rank 8+8=16).
    Total: 3 * 16 = 48.
    """
    n, m, p = 3, 4, 8
    
    U_list = []
    V_list = []
    W_list = []
    
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
    
    def add_1x4x8(i_off, j_off, k_off):
        """Add a 1x4x8 subproblem at offset (i_off, j_off, k_off)."""
        u_sub, v_sub, w_sub = get_1x2x4()
        rank_sub = u_sub.shape[1]
        
        # First half: j in [j_off, j_off+2), k in [k_off, k_off+4)
        for c in range(rank_sub):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            # U: 1x2 block at (i_off, j_off)
            for j in range(2):
                u_col[i_off * m + (j_off + j)] = u_sub[j, c]
            
            # V: 2x4 block at (j_off, k_off)
            for j in range(2):
                for k in range(4):
                    v_col[(j_off + j) * p + (k_off + k)] = v_sub[j * 4 + k, c]
            
            # W: 4x1 block at (k_off, i_off)
            for k in range(4):
                w_col[(k_off + k) * n + i_off] = w_sub[k, c]
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)
        
        # Second half: j in [j_off+2, j_off+4), k in [k_off, k_off+4)
        for c in range(rank_sub):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            # U: 1x2 block at (i_off, j_off+2)
            for j in range(2):
                u_col[i_off * m + (j_off + 2 + j)] = u_sub[j, c]
            
            # V: 2x4 block at (j_off+2, k_off)
            for j in range(2):
                for k in range(4):
                    v_col[(j_off + 2 + j) * p + (k_off + k)] = v_sub[j * 4 + k, c]
            
            # W: 4x1 block at (k_off, i_off)
            for k in range(4):
                w_col[(k_off + k) * n + i_off] = w_sub[k, c]
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)
    
    # Three 1x4x8 subproblems for each row of A
    for i in range(3):
        add_1x4x8(i, 0, 0)
    
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
    
    # Try Rank 48
    if time.time() < deadline:
        try:
            U_48, V_48, W_48 = _construct_rank_48()
            if _verify_safe(U_48.tolist(), V_48.tolist(), W_48.tolist()):
                if U_48.shape[1] < best_rank:
                    best_result = {"U": U_48.tolist(), "V": V_48.tolist(), "W": W_48.tolist()}
                    best_rank = U_48.shape[1]
        except Exception:
            pass
            
    return best_result
