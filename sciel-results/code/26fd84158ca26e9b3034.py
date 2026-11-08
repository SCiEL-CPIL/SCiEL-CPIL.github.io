import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 125  # seconds, safety margin from 150s

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_baseline = 50
    U = [[0] * rank_baseline for _ in range(n * m)]
    V = [[0] * rank_baseline for _ in range(m * p)]
    W = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1; V[h_idx][col] += 1
        W[c00_idx][col] += 1; W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1; W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx][col] += 1
        V[f_idx][col] += 1; V[h_idx][col] -= 1
        W[c01_idx][col] += 1; W[c11_idx][col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx][col] += 1
        V[g_idx][col] += 1; V[e_idx][col] -= 1
        W[c00_idx][col] += 1; W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx][col] += 1; U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1; W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx][col] += 1; U[a_idx][col] -= 1
        V[e_idx][col] += 1; V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx][col] += 1; U[d_idx][col] -= 1
        V[g_idx][col] += 1; V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00_idx][col] += 1
        col += 1
        U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00_idx][col] += 1
        col += 1
        U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10_idx][col] += 1
        col += 1
        U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10_idx][col] += 1
        col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)
    
    assert col == rank_baseline, f"Expected rank {rank_baseline}, got {col}"
    
    # Convert to numpy arrays for numerical rank reduction
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify the baseline is correct
    def verify_decomposition(U_arr, V_arr, W_arr):
        # T[i,j,k] should equal sum_r U[i,r]*V[j,r]*W[k,r]
        # For matrix mult: T[i,j,k] = 1 if j is the inner index connecting A[i,j] and B[j,k]
        # Actually T[i,j,k] = delta_{i,j,k} where the tensor is T[i,j,k] = 1 if the multiplication
        # C[i,k] = sum_j A[i,j] * B[j,k] has term A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k (it's the identity-like tensor for matmul)
        # Wait, the matrix multiplication tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # No, that's not right. The tensor is T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
        # The tensor has shape (n*m, m*p, p*n) in the flattened index convention
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # The decomposition is: for each (i,j,k) in the tensor, T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Wait, let me re-read the convention.
        # U[i*m+j, r] corresponds to A[i,j]
        # V[j*p+k, r] corresponds to B[j,k]  
        # W[k*n+i, r] corresponds to C[k,i]... wait, W[k*n+i, r]
        # So W index is k*n + i, meaning W is indexed by (k, i) where k is col of C, i is row of C
        # The tensor T has entries T[a_idx, b_idx, c_idx] where:
        # a_idx = i*m + j (A[i,j])
        # b_idx = j*p + k (B[j,k])
        # c_idx = k*n + i (C[i,k] stored as W[k*n+i])
        # T[a_idx, b_idx, c_idx] = 1 if a_idx, b_idx, c_idx are consistent (same j connects them)
        # i.e., a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
        # For each (i,j,k): a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
        # T should have 1 at (i*m+j, j*p+k, k*n+i)
        
        n_val, m_val, p_val = n, m, p
        T_expected = np.zeros((n_val * m_val, m_val * p_val, p_val * n_val))
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    a_idx = i * m_val + j
                    b_idx = j * p_val + k
                    c_idx = k * n_val + i
                    T_expected[a_idx, b_idx, c_idx] = 1
        
        # Compute T_decomposed
        # T_decomposed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is expensive for large dimensions, but our dims are small
        T_decomposed = np.zeros((n_val * m_val, m_val * p_val, p_val * n_val))
        for r in range(U_arr.shape[1]):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            # Outer product of three vectors
            # T_decomposed[a,b,c] += u_col[a] * v_col[b] * w_col[c]
            T_decomposed += np.einsum('a,b,c->abc', u_col, v_col, w_col)
        
        return np.allclose(T_decomposed, T_expected, atol=1e-10)
    
    assert verify_decomposition(U_np, V_np, W_np), "Baseline decomposition is incorrect!"
    
    # Try to reduce rank by finding linear dependencies
    # Fast verification using the fact that columns are sparse
    def verify_fast(U_arr, V_arr, W_arr):
        """Fast verification: check T_decomposed == T_expected"""
        n_val, m_val, p_val = n, m, p
        T_expected = np.zeros((n_val * m_val, m_val * p_val, p_val * n_val))
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    a_idx = i * m_val + j
                    b_idx = j * p_val + k
                    c_idx = k * n_val + i
                    T_expected[a_idx, b_idx, c_idx] = 1
        
        # Compute T_decomposed efficiently
        T_decomposed = np.zeros((n_val * m_val, m_val * p_val, p_val * n_val))
        for r in range(U_arr.shape[1]):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            # Only process if all three have non-zero entries
            if np.any(u_col != 0) and np.any(v_col != 0) and np.any(w_col != 0):
                T_decomposed += np.einsum('a,b,c->abc', u_col, v_col, w_col)
        
        return np.allclose(T_decomposed, T_expected, atol=1e-10)
    
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    best_rank = rank_baseline
    
    # Iteratively try to remove columns
    progress = True
    while progress and time.time() - START_TIME < TIME_LIMIT - 20:
        progress = False
        for r in range(U_np.shape[1]):
            if time.time() - START_TIME > TIME_LIMIT - 20:
                break
            # Try removing column r
            U_test = np.delete(U_np, r, axis=1)
            V_test = np.delete(V_np, r, axis=1)
            W_test = np.delete(W_np, r, axis=1)
            
            if verify_fast(U_test, V_test, W_test):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = U_np.shape[1] - 1
                U_np, V_np, W_np = U_test, V_test, W_test
                progress = True
                break
    
    # Try to replace one column with a linear combination of others
    # If col_r = sum_{s in S} alpha_s * col_s for some subset S and scalars,
    # we can remove col_r
    for r in range(U_np.shape[1]):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        # Try to express column r as a combination of other columns
        # This is a hard problem, so try small subsets
        
        # First, try single other column (proportional)
        for s in range(U_np.shape[1]):
            if s == r:
                continue
            # Check if U[:,r] is proportional to U[:,s]
            mask_u = np.abs(U_np[:, s]) > 1e-10
            if np.any(mask_u):
                try:
                    alpha_u = U_np[mask_u, r] / U_np[mask_u, s]
                    if np.allclose(alpha_u, alpha_u[0], atol=1e-10):
                        alpha = alpha_u[0]
                        # Check V
                        mask_v = np.abs(V_np[:, s]) > 1e-10
                        if np.any(mask_v):
                            alpha_v = V_np[mask_v, r] / V_np[mask_v, s]
                            if np.allclose(alpha_v, alpha_v[0], atol=1e-10) and np.isclose(alpha_v[0], alpha, atol=1e-10):
                                # Check W
                                mask_w = np.abs(W_np[:, s]) > 1e-10
                                if np.any(mask_w):
                                    alpha_w = W_np[mask_w, r] / W_np[mask_w, s]
                                    if np.allclose(alpha_w, alpha_w[0], atol=1e-10) and np.isclose(alpha_w[0], alpha, atol=1e-10):
                                        # Can remove r
                                        U_test = np.delete(U_np, r, axis=1)
                                        V_test = np.delete(V_np, r, axis=1)
                                        W_test = np.delete(W_np, r, axis=1)
                                        if verify_fast(U_test, V_test, W_test):
                                            U_np, V_np, W_np = U_test, V_test, W_test
                                            best_U, best_V, best_W = U_test, V_test, W_test
                                            best_rank = U_np.shape[1]
                                            break
                except:
                    pass
            if time.time() - START_TIME > TIME_LIMIT - 15:
                break
    
    # Try a more aggressive approach: use optimization to find lower rank
    # We can try to find a decomposition with rank < current by solving
    # a system of equations
    
    # For now, let's try to see if we can find a rank-40 or lower decomposition
    # by trying different block configurations
    
    # Actually, let's try to use the structure more cleverly
    # The 2x4x7 matmul can be decomposed as:
    # - 2x2x7 blocks (use Strassen for each 2x2x2 subblock)
    # - Or 2x4x4 + 2x4x3
    
    # Let's try a different decomposition strategy
    # Use 2x2x4 blocks: each 2x2x4 can be done in 2*7=14 ops (2 Strassen 2x2x2)
    # Plus remaining 2x2x3 = 2*7 + 1*7 = 21? No...
    
    # Actually, let me try to see if we can do better with a different tiling
    # 2x4x7 = 2x2x7 + 2x2x7 (two 2x2x7 blocks)
    # Each 2x2x7 = 7 * (2x2x1) = 7 * 4 = 28 schoolbook, or
    # Each 2x2x7 can use Strassen: 7 Strassen 2x2x2 = 7*7 = 49? No...
    
    # Let me just try to find if any column can be removed by solving
    # the tensor equation
    
    # Convert back to list format
    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()
    
    # Round to nearest integer or half-integer if close
    for arr in [U_final, V_final, W_final]:
        for row in arr:
            for idx in range(len(row)):
                val = row[idx]
                if abs(val - round(val)) < 1e-10:
                    row[idx] = int(round(val))
                elif abs(val * 2 - round(val * 2)) < 1e-10:
                    row[idx] = round(val * 2) / 2
    
    return {"U": U_final, "V": V_final, "W": W_final}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"Time: {time.time() - START_TIME:.2f}s")
