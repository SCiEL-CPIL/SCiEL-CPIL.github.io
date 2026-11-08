import numpy as np
from datetime import datetime, timedelta
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    U0, V0, W0 = _rank50_baseline(n, m, p)
    best_rank = 50
    best_U, best_V, best_W = U0, V0, W0
    
    start_time = datetime.now()
    timeout = timedelta(seconds=110)
    
    # Try to reduce rank by merging columns
    improved = _merge_columns_search(best_U, best_V, best_W, n, m, p, start_time, timeout)
    if improved is not None:
        U, V, W = improved
        if _verify_decomposition(U, V, W, n, m, p):
            best_rank = len(U[0])
            best_U, best_V, best_W = U, V, W
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _merge_columns_search(U, V, W, n, m, p, start_time, timeout):
    """Try to reduce rank by merging pairs of columns."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    current_rank = U_np.shape[1]
    improved = False
    
    for iteration in range(50):
        if datetime.now() - start_time > timeout:
            break
        
        merged = False
        for r1 in range(current_rank):
            if datetime.now() - start_time > timeout:
                break
            for r2 in range(r1+1, current_rank):
                # Check if columns r1 and r2 can be merged
                # Sum tensor: T_sum = u_r1⊗v_r1⊗w_r1 + u_r2⊗v_r2⊗w_r2
                # Check if T_sum is rank 1
                
                # Build the 3D tensor
                T_sum = np.outer(np.outer(U_np[:, r1], V_np[:, r1]), W_np[:, r1]) + \
                        np.outer(np.outer(U_np[:, r2], V_np[:, r2]), W_np[:, r2])
                
                # Reshape to (n*m, m*p, p*n)
                T_3d = T_sum.reshape(n*m, m*p, p*n)
                
                # Check rank by flattening two dimensions
                # T_3d[a, b, c] - check if this is rank 1 in the (a, (b,c)) space
                T_mat = T_3d.reshape(n*m, m*p*p*n)
                
                # Compute SVD
                try:
                    _, s, _ = np.linalg.svd(T_mat, full_matrices=False)
                    if len(s) > 1 and s[1] < 1e-10:
                        # Rank 1! Merge the columns
                        # New column is the sum
                        U_new = U_np[:, r1] + U_np[:, r2]  # This is wrong
                        # Actually, we need to find u, v, w such that u⊗v⊗w = T_sum
                        # Since T_sum is rank 1, we can extract u, v, w from the SVD
                        U_r, s_r, Vt_r = np.linalg.svd(T_mat, full_matrices=False)
                        if s_r[0] > 1e-10:
                            u = U_r[:, 0] / np.sqrt(s_r[0])
                            v_w = Vt_r[0, :] / np.sqrt(s_r[0])
                            # v_w is of length m*p*p*n, reshape to (m*p, p*n)
                            v_w_2d = v_w.reshape(m*p, p*n)
                            # v_w_2d should be rank 1: v * w^T
                            U_r2, s_r2, Vt_r2 = np.linalg.svd(v_w_2d, full_matrices=False)
                            if len(s_r2) > 1 and s_r2[1] < 1e-10:
                                v = U_r2[:, 0] / np.sqrt(s_r2[0])
                                w = Vt_r2[0, :] / np.sqrt(s_r2[0])
                                
                                # Replace columns r1 and r2 with the new column
                                U_np = np.delete(U_np, [r1, r2], axis=1)
                                V_np = np.delete(V_np, [r1, r2], axis=1)
                                W_np = np.delete(W_np, [r1, r2], axis=1)
                                U_np = np.column_stack([U_np, u])
                                V_np = np.column_stack([V_np, v])
                                W_np = np.column_stack([W_np, w])
                                current_rank -= 1
                                merged = True
                                improved = True
                                break
                except:
                    pass
            
            if merged:
                break
        
        if not merged:
            break
    
    if improved:
        return U_np.tolist(), V_np.tolist(), W_np.tolist()
    
    return None

def _rank50_baseline(n, m, p):
    """Construct the known rank-50 decomposition."""
    # Same as before
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; U[u_d][col] += 1
        V[v_e][col] += 1; V[v_h][col] += 1
        
        U[u_c][col+1] += 1; U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1; V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1; V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1; U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1
        V[v_e][col+5] += 1; V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1
        V[v_g][col+6] += 1; V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0; w_01 = k1 * n + 0
        w_10 = k0 * n + 1; w_11 = k1 * n + 1
        
        W[w_00][col] += 1; W[w_00][col+3] += 1
        W[w_00][col+4] -= 1; W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1; W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1; W[w_10][col+3] += 1
        
        W[w_11][col] += 1; W[w_11][col+1] -= 1
        W[w_11][col+2] += 1; W[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = 0 * p + k8; v_g = 1 * p + k8
    u_a = 0 * m + 0; u_b = 0 * m + 1
    u_c = 1 * m + 0; u_d = 1 * m + 1
    
    U[u_a][col] += 1; V[v_e][col] += 1
    U[u_b][col+1] += 1; V[v_g][col+1] += 1
    U[u_c][col+2] += 1; V[v_e][col+2] += 1
    U[u_d][col+3] += 1; V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0; w_18 = k8 * n + 1
    W[w_08][col] += 1; W[w_08][col+1] += 1
    W[w_18][col+2] += 1; W[w_18][col+3] += 1
    col += 4
    
    u_ar0 = 0 * m + 2; u_ar1 = 1 * m + 2
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
        col += 2
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    if not U or not V or not W:
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    if U_np.shape[0] != n * m:
        return False
    if V_np.shape[0] != m * p:
        return False
    if W_np.shape[0] != p * n:
        return False
    if U_np.shape[1] != rank or V_np.shape[1] != rank or W_np.shape[1] != rank:
        return False
    
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += A[i, j] * V_np[j * p + k, r] * W_np[k * n + i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True
