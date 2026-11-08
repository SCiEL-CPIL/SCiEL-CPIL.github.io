import numpy as np
from datetime import datetime, timedelta
import random
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Start with the known rank-50 decomposition
    U0, V0, W0 = _rank50_baseline(n, m, p)
    best_rank = 50
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to reduce rank by merging columns
    improved = _merge_columns_search(best_U, best_V, best_W, n, m, p, start_time, deadline)
    if improved is not None:
        U, V, W = improved
        if _verify_decomposition(U, V, W, n, m, p):
            new_rank = len(U[0])
            if new_rank < best_rank:
                best_rank = new_rank
                best_U, best_V, best_W = U, V, W
    
    # Final polish pass: try to reduce rank further with tighter tolerance
    if time.time() < deadline - 5:
        polished = _polish_decomposition(best_U, best_V, best_W, n, m, p, start_time, deadline)
        if polished is not None:
            U, V, W = polished
            if _verify_decomposition(U, V, W, n, m, p):
                new_rank = len(U[0])
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U, best_V, best_W = U, V, W
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _merge_columns_search(U, V, W, n, m, p, start_time, deadline):
    """Try to reduce rank by merging pairs of columns that form a rank-1 sum."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    current_rank = U_np.shape[1]
    improved = False
    
    for iteration in range(300):
        if time.time() > deadline:
            break
        
        merged = False
        for r1 in range(current_rank):
            if time.time() > deadline:
                break
            for r2 in range(r1 + 1, current_rank):
                u1 = U_np[:, r1]
                u2 = U_np[:, r2]
                v1 = V_np[:, r1]
                v2 = V_np[:, r2]
                w1 = W_np[:, r1]
                w2 = W_np[:, r2]
                
                # Quick filter: check if any pair of vectors is nearly proportional
                nu1 = np.linalg.norm(u1)
                nu2 = np.linalg.norm(u2)
                nv1 = np.linalg.norm(v1)
                nv2 = np.linalg.norm(v2)
                nw1 = np.linalg.norm(w1)
                nw2 = np.linalg.norm(w2)
                
                if min(nu1, nu2, nv1, nv2, nw1, nw2) < 1e-12:
                    continue
                
                # Check proportionality
                a_u = np.dot(u1, u2) / (nu1 * nu1)
                b_v = np.dot(v1, v2) / (nv1 * nv1)
                c_w = np.dot(w1, w2) / (nw1 * nw1)
                res_u = np.linalg.norm(u2 - a_u * u1) / nu2
                res_v = np.linalg.norm(v2 - b_v * v1) / nv2
                res_w = np.linalg.norm(w2 - c_w * w1) / nw2
                
                # If no vector pair is close to proportional, skip
                if min(res_u, res_v, res_w) >= 1e-6:
                    continue
                
                # Try to merge: check if the sum of two rank-1 tensors is rank-1
                try:
                    # Build the 2D matrix representing the sum of two outer products
                    # T = u1 ⊗ v1 ⊗ w1 + u2 ⊗ v2 ⊗ w2
                    # Check if this tensor has rank 1 by flattening appropriately
                    
                    # Method 1: Check if u1 and u2 are proportional
                    if res_u < 1e-9:
                        # u2 ≈ a*u1, so T ≈ u1 ⊗ (v1 + a*v2) ⊗ w1 + u1 ⊗ v2 ⊗ (w2 - c*w1)
                        # This is more complex; use SVD approach
                        pass
                    
                    # Use SVD on the combined tensor
                    term1 = np.outer(u1, np.outer(v1, w1).flatten())
                    term2 = np.outer(u2, np.outer(v2, w2).flatten())
                    T_mat = term1 + term2
                    
                    s = np.linalg.svd(T_mat, full_matrices=False, compute_uv=False)
                    if len(s) > 1 and s[0] > 1e-10 and s[1] < 1e-7 * s[0]:
                        # The sum is approximately rank-1
                        U_r, s_r, Vt_r = np.linalg.svd(T_mat, full_matrices=False)
                        if s_r[0] > 1e-10:
                            u_new = U_r[:, 0] / np.sqrt(s_r[0])
                            vw_flat = Vt_r[0, :] / np.sqrt(s_r[0])
                            vw_2d = vw_flat.reshape(m * p, p * n)
                            s2 = np.linalg.svd(vw_2d, full_matrices=False, compute_uv=False)
                            if len(s2) > 1 and s2[0] > 1e-10 and s2[1] < 1e-7 * s2[0]:
                                U_r2, s_r2, Vt_r2 = np.linalg.svd(vw_2d, full_matrices=False)
                                if s_r2[0] > 1e-10:
                                    v_new = U_r2[:, 0] / np.sqrt(s_r2[0])
                                    w_new = Vt_r2[0, :] / np.sqrt(s_r2[0])
                                    
                                    # Verify the merge is valid
                                    U_test = np.delete(U_np, [r1, r2], axis=1)
                                    V_test = np.delete(V_np, [r1, r2], axis=1)
                                    W_test = np.delete(W_np, [r1, r2], axis=1)
                                    U_test = np.column_stack([U_test, u_new])
                                    V_test = np.column_stack([V_test, v_new])
                                    W_test = np.column_stack([W_test, w_new])
                                    
                                    if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p):
                                        U_np = U_test
                                        V_np = V_test
                                        W_np = W_test
                                        current_rank -= 1
                                        improved = True
                                        merged = True
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

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification with fewer random trials."""
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
    
    np.random.seed(123)
    for trial in range(2):
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
        
        if not np.allclose(C_computed, C_expected, atol=1e-8):
            return False
    
    return True

def _polish_decomposition(U, V, W, n, m, p, start_time, deadline):
    """Final polish: try to reduce rank with tighter tolerance and local optimization."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    current_rank = U_np.shape[1]
    improved = False
    
    # Try all pairs with very tight tolerance
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        merged = False
        for r1 in range(current_rank):
            if time.time() > deadline:
                break
            for r2 in range(r1 + 1, current_rank):
                u1 = U_np[:, r1]
                u2 = U_np[:, r2]
                v1 = V_np[:, r1]
                v2 = V_np[:, r2]
                w1 = W_np[:, r1]
                w2 = W_np[:, r2]
                
                nu1 = np.linalg.norm(u1)
                nu2 = np.linalg.norm(u2)
                nv1 = np.linalg.norm(v1)
                nv2 = np.linalg.norm(v2)
                nw1 = np.linalg.norm(w1)
                nw2 = np.linalg.norm(w2)
                
                if min(nu1, nu2, nv1, nv2, nw1, nw2) < 1e-14:
                    continue
                
                # Very tight proportionality check
                a_u = np.dot(u1, u2) / (nu1 * nu1)
                b_v = np.dot(v1, v2) / (nv1 * nv1)
                c_w = np.dot(w1, w2) / (nw1 * nw1)
                res_u = np.linalg.norm(u2 - a_u * u1) / nu2
                res_v = np.linalg.norm(v2 - b_v * v1) / nv2
                res_w = np.linalg.norm(w2 - c_w * w1) / nw2
                
                if min(res_u, res_v, res_w) >= 1e-10:
                    continue
                
                # Try merge with very tight SVD tolerance
                try:
                    term1 = np.outer(u1, np.outer(v1, w1).flatten())
                    term2 = np.outer(u2, np.outer(v2, w2).flatten())
                    T_mat = term1 + term2
                    
                    s = np.linalg.svd(T_mat, full_matrices=False, compute_uv=False)
                    if len(s) > 1 and s[0] > 1e-12 and s[1] < 1e-10 * s[0]:
                        U_r, s_r, Vt_r = np.linalg.svd(T_mat, full_matrices=False)
                        if s_r[0] > 1e-12:
                            u_new = U_r[:, 0] / np.sqrt(s_r[0])
                            vw_flat = Vt_r[0, :] / np.sqrt(s_r[0])
                            vw_2d = vw_flat.reshape(m * p, p * n)
                            s2 = np.linalg.svd(vw_2d, full_matrices=False, compute_uv=False)
                            if len(s2) > 1 and s2[0] > 1e-12 and s2[1] < 1e-10 * s2[0]:
                                U_r2, s_r2, Vt_r2 = np.linalg.svd(vw_2d, full_matrices=False)
                                if s_r2[0] > 1e-12:
                                    v_new = U_r2[:, 0] / np.sqrt(s_r2[0])
                                    w_new = Vt_r2[0, :] / np.sqrt(s_r2[0])
                                    
                                    U_test = np.delete(U_np, [r1, r2], axis=1)
                                    V_test = np.delete(V_np, [r1, r2], axis=1)
                                    W_test = np.delete(W_np, [r1, r2], axis=1)
                                    U_test = np.column_stack([U_test, u_new])
                                    V_test = np.column_stack([V_test, v_new])
                                    W_test = np.column_stack([W_test, w_new])
                                    
                                    if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p):
                                        U_np = U_test
                                        V_np = V_test
                                        W_np = W_test
                                        current_rank -= 1
                                        improved = True
                                        merged = True
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
    """Construct the known rank-50 decomposition using Strassen's algorithm."""
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
    for trial in range(5):
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
