import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n*m = 4, m*p = 22, p*n = 22
    
    # Build the multiplication tensor
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta (1 if valid product term)
    
    # Build list of (u_idx, v_idx, w_idx) for non-zero tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(tensor_entries)  # 44
    
    # Pre-compute entries grouped by u, v, w for fast ALS
    entries_by_u = {}
    entries_by_v = {}
    entries_by_w = {}
    for (u_idx, v_idx, w_idx) in tensor_entries:
        entries_by_u.setdefault(u_idx, []).append((v_idx, w_idx))
        entries_by_v.setdefault(v_idx, []).append((u_idx, w_idx))
        entries_by_w.setdefault(w_idx, []).append((u_idx, v_idx))
    
    # Known rank-39 decomposition
    def make_rank39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0; v_f = k1; v_g = 11 + k0; v_h = 11 + k1
            w_c00 = 2 * k0; w_c10 = 2 * k0 + 1; w_c01 = 2 * k1; w_c11 = 2 * k1 + 1
            
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        
        k2 = 10
        v_e2 = k2; v_g2 = 11 + k2
        w_c02 = 2 * k2; w_c12 = 2 * k2 + 1
        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Full verification: check non-zero entries = 1 AND all other entries = 0
    def verify_full(U, V, W):
        rank = len(U[0])
        # Check non-zero entries
        for (u_idx, v_idx, w_idx) in tensor_entries:
            total = 0.0
            for r in range(rank):
                total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            if abs(total - 1) > 1e-9:
                return False
        # Check all zero entries (sample for speed, but for small dims check all)
        # Total entries: 4*22*22 = 1936, non-zero: 44, so 1892 zero entries
        # This is feasible to check all
        nz_set = set(tensor_entries)
        for u_idx in range(n * m):
            for v_idx in range(m * p):
                for w_idx in range(p * n):
                    if (u_idx, v_idx, w_idx) not in nz_set:
                        total = 0.0
                        for r in range(rank):
                            total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                        if abs(total) > 1e-9:
                            return False
        return True
    
    # Fast verification for non-zero entries only (used during search)
    def verify_nz(U, V, W):
        rank = len(U[0])
        for (u_idx, v_idx, w_idx) in tensor_entries:
            total = 0.0
            for r in range(rank):
                total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            if abs(total - 1) > 1e-9:
                return False
        return True
    
    # Compute full error including zero-constraint penalty
    def compute_error(U, V, W, zero_weight=1.0):
        rank = U.shape[1]
        err = 0.0
        # Non-zero entries
        for (u_idx, v_idx, w_idx) in tensor_entries:
            val = np.dot(U[u_idx] * V[v_idx], W[w_idx])
            err += (val - 1.0) ** 2
        # Zero entries (sampled or full)
        nz_set = set(tensor_entries)
        count_zero = 0
        for u_idx in range(n * m):
            for v_idx in range(m * p):
                for w_idx in range(p * n):
                    if (u_idx, v_idx, w_idx) not in nz_set:
                        val = np.dot(U[u_idx] * V[v_idx], W[w_idx])
                        err += zero_weight * val * val
                        count_zero += 1
        return err
    
    # Try to find a lower rank decomposition using ALS + rounding
    def try_als_rank(target_rank, max_iters=200, seed=42, init_type='random'):
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        if init_type == 'random':
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
        elif init_type == 'strassen_perturb':
            # Start from rank-39 and truncate/perturb
            r39 = make_rank39()
            U_full = np.array(r39['U'], dtype=float)
            V_full = np.array(r39['V'], dtype=float)
            W_full = np.array(r39['W'], dtype=float)
            # Keep first target_rank columns and add small perturbation
            U = U_full[:, :target_rank].copy() + np.random.randn(n * m, target_rank) * 0.1
            V = V_full[:, :target_rank].copy() + np.random.randn(m * p, target_rank) * 0.1
            W = W_full[:, :target_rank].copy() + np.random.randn(p * n, target_rank) * 0.1
        else:
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for u_idx in range(n * m):
                entries_u = entries_by_u.get(u_idx, [])
                if not entries_u:
                    continue
                A_mat = np.array([V[v_idx] * W[w_idx] for (v_idx, w_idx) in entries_u])
                b_vec = np.ones(len(entries_u))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[u_idx] = x
                except:
                    pass
            
            # Update V
            for v_idx in range(m * p):
                entries_v = entries_by_v.get(v_idx, [])
                if not entries_v:
                    continue
                A_mat = np.array([U[u_idx] * W[w_idx] for (u_idx, w_idx) in entries_v])
                b_vec = np.ones(len(entries_v))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V[v_idx] = x
                except:
                    pass
            
            # Update W
            for w_idx in range(p * n):
                entries_w = entries_by_w.get(w_idx, [])
                if not entries_w:
                    continue
                A_mat = np.array([U[u_idx] * V[v_idx] for (u_idx, v_idx) in entries_w])
                b_vec = np.ones(len(entries_w))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W[w_idx] = x
                except:
                    pass
            
            # Compute error (non-zero entries only for speed during search)
            err = 0.0
            for (u_idx, v_idx, w_idx) in tensor_entries:
                val = np.dot(U[u_idx] * V[v_idx], W[w_idx])
                err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-12:
                break
        
        if best_U is None:
            return None
        
        # Round to half-integers
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        U_list = U_rounded.tolist()
        V_list = V_rounded.tolist()
        W_list = W_rounded.tolist()
        
        # Full verification including zero constraints
        if verify_full(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 38 down to 35
    for target_rank in [38, 37, 36, 35]:
        if time.time() > deadline - 15:
            break
        
        # Try multiple seeds with different initialization strategies
        for seed in range(6):
            if time.time() > deadline - 15:
                break
            
            if seed < 3:
                init_type = 'random'
            else:
                init_type = 'strassen_perturb'
            
            result = try_als_rank(target_rank, max_iters=80, seed=seed, init_type=init_type)
            if result is not None:
                return result
    
    # Fall back to known rank-39
    return make_rank39()
