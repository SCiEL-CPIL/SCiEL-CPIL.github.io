import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    rank = U.shape[1]
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    T_true = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    residual = M - T_true
    return np.max(np.abs(residual))

def _get_tensor_indices(n, m, p):
    tensor_i = []
    tensor_j = []
    tensor_k = []
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                tensor_i.append(i)
                tensor_j.append(j)
                tensor_k.append(k)
    return np.array(tensor_i), np.array(tensor_j), np.array(tensor_k)

def _get_non_tensor_samples(n, m, p, num_samples, seed=42):
    nm, mp, pn = n * m, m * p, p * n
    rng = np.random.RandomState(seed)
    
    # Generate random indices
    all_i = rng.randint(0, nm, num_samples * 3)
    all_j = rng.randint(0, mp, num_samples * 3)
    all_k = rng.randint(0, pn, num_samples * 3)
    
    # Filter out tensor entries
    valid = []
    for idx in range(num_samples * 3):
        i, j, k = all_i[idx], all_j[idx], all_k[idx]
        a_from_i = i // m
        b_from_i = i % m
        b_from_j = j // p
        c_from_j = j % p
        c_from_k = k // n
        a_from_k = k % n
        if not (b_from_i == b_from_j and c_from_j == c_from_k and a_from_i == a_from_k):
            valid.append(idx)
        if len(valid) >= num_samples:
            break
    
    valid = np.array(valid[:num_samples])
    return all_i[valid], all_j[valid], all_k[valid]

def _soft_max_abs(vals, scale=1.0):
    """Smooth approximation of max(|vals|) using log-sum-exp."""
    abs_vals = np.abs(vals)
    if len(abs_vals) == 0:
        return 0.0
    scaled = abs_vals / scale
    max_val = np.max(scaled)
    if max_val > 50:
        # Avoid overflow
        shifted = scaled - max_val
        log_sum = max_val + np.log(np.sum(np.exp(shifted)))
        return log_sum * scale
    else:
        return scale * np.log(np.sum(np.exp(scaled)))

def _try_rank31(U0, V0, W0, n, m, p, time_budget):
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    rank = 32
    
    tensor_i, tensor_j, tensor_k = _get_tensor_indices(n, m, p)
    num_tensor = len(tensor_i)
    
    non_tensor_i, non_tensor_j, non_tensor_k = _get_non_tensor_samples(n, m, p, 80, seed=42)
    num_non_tensor = len(non_tensor_i)
    
    best_result = None
    
    for target_col in range(rank):
        if time.time() - start_time > time_budget * 0.8:
            break
        
        # Multiple random restarts per column
        for restart in range(3):
            if time.time() - start_time > time_budget * 0.85:
                break
            
            U = U0.copy()
            V = V0.copy()
            W = W0.copy()
            
            # Structured perturbation: mix random with column-specific patterns
            seed = target_col * 100 + restart * 7 + 3
            rng = np.random.RandomState(seed)
            
            # Small random perturbation
            U[:, target_col] += 0.03 * rng.randn(nm)
            V[:, target_col] += 0.03 * rng.randn(mp)
            W[:, target_col] += 0.03 * rng.randn(pn)
            
            # Also perturb other columns slightly to allow re-optimization
            for other_col in range(rank):
                if other_col != target_col:
                    U[:, other_col] += 0.01 * rng.randn(nm)
                    V[:, other_col] += 0.01 * rng.randn(mp)
                    W[:, other_col] += 0.01 * rng.randn(pn)
            
            # Objective: soft-max of all residuals
            def objective(params):
                U_flat = params[:nm * rank]
                V_flat = params[nm * rank : nm * rank + mp * rank]
                W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
                
                U_mat = U_flat.reshape(nm, rank)
                V_mat = V_flat.reshape(mp, rank)
                W_mat = W_flat.reshape(pn, rank)
                
                # Tensor residuals
                U_t = U_mat[tensor_i]
                V_t = V_mat[tensor_j]
                W_t = W_mat[tensor_k]
                tensor_vals = np.einsum('ir,jr,kr->r', U_t, V_t, W_t)
                tensor_res = tensor_vals - 1.0
                
                # Non-tensor residuals
                U_nt = U_mat[non_tensor_i]
                V_nt = V_mat[non_tensor_j]
                W_nt = W_mat[non_tensor_k]
                non_tensor_vals = np.einsum('ir,jr,kr->r', U_nt, V_nt, W_nt)
                
                # Combined residual
                all_res = np.concatenate([tensor_res, non_tensor_vals])
                
                # Soft-max approximation
                soft_max = _soft_max_abs(all_res, scale=0.5)
                
                # Column penalty to encourage sparsity in target_col
                col_penalty = np.sum(U_mat[:, target_col]**2) + np.sum(V_mat[:, target_col]**2) + np.sum(W_mat[:, target_col]**2)
                
                return soft_max + 2.0 * col_penalty
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B',
                                  options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13})
            except Exception:
                continue
            
            U_flat = result.x[:nm * rank]
            V_flat = result.x[nm * rank : nm * rank + mp * rank]
            W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            U_mat = U_flat.reshape(nm, rank)
            V_mat = V_flat.reshape(mp, rank)
            W_mat = W_flat.reshape(pn, rank)
            
            col_norm = np.sqrt(np.sum(U_mat[:, target_col]**2) + np.sum(V_mat[:, target_col]**2) + np.sum(W_mat[:, target_col]**2))
            
            if col_norm < 0.01:
                # Try to polish: remove the column and verify
                U31 = np.delete(U_mat, target_col, axis=1)
                V31 = np.delete(V_mat, target_col, axis=1)
                W31 = np.delete(W_mat, target_col, axis=1)
                score31 = _compute_residual(U31, V31, W31, n, m, p)
                if score31 < 1e-8:
                    return U31, V31, W31, score31
                
                # If close, try high-precision polish on the 31-rank system
                if score31 < 0.1 and time.time() - start_time < time_budget * 0.9:
                    polished = _polish_rank31(U31, V31, W31, n, m, p, time_budget * 0.05)
                    if polished is not None:
                        U31p, V31p, W31p, score31p = polished
                        if score31p < 1e-8:
                            return U31p, V31p, W31p, score31p
    
    return None

def _polish_rank31(U, V, W, n, m, p, time_budget):
    """High-precision polish using SLSQP on exact max-residual."""
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    tensor_i, tensor_j, tensor_k = _get_tensor_indices(n, m, p)
    non_tensor_i, non_tensor_j, non_tensor_k = _get_non_tensor_samples(n, m, p, 100, seed=123)
    
    def objective(params):
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U_mat = U_flat.reshape(nm, rank)
        V_mat = V_flat.reshape(mp, rank)
        W_mat = W_flat.reshape(pn, rank)
        
        # Tensor residuals
        U_t = U_mat[tensor_i]
        V_t = V_mat[tensor_j]
        W_t = W_mat[tensor_k]
        tensor_vals = np.einsum('ir,jr,kr->r', U_t, V_t, W_t)
        tensor_res = tensor_vals - 1.0
        
        # Non-tensor residuals
        U_nt = U_mat[non_tensor_i]
        V_nt = V_mat[non_tensor_j]
        W_nt = W_mat[non_tensor_k]
        non_tensor_vals = np.einsum('ir,jr,kr->r', U_nt, V_nt, W_nt)
        
        all_res = np.concatenate([tensor_res, non_tensor_vals])
        return np.max(np.abs(all_res))
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = minimize(objective, x0, method='SLSQP',
                          options={'maxiter': 300, 'ftol': 1e-18})
    except Exception:
        return None
    
    if time.time() - start_time > time_budget:
        return None
    
    U_flat = result.x[:nm * rank]
    V_flat = result.x[nm * rank : nm * rank + mp * rank]
    W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
    U_mat = U_flat.reshape(nm, rank)
    V_mat = V_flat.reshape(mp, rank)
    W_mat = W_flat.reshape(pn, rank)
    
    score = _compute_residual(U_mat, V_mat, W_mat, n, m, p)
    if score < 1e-10:
        return U_mat, V_mat, W_mat, score
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    U, V, W = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
    
    # Try rank-31 search
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        search_budget = min(remaining_time - 25, 90)
        result = _try_rank31(U, V, W, n, m, p, search_budget)
        if result is not None:
            U31, V31, W31, score = result
            if score < 1e-8:
                return {"U": to_list(U31), "V": to_list(V31), "W": to_list(W31)}
    
    # Fallback: return rank-32
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
