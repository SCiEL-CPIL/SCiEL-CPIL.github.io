"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Build the target tensor T
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def check_and_snap(U, V, W, r):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        U_snap = np.round(U_np * 2) / 2
        V_snap = np.round(V_np * 2) / 2
        W_snap = np.round(W_np * 2) / 2
        
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return None
            
        U_out = [[float(x) for x in row] for row in U_snap]
        V_out = [[float(x) for x in row] for row in V_snap]
        W_out = [[float(x) for x in row] for row in W_snap]
        
        return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    def try_rank(r, seed=42, init_type='structured'):
        """Try to find a rank-r decomposition with given initialization strategy."""
        if time.time() > deadline:
            return None
            
        np.random.seed(seed)
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        if init_type == 'structured':
            # Structured initialization exploiting n=2 symmetry
            # For each k, we have 5 j-values. Use rank terms that combine rows.
            # Pattern: group j's into pairs and use +/- combinations
            for idx in range(r):
                # Determine which k this term primarily serves
                k_idx = idx % p
                j_block = (idx // p) % m
                i_pattern = (idx // (p * m)) % 2
                
                # U part: depends on i_pattern and j_block
                if i_pattern == 0:
                    # Row 0 contributions
                    for j in range(m):
                        if j == j_block:
                            U[0 * m + j, idx] = 1.0
                    # Maybe also row 1 with sign
                    if r > p * m:
                        U[1 * m + j_block, idx] = (-1) ** (idx % 2)
                else:
                    # Row 1 contributions
                    for j in range(m):
                        if j == j_block:
                            U[1 * m + j, idx] = 1.0
                    if r > p * m:
                        U[0 * m + j_block, idx] = (-1) ** (idx % 2)
                
                # V part: column k_idx
                for j in range(m):
                    V[j * p + k_idx, idx] = 1.0 if (j == j_block or r > p * m) else 0.0
                if r > p * m:
                    V[j_block * p + k_idx, idx] = (-1) ** (idx % 2)
                
                # W part: output (k_idx, i_pattern)
                W[k_idx * n + i_pattern, idx] = 1.0
                if r > p * m:
                    W[k_idx * n + (1 - i_pattern), idx] = 0.5
        
        elif init_type == 'winograd':
            # Winograd-style: use DFT-like structure over j dimension
            # For each k, create 5 terms using cyclic shifts
            omega = np.exp(2j * np.pi / m)
            for idx in range(r):
                k_idx = idx % p
                freq = (idx // p) % m
                
                # U: row 0 gets DFT basis, row 1 gets conjugate or shifted
                for j in range(m):
                    U[0 * m + j, idx] = np.cos(2 * np.pi * freq * j / m)
                    U[1 * m + j, idx] = np.sin(2 * np.pi * freq * j / m) if freq > 0 else 0.0
                
                # V: column k_idx with phase
                for j in range(m):
                    V[j * p + k_idx, idx] = 1.0
                
                # W: output
                if freq == 0:
                    W[k_idx * n + 0, idx] = 1.0
                    W[k_idx * n + 1, idx] = 0.0
                else:
                    W[k_idx * n + 0, idx] = 0.5
                    W[k_idx * n + 1, idx] = 0.5
        
        else:  # random
            for idx in range(r):
                for i in range(n):
                    for j in range(m):
                        if np.random.rand() < 0.3:
                            U[i*m+j, idx] = np.random.choice([-1, 0, 1])
                for j in range(m):
                    for k in range(p):
                        if np.random.rand() < 0.3:
                            V[j*p+k, idx] = np.random.choice([-1, 0, 1])
                for k in range(p):
                    for i in range(n):
                        if np.random.rand() < 0.3:
                            W[k*n+i, idx] = np.random.choice([-1, 0, 1])
        
        def loss(params):
            size_U = (n * m) * r
            size_V = (m * p) * r
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]
            
            U_p = U_flat.reshape((n * m), r)
            V_p = V_flat.reshape((m * p), r)
            W_p = W_flat.reshape((p * n), r)
            
            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            err = np.sum((T_recon - T_target) ** 2)
            return err
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 800, 'maxfun': 5000, 'ftol': 1e-15})
        except Exception:
            return None
            
        if res.fun < 1e-8:
            size_U = (n * m) * r
            size_V = (m * p) * r
            U_sol = res.x[:size_U].reshape((n * m), r)
            V_sol = res.x[size_U:size_U + size_V].reshape((m * p), r)
            W_sol = res.x[size_U + size_V:].reshape((p * n), r)
            
            result = check_and_snap(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), r)
            if result:
                return result
        
        # Try a polish pass if close
        if res.fun < 1e-5:
            size_U = (n * m) * r
            size_V = (m * p) * r
            U_sol = res.x[:size_U].reshape((n * m), r)
            V_sol = res.x[size_U:size_U + size_V].reshape((m * p), r)
            W_sol = res.x[size_U + size_V:].reshape((p * n), r)
            
            # Snap to nearest half-integers and check
            U_snap = np.round(U_sol * 2) / 2
            V_snap = np.round(V_sol * 2) / 2
            W_snap = np.round(W_sol * 2) / 2
            
            T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
            if np.allclose(T_recon, T_target, atol=1e-9):
                return check_and_snap(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), r)
        
        return None
    
    # Try ranks from 48 down to 30
    ranks_to_try = list(range(48, 29, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        found = False
        
        # Try different initialization strategies
        for init_type in ['structured', 'winograd', 'random']:
            for seed in [42, 123, 777, 999, 12345]:
                if time.time() > deadline:
                    break
                result = try_rank(r, seed, init_type)
                if result:
                    best_result = result
                    best_rank = r
                    found = True
                    break
            if found:
                break
        
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
