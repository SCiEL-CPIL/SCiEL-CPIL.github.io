import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _compute_tensor(U, V, W):
    """Compute the tensor from the factor matrices."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        T += np.outer(U[:, c], np.outer(V[:, c], W[:, c]))
    return T

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Build Strassen 2x2x2 blocks
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Compute target tensor
    T_target = _compute_tensor(U_np, V_np, W_np)
    
    # Greedy column removal
    active_cols = list(range(rank))
    rng = np.random.RandomState(42)
    
    for _ in range(50): # Passes
        if time.time() > deadline - 10:
            break
            
        removed_any = False
        # Shuffle the order of columns to try
        indices = list(range(len(active_cols)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5:
                break
            if i >= len(active_cols):
                continue
                
            # Try to remove active_cols[i]
            new_active = active_cols[:i] + active_cols[i+1:]
            if not new_active:
                break
                
            U_sub = U_np[:, new_active]
            V_sub = V_np[:, new_active]
            W_sub = W_np[:, new_active]
            
            T_comp = _compute_tensor(U_sub, V_sub, W_sub)
            
            if np.allclose(T_comp, T_target, atol=1e-9):
                active_cols = new_active
                removed_any = True
                # Adjust indices? No, we just removed one.
                # The list `indices` is now stale, but we just continue.
                # Actually, if we remove i, the subsequent indices in `indices`
                # might refer to shifted positions.
                # To be safe, we can break and restart the pass.
                break
        
        if not removed_any:
            break
            
    U_final = U_np[:, active_cols]
    V_final = V_np[:, active_cols]
    W_final = W_np[:, active_cols]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
