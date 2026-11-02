import numpy as np
import time
from itertools import product

_START_TIME = time.time()
TIME_LIMIT = 140  # Leave 10s safety margin

def construct():
    n, m, p = 2, 3, 10
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x10) using 5 Strassen blocks
    for g in range(5):
        k_base = 2 * g
        k0 = k_base
        k1 = k_base + 1
        
        v_idx_b00 = 0 * p + k0
        v_idx_b01 = 0 * p + k1
        v_idx_b10 = 1 * p + k0
        v_idx_b11 = 1 * p + k1
        
        u_idx_a00 = 0 * m + 0
        u_idx_a01 = 0 * m + 1
        u_idx_a10 = 1 * m + 0
        u_idx_a11 = 1 * m + 1
        
        w_idx_c00 = k0 * n + 0
        w_idx_c01 = k1 * n + 0
        w_idx_c10 = k0 * n + 1
        w_idx_c11 = k1 * n + 1
        
        # Term 1: m1 = (a00+a11)(b00+b11)
        U[u_idx_a00][col] = 1
        U[u_idx_a11][col] = 1
        V[v_idx_b00][col] = 1
        V[v_idx_b11][col] = 1
        W[w_idx_c00][col] = 1
        W[w_idx_c11][col] = 1
        col += 1
        
        # Term 2: m2 = (a10+a11)b00
        U[u_idx_a10][col] = 1
        U[u_idx_a11][col] = 1
        V[v_idx_b00][col] = 1
        W[w_idx_c10][col] = 1
        W[w_idx_c11][col] = -1
        col += 1
        
        # Term 3: m3 = a00(b01-b11)
        U[u_idx_a00][col] = 1
        V[v_idx_b01][col] = 1
        V[v_idx_b11][col] = -1
        W[w_idx_c01][col] = 1
        W[w_idx_c11][col] = 1
        col += 1
        
        # Term 4: m4 = a11(b10-b00)
        U[u_idx_a11][col] = 1
        V[v_idx_b10][col] = 1
        V[v_idx_b00][col] = -1
        W[w_idx_c00][col] = 1
        W[w_idx_c10][col] = 1
        col += 1
        
        # Term 5: m5 = (a00+a01)b11
        U[u_idx_a00][col] = 1
        U[u_idx_a01][col] = 1
        V[v_idx_b11][col] = 1
        W[w_idx_c01][col] = 1
        W[w_idx_c00][col] = -1
        col += 1
        
        # Term 6: m6 = (a10-a00)(b00+b01)
        U[u_idx_a10][col] = 1
        U[u_idx_a00][col] = -1
        V[v_idx_b00][col] = 1
        V[v_idx_b01][col] = 1
        W[w_idx_c11][col] = 1
        col += 1
        
        # Term 7: m7 = (a01-a11)(b10+b11)
        U[u_idx_a01][col] = 1
        U[u_idx_a11][col] = -1
        V[v_idx_b10][col] = 1
        V[v_idx_b11][col] = 1
        W[w_idx_c00][col] = 1
        col += 1

    # Part 2: A2 (2x1) * B2 (1x10) using Schoolbook
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Try to reduce rank to 54 by finding a linear dependency
    if time.time() - _START_TIME < TIME_LIMIT - 20:
        # Convert to numpy for efficient linear algebra
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # For each pair of columns, check if they can be merged
        # by solving for coefficients that make one column redundant
        for r_elim in range(55):
            if time.time() - _START_TIME > TIME_LIMIT - 10:
                break
            
            # Try to express column r_elim as a combination of others
            # This is a complex search, so we use a heuristic approach
            # Check if removing this column and adjusting one other works
            
            r_target = (r_elim + 1) % 55
            if r_target == r_elim:
                r_target = (r_elim + 2) % 55
            
            # Create modified tensors
            U_mod = U_np.copy()
            V_mod = V_np.copy()
            W_mod = W_np.copy()
            
            # Zero out column r_elim
            U_mod[:, r_elim] = 0
            V_mod[:, r_elim] = 0
            W_mod[:, r_elim] = 0
            
            # Try to find alpha such that adding alpha * column r_target 
            # to column r_elim gives zero tensor contribution
            # This requires: U_r_elim ⊗ V_r_elim ⊗ W_r_elim = -alpha * U_r_target ⊗ V_r_target ⊗ W_r_target
            
            # Check if the rank-1 tensors are proportional
            u_e = U_np[:, r_elim]
            v_e = V_np[:, r_elim]
            w_e = W_np[:, r_elim]
            u_t = U_np[:, r_target]
            v_t = V_np[:, r_target]
            w_t = W_np[:, r_target]
            
            # Check proportionality
            if np.allclose(u_e, 0) or np.allclose(v_e, 0) or np.allclose(w_e, 0):
                continue
            if np.allclose(u_t, 0) or np.allclose(v_t, 0) or np.allclose(w_t, 0):
                continue
            
            # Find alpha for each component
            alpha_u = np.where(u_t != 0, u_e / u_t, 0)
            alpha_v = np.where(v_t != 0, v_e / v_t, 0)
            alpha_w = np.where(w_t != 0, w_e / w_t, 0)
            
            # Check if all alphas are the same
            if np.allclose(alpha_u[alpha_u != 0], alpha_v[alpha_v != 0]) and \
               np.allclose(alpha_u[alpha_u != 0], alpha_w[alpha_w != 0]):
                alpha = alpha_u[alpha_u != 0][0]
                
                # Modify column r_target to absorb column r_elim
                U_mod[:, r_target] += alpha * u_e
                V_mod[:, r_target] += alpha * v_e
                W_mod[:, r_target] += alpha * w_e
                
                # Verify the modified decomposition
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            computed = 0.0
                            for r in range(55):
                                if r == r_elim:
                                    continue
                                computed += U_mod[i*m+j, r] * V_mod[j*p+k, r] * W_mod[k*n+i, r]
                            err = abs(computed - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-9:
                    # Success! Return rank-54 decomposition
                    U_54 = U_mod[:, :r_elim] if r_elim < r_target else np.concatenate([U_mod[:, :r_elim], U_mod[:, r_elim+1:]])
                    V_54 = V_mod[:, :r_elim] if r_elim < r_target else np.concatenate([V_mod[:, :r_elim], V_mod[:, r_elim+1:]])
                    W_54 = W_mod[:, :r_elim] if r_elim < r_target else np.concatenate([W_mod[:, :r_elim], W_mod[:, r_elim+1:]])
                    
                    # Convert back to list format
                    U_out = [[float(x) for x in row] for row in U_54]
                    V_out = [[float(x) for x in row] for row in V_54]
                    W_out = [[float(x) for x in row] for row in W_54]
                    
                    return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fall back to rank-55
    U_out = [[float(x) for x in row] for row in U_np]
    V_out = [[float(x) for x in row] for row in V_np]
    W_out = [[float(x) for x in row] for row in W_np]
    
    return {"U": U_out, "V": V_out, "W": W_out}
