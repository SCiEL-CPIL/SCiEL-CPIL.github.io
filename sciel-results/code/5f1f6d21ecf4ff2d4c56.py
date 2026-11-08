"""Optimized program for matrix multiplication 2x4x5 with improved search."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    def build_strassen_36():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def u_idx(i, j):
            return i * m + j
        
        def v_idx(j, k):
            return j * p + k
        
        def w_idx(k, i):
            return k * n + i
        
        col = 0
        
        def strassen_block(y_indices, z_indices, w_offsets):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices
            
            # M1 = (y00+y11)(z00+z11)
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M2 = (y10+y11)z00
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
            col += 1
            
            # M3 = y00(z01-z11)
            U[u_idx(y00[0], y00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M4 = y11(z10-z00)
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] -= 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            col += 1
            
            # M5 = (y00+y01)z11
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y01[0], y01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            col += 1
            
            # M6 = (y10-y00)(z00+z01)
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y00[0], y00[1])][col] -= 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M7 = (y01-y11)(z10+z11)
            U[u_idx(y01[0], y01[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] -= 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            col += 1
        
        # Block 1: A_L * B_11 -> C_11
        y1 = [(0,0), (0,1), (1,0), (1,1)]
        z1 = [(0,0), (0,1), (1,0), (1,1)]
        w1 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y1, z1, w1)
        
        # Block 2: A_R * B_21 -> C_11
        y2 = [(0,2), (0,3), (1,2), (1,3)]
        z2 = [(2,0), (2,1), (3,0), (3,1)]
        w2 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y2, z2, w2)
        
        # Block 3: A_L * B_12 -> C_12
        y3 = [(0,0), (0,1), (1,0), (1,1)]
        z3 = [(0,2), (0,3), (1,2), (1,3)]
        w3 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y3, z3, w3)
        
        # Block 4: A_R * B_22 -> C_12
        y4 = [(0,2), (0,3), (1,2), (1,3)]
        z4 = [(2,2), (2,3), (3,2), (3,3)]
        w4 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y4, z4, w4)
        
        # Block 5: A_L * B_15 (2x2 * 2x1)
        U[u_idx(0,0)][col] += 1
        V[v_idx(0,4)][col] += 1
        W[w_idx(4,0)][col] += 1
        col += 1
        U[u_idx(0,1)][col] += 1
        V[v_idx(1,4)][col] += 1
        W[w_idx(4,0)][col] += 1
        col += 1
        U[u_idx(1,0)][col] += 1
        V[v_idx(0,4)][col] += 1
        W[w_idx(4,1)][col] += 1
        col += 1
        U[u_idx(1,1)][col] += 1
        V[v_idx(1,4)][col] += 1
        W[w_idx(4,1)][col] += 1
        col += 1
        
        # Block 6: A_R * B_25 (2x2 * 2x1)
        U[u_idx(0,2)][col] += 1
        V[v_idx(2,4)][col] += 1
        W[w_idx(4,0)][col] += 1
        col += 1
        U[u_idx(0,3)][col] += 1
        V[v_idx(3,4)][col] += 1
        W[w_idx(4,0)][col] += 1
        col += 1
        U[u_idx(1,2)][col] += 1
        V[v_idx(2,4)][col] += 1
        W[w_idx(4,1)][col] += 1
        col += 1
        U[u_idx(1,3)][col] += 1
        V[v_idx(3,4)][col] += 1
        W[w_idx(4,1)][col] += 1
        col += 1
        
        assert col == rank, f"Expected {rank}, got {col}"
        return {"U": U, "V": V, "W": W}
    
    best_result = build_strassen_36()
    best_rank = 36
    
    def verify_decomposition(U, V, W, tol=1e-6):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A_idx = i * m + j
                    B_idx = j * p + k
                    C_idx = k * n + i
                    computed = np.sum(U_arr[A_idx, :] * V_arr[B_idx, :] * W_arr[C_idx, :])
                    if abs(computed - 1) > tol:
                        return False
        
        # Check a subset of non-matching entries
        rng = np.random.RandomState(42)
        for _ in range(1000):
            A_idx = rng.randint(0, n*m)
            B_idx = rng.randint(0, m*p)
            C_idx = rng.randint(0, p*n)
            
            i = A_idx // m
            j = A_idx % m
            j2 = B_idx // p
            k = B_idx % p
            k2 = C_idx // n
            i2 = C_idx % n
            
            if not (j == j2 and k == k2 and i == i2):
                computed = np.sum(U_arr[A_idx, :] * V_arr[B_idx, :] * W_arr[C_idx, :])
                if abs(computed) > tol:
                    return False
        
        return True
    
    # Pre-compute target and non-target indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
    
    all_indices = set()
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                all_indices.add((a,b,c))
    
    target_set = set(target_indices)
    non_target_list = list(all_indices - target_set)
    rng = np.random.RandomState(123)
    rng.shuffle(non_target_list)
    # Use more non-target indices for robustness
    non_target_indices = non_target_list[:800]
    
    def objective(params, rank):
        num_u = n * m * rank
        num_v = m * p * rank
        num_w = p * n * rank
        
        U_flat = params[:num_u]
        V_flat = params[num_u:num_u + num_v]
        W_flat = params[num_u + num_v:]
        
        U = U_flat.reshape((n * m, rank))
        V = V_flat.reshape((m * p, rank))
        W = W_flat.reshape((p * n, rank))
        
        error = 0.0
        for a, b, c in target_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            error += (val - 1.0) ** 2
        
        for a, b, c in non_target_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            error += val ** 2
            
        return error
    
    # Try to find rank 35 solution
    target_rank = 35
    U0 = np.array(best_result['U'], dtype=float)
    V0 = np.array(best_result['V'], dtype=float)
    W0 = np.array(best_result['W'], dtype=float)
    
    max_attempts = 30
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if attempt < 15:
            # Warm start: drop a random column from rank 36
            drop_idx = attempt % 36
            U_init = np.delete(U0, drop_idx, axis=1)
            V_init = np.delete(V0, drop_idx, axis=1)
            W_init = np.delete(W0, drop_idx, axis=1)
            
            # Add small noise
            noise_scale = 0.05 + 0.01 * (attempt % 5)
            U_init += np.random.normal(0, noise_scale, U_init.shape)
            V_init += np.random.normal(0, noise_scale, V_init.shape)
            W_init += np.random.normal(0, noise_scale, W_init.shape)
        else:
            # Random initialization with structured bias
            U_init = np.random.normal(0, 0.5, (n*m, target_rank))
            V_init = np.random.normal(0, 0.5, (m*p, target_rank))
            W_init = np.random.normal(0, 0.5, (p*n, target_rank))
        
        params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = minimize(objective, params0, args=(target_rank,), method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < 1e-8:
                num_u = n * m * target_rank
                num_v = m * p * target_rank
                num_w = p * n * target_rank
                
                U_sol = res.x[:num_u].reshape((n * m, target_rank))
                V_sol = res.x[num_u:num_u + num_v].reshape((m * p, target_rank))
                W_sol = res.x[num_u + num_v:].reshape((p * n, target_rank))
                
                # Try rounding to half-integers
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                if verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                
                # Try integer rounding
                U_int = np.round(U_sol)
                V_int = np.round(V_sol)
                W_int = np.round(W_sol)
                
                if verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                    
        except Exception:
            pass
    
    return best_result
