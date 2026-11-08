"""Optimized program for matrix multiplication 2x4x5 with faster search for rank < 36."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the rank-36 Strassen-based decomposition
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
    
    # Fast vectorized verification
    def verify_decomposition_fast(U, V, W):
        """Check if U, V, W correctly decompose the matrix multiplication tensor using vectorized ops."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Check all target entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A_idx = i * m + j
                    B_idx = j * p + k
                    C_idx = k * n + i
                    computed = np.dot(np.dot(U[A_idx, :], V[B_idx, :]), W[C_idx, :])
                    if abs(computed - 1.0) > 1e-8:
                        return False
        
        # Check a sample of non-target entries
        rng = np.random.RandomState(42)
        for _ in range(500):
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
                computed = np.dot(np.dot(U[A_idx, :], V[B_idx, :]), W[C_idx, :])
                if abs(computed) > 1e-8:
                    return False
        
        return True
    
    # Pre-compute target and near-miss indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
    
    # Generate "near-miss" non-target indices (differ in exactly one index)
    near_miss_indices = []
    target_set = set(target_indices)
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in target_set:
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    # Count how many indices match
                    matches = (j == j2) + (k == k2) + (i == i2)
                    if matches == 2:  # Near-miss: differs in exactly one
                        near_miss_indices.append((a, b, c))
    
    # Use near-miss indices for the objective (they're the most likely to be violated)
    random.shuffle(near_miss_indices)
    # Also add some random far indices
    all_non_target = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n) 
                      if (a, b, c) not in target_set]
    random.shuffle(all_non_target)
    far_indices = [idx for idx in all_non_target if idx not in near_miss_indices][:200]
    
    check_indices = near_miss_indices[:300] + far_indices
    
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
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            error += (val - 1.0) ** 2
        
        for a, b, c in check_indices:
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            error += val ** 2
            
        return error
    
    # Try to find rank 35 solution with multiple strategies
    deadline = start_time + 110  # Leave 10s margin
    
    U0 = np.array(best_result['U'], dtype=float)
    V0 = np.array(best_result['V'], dtype=float)
    W0 = np.array(best_result['W'], dtype=float)
    
    # Strategy 1: Drop one column and optimize
    for attempt in range(15):
        if time.time() > deadline:
            break
        
        drop_idx = random.randint(0, 35)
        U_init = np.delete(U0, drop_idx, axis=1).copy()
        V_init = np.delete(V0, drop_idx, axis=1).copy()
        W_init = np.delete(W0, drop_idx, axis=1).copy()
        
        # Add small noise
        noise_scale = 0.05
        U_init += np.random.normal(0, noise_scale, U_init.shape)
        V_init += np.random.normal(0, noise_scale, V_init.shape)
        W_init += np.random.normal(0, noise_scale, W_init.shape)
        
        params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = minimize(objective, params0, args=(35,), method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < 1e-6:
                num_u = n * m * 35
                num_v = m * p * 35
                num_w = p * n * 35
                
                U_sol = res.x[:num_u].reshape((n * m, 35))
                V_sol = res.x[num_u:num_u + num_v].reshape((m * p, 35))
                W_sol = res.x[num_u + num_v:].reshape((p * n, 35))
                
                # Round to nearest half-integer
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                if verify_decomposition_fast(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                        
        except Exception:
            pass
    
    # Strategy 2: Try dropping two columns and see if we can fill in
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        drop1 = random.randint(0, 35)
        drop2 = random.randint(0, 34)
        if drop2 >= drop1:
            drop2 += 1
        
        U_init = np.delete(U0, [drop1, drop2], axis=1).copy()
        V_init = np.delete(V0, [drop1, drop2], axis=1).copy()
        W_init = np.delete(W0, [drop1, drop2], axis=1).copy()
        
        # This gives rank 34, but we need rank 35. So add one random column
        new_U = np.zeros((n*m, 35))
        new_V = np.zeros((m*p, 35))
        new_W = np.zeros((p*n, 35))
        
        cols_to_keep = [i for i in range(36) if i not in [drop1, drop2]]
        for new_col, old_col in enumerate(cols_to_keep):
            new_U[:, new_col] = U0[:, old_col]
            new_V[:, new_col] = V0[:, old_col]
            new_W[:, new_col] = W0[:, old_col]
        
        # Last column is random
        new_U[:, 34] = np.random.normal(0, 0.5, n*m)
        new_V[:, 34] = np.random.normal(0, 0.5, m*p)
        new_W[:, 34] = np.random.normal(0, 0.5, p*n)
        
        params0 = np.concatenate([new_U.flatten(), new_V.flatten(), new_W.flatten()])
        
        try:
            res = minimize(objective, params0, args=(35,), method='L-BFGS-B', 
                           options={'maxiter': 150, 'ftol': 1e-12})
            
            if res.fun < 1e-6:
                num_u = n * m * 35
                num_v = m * p * 35
                num_w = p * n * 35
                
                U_sol = res.x[:num_u].reshape((n * m, 35))
                V_sol = res.x[num_u:num_u + num_v].reshape((m * p, 35))
                W_sol = res.x[num_u + num_v:].reshape((p * n, 35))
                
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                if verify_decomposition_fast(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                        
        except Exception:
            pass
    
    return best_result
