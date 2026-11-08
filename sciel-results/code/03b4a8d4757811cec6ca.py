import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill
    
    # 1. Construct the known rank 36 decomposition (Strassen-style blocks)
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
    
    # Pre-compute indices for verification and objective
    # Target: T[i, j, k] = 1 for all i, j, k
    # Indices: a = i*m+j, b = j*p+k, c = k*n+i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
                
    # We need to check all non-target indices to ensure they are 0
    # Total size is (n*m) * (m*p) * (p*n) = 8 * 20 * 10 = 1600
    # This is small enough to check all in verification, but for objective we might sample if needed.
    # However, for exactness, we should ideally penalize all non-targets.
    # Let's create a list of all indices and mark targets.
    all_indices = []
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                all_indices.append((a, b, c))
                
    target_set = set(target_indices)
    non_target_indices = [idx for idx in all_indices if idx not in target_set]
    
    def verify_decomposition(U, V, W, tol=1e-6):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Efficient check:
        # For each target (a,b,c), sum_r U[a,r] V[b,r] W[c,r] should be 1.
        # For each non-target (a,b,c), sum_r U[a,r] V[b,r] W[c,r] should be 0.
        
        # We can compute the full tensor contraction T = U V W (tensor product)
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is equivalent to reshaping and using einsum or loop.
        # Given small size, einsum is fast.
        T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        for a, b, c in target_indices:
            if abs(T[a, b, c] - 1.0) > tol:
                return False
        
        for a, b, c in non_target_indices:
            if abs(T[a, b, c]) > tol:
                return False
                
        return True

    def calculate_error(params, rank):
        """
        Objective function for optimization.
        params: flattened vector of U, V, W
        rank: current rank being tested
        """
        num_u = n * m * rank
        num_v = m * p * rank
        num_w = p * n * rank
        
        U_flat = params[:num_u]
        V_flat = params[num_u:num_u + num_v]
        W_flat = params[num_u + num_v:]
        
        U = U_flat.reshape((n * m, rank))
        V = V_flat.reshape((m * p, rank))
        W = W_flat.reshape((p * n, rank))
        
        # Compute T = U V W
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # To avoid creating the full (8, 20, 10) tensor in memory repeatedly, 
        # we can compute the error directly for the specific indices.
        # But einsum is very fast for small arrays.
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        error = 0.0
        # Target error
        for a, b, c in target_indices:
            val = T[a, b, c]
            error += (val - 1.0) ** 2
            
        # Non-target error
        # To speed up, we can vectorize this if we pre-store indices as arrays
        # But 1600-40=1560 iterations is fine in Python if inside a tight loop? 
        # Actually, Python loops are slow. Let's use NumPy advanced indexing.
        
        # Pre-computed indices for NumPy
        # We'll pass these as closure variables or compute once outside.
        
        # Let's use the pre-computed lists converted to arrays for speed
        # We need to update the objective to use pre-computed arrays.
        
        return error

    # Pre-compute NumPy arrays for indices to speed up objective
    target_a = np.array([idx[0] for idx in target_indices])
    target_b = np.array([idx[1] for idx in target_indices])
    target_c = np.array([idx[2] for idx in target_indices])
    
    non_target_a = np.array([idx[0] for idx in non_target_indices])
    non_target_b = np.array([idx[1] for idx in non_target_indices])
    non_target_c = np.array([idx[2] for idx in non_target_indices])
    
    def fast_objective(params, rank):
        num_u = n * m * rank
        num_v = m * p * rank
        num_w = p * n * rank
        
        U_flat = params[:num_u]
        V_flat = params[num_u:num_u + num_v]
        W_flat = params[num_u + num_v:]
        
        U = U_flat.reshape((n * m, rank))
        V = V_flat.reshape((m * p, rank))
        W = W_flat.reshape((p * n, rank))
        
        # Compute T
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Extract values for target indices
        T_target = T[target_a, target_b, target_c]
        # Extract values for non-target indices
        T_non_target = T[non_target_a, non_target_b, non_target_c]
        
        # Calculate SSE
        error_target = np.sum((T_target - 1.0) ** 2)
        error_non_target = np.sum(T_non_target ** 2)
        
        return error_target + error_non_target

    # Try to find rank 35 solution
    target_rank = 35
    U0 = np.array(best_result['U'], dtype=float)
    V0 = np.array(best_result['V'], dtype=float)
    W0 = np.array(best_result['W'], dtype=float)
    
    max_attempts = 20
    rng = np.random.RandomState(42)
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
            
        # Initialization: Drop a random column from rank 36
        drop_idx = attempt % 36
        U_init = np.delete(U0, drop_idx, axis=1).copy()
        V_init = np.delete(V0, drop_idx, axis=1).copy()
        W_init = np.delete(W0, drop_idx, axis=1).copy()
        
        # Add noise to escape local minima
        # Scale noise based on rank?
        noise_scale = 0.5
        U_init += rng.normal(0, noise_scale, U_init.shape)
        V_init += rng.normal(0, noise_scale, V_init.shape)
        W_init += rng.normal(0, noise_scale, W_init.shape)
        
        params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # Use L-BFGS-B
            res = minimize(fast_objective, params0, args=(target_rank,), method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 50000})
            
            if res.fun < 1e-4: # Good enough to try rounding
                num_u = n * m * target_rank
                num_v = m * p * target_rank
                num_w = p * n * target_rank
                
                U_sol = res.x[:num_u].reshape((n * m, target_rank))
                V_sol = res.x[num_u:num_u + num_v].reshape((m * p, target_rank))
                W_sol = res.x[num_u + num_v:].reshape((p * n, target_rank))
                
                # Round to half-integers
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                # Verify
                if verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    
                # If direct rounding failed, try local search on the grid
                # This is expensive, so only do it if we have time
                if time.time() < deadline - 10:
                    # Simple local search: try flipping signs or adding 0.5 to a few elements?
                    # Or just accept failure for this attempt and try next init.
                    pass
                    
        except Exception as e:
            pass
            
    # If rank 35 not found, return rank 36
    return best_result
