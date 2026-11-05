"""Matrix multiplication 4,4,9 tensor decomposition - optimized Strassen with dependency exploitation."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 130
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    n_dim = n * m  # 16
    m_dim = m * p  # 36
    p_dim = p * n  # 36
    
    # Build the target tensor
    target = np.zeros((n_dim, m_dim, p_dim), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target[a_idx, b_idx, c_idx] = 1.0
    
    def verify(U, V, W):
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, target, atol=1e-9)
    
    # Strassen 2x2x2 base: 7 terms
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    def build_strassen_4x4x4(i_base, j_base, k_base):
        """Build 4x4x4 Strassen with 49 terms."""
        terms = []
        block_configs = [
            # (A_blocks, B_blocks, C_blocks) where each is list of (block_row, block_col, coeff)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        
        for u_c, v_c, w_c in strassen_2x2:
            u_vec = np.zeros(n_dim)
            v_vec = np.zeros(m_dim)
            w_vec = np.zeros(p_dim)
            
            for (loc_i, loc_j), coeff in u_c.items():
                for (br, bc, bc_coeff) in block_configs[0][0]:
                    gi = i_base + br * 2 + loc_i
                    gj = j_base + bc * 2 + loc_j
                    if gi < n and gj < m:
                        u_vec[gi * m + gj] += coeff * bc_coeff
            
            for (loc_j, loc_k), coeff in v_c.items():
                for (br, bc, bc_coeff) in block_configs[0][1]:
                    gj = j_base + br * 2 + loc_j
                    gk = k_base + bc * 2 + loc_k
                    if gj < m and gk < 4:
                        v_vec[gj * p + gk] += coeff * bc_coeff
            
            for (loc_i, loc_k), coeff in w_c.items():
                for (br, bc, bc_coeff) in block_configs[0][2]:
                    gi = i_base + br * 2 + loc_i
                    gk = k_base + bc * 2 + loc_k
                    if gi < n and gk < 4:
                        w_vec[gk * n + gi] += coeff * bc_coeff
            
            terms.append((u_vec, v_vec, w_vec))
        
        return terms
    
    def build_full_strassen():
        """Build rank-114 Strassen solution."""
        all_terms = []
        # Two 4x4x4 blocks for k=0..3 and k=4..7
        all_terms.extend(build_strassen_4x4x4(0, 0, 0))
        all_terms.extend(build_strassen_4x4x4(0, 0, 4))
        # Schoolbook for k=8
        for i in range(n):
            for j in range(m):
                u_vec = np.zeros(n_dim)
                v_vec = np.zeros(m_dim)
                w_vec = np.zeros(p_dim)
                u_vec[i * m + j] = 1
                v_vec[j * p + 8] = 1
                w_vec[8 * n + i] = 1
                all_terms.append((u_vec, v_vec, w_vec))
        return all_terms
    
    # Build base rank-114 solution
    base_terms = build_full_strassen()
    base_rank = len(base_terms)
    U_base = np.zeros((n_dim, base_rank))
    V_base = np.zeros((m_dim, base_rank))
    W_base = np.zeros((p_dim, base_rank))
    
    for r, (u_vec, v_vec, w_vec) in enumerate(base_terms):
        U_base[:, r] = u_vec
        V_base[:, r] = v_vec
        W_base[:, r] = w_vec
    
    # Verify base solution
    if not verify(U_base, V_base, W_base):
        # Fallback to schoolbook rank 144
        U_sb = np.zeros((n_dim, 144))
        V_sb = np.zeros((m_dim, 144))
        W_sb = np.zeros((p_dim, 144))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_U, best_V, best_W, best_rank = U_base, V_base, W_base, base_rank
    
    # Try to reduce rank by finding linear dependencies
    # Key idea: if some columns of [U;V;W] are linearly dependent in a way that
    # preserves the tensor, we can eliminate them
    
    def try_reduce_rank(current_rank, max_time=30):
        """Try to reduce rank by solving for linear combinations."""
        start_time = time.time()
        
        # Build the constraint matrix: for each (a,b,c) with target[a,b,c]=1,
        # we need sum_r U[a,r]*V[b,r]*W[c,r] = 1
        # This is nonlinear, so we use an iterative approach
        
        # Strategy: randomly set some entries to zero and solve
        import random
        rng = random.Random(42)
        
        for attempt in range(50):
            if time.time() - start_time > max_time:
                break
            if time.time() > deadline - 20:
                break
            
            target_rank = current_rank - 1 - (attempt % 5)
            if target_rank < 80:
                continue
            
            # Start from base and randomly zero out some columns
            U_try = U_base.copy()
            V_try = V_base.copy()
            W_try = W_base.copy()
            
            # Randomly select columns to modify
            num_modify = current_rank - target_rank
            cols_to_modify = rng.sample(range(current_rank), num_modify)
            
            # Set these columns to zero and try to solve
            for c in cols_to_modify:
                U_try[:, c] = 0
                V_try[:, c] = 0
                W_try[:, c] = 0
            
            # Now solve for the remaining columns to satisfy the tensor
            # This is still nonlinear, so we use a linearized approach
            # Fix U and V, solve for W (or vice versa)
            
            # Build D matrix: D[(a,b), r] = U[a,r] * V[b,r]
            D = np.zeros((n_dim * m_dim, current_rank))
            for a in range(n_dim):
                for b in range(m_dim):
                    D[a * m_dim + b, :] = U_try[a, :] * V_try[b, :]
            
            T_flat = target.reshape(n_dim * m_dim, p_dim)
            
            try:
                W_sol, residuals, rank_sys, sv = np.linalg.lstsq(D, T_flat.T, rcond=None)
                W_try = W_sol.T
            except:
                continue
            
            # Check if solution is valid
            T_check = np.einsum('ar,br,cr->abc', U_try, V_try, W_try)
            if np.allclose(T_check, target, atol=1e-8):
                # Check for rational entries
                all_vals = np.concatenate([U_try.flatten(), V_try.flatten(), W_try.flatten()])
                for denom in [1, 2, 4, 8, 16, 32, 64]:
                    scaled = all_vals * denom
                    if np.allclose(scaled, np.round(scaled), atol=1e-6):
                        U_exact = np.round(U_try * denom) / float(denom)
                        V_exact = np.round(V_try * denom) / float(denom)
                        W_exact = np.round(W_try * denom) / float(denom)
                        T_final = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                        if np.allclose(T_final, target, atol=1e-10):
                            # Count actual rank (non-zero columns)
                            actual_rank = sum(1 for r in range(current_rank) 
                                             if not (np.allclose(U_exact[:,r],0) and 
                                                     np.allclose(V_exact[:,r],0) and 
                                                     np.allclose(W_exact[:,r],0)))
                            if actual_rank < best_rank:
                                return U_exact, V_exact, W_exact, actual_rank
        
        return None
    
    # Try to reduce rank
    result = try_reduce_rank(base_rank, max_time=40)
    if result is not None:
        best_U, best_V, best_W, best_rank = result
    
    # Try another approach: use different Strassen block configurations
    def try_alternative_strassen():
        """Try alternative block decompositions."""
        start_time = time.time()
        
        # Alternative: split into 4x4x4 + 4x4x4 + 4x4x1 but with different Strassen patterns
        # Or try 4x4x5 + 4x4x4
        
        import random
        rng = random.Random(123)
        
        for attempt in range(30):
            if time.time() - start_time > 40:
                break
            if time.time() > deadline - 20:
                break
            
            # Try different rank targets
            target_rank = 110 + (attempt % 10)
            if target_rank >= best_rank:
                continue
            
            # Random structured initialization
            U_try = np.zeros((n_dim, target_rank))
            V_try = np.zeros((m_dim, target_rank))
            W_try = np.zeros((p_dim, target_rank))
            
            # Initialize with Strassen-like patterns
            for r in range(target_rank):
                # Random support sizes
                u_size = rng.randint(1, 4)
                v_size = rng.randint(1, 4)
                w_size = rng.randint(1, 4)
                
                u_support = rng.sample(range(n_dim), u_size)
                v_support = rng.sample(range(m_dim), v_size)
                w_support = rng.sample(range(p_dim), w_size)
                
                for pos in u_support:
                    U_try[pos, r] = rng.choice([-1, 1])
                for pos in v_support:
                    V_try[pos, r] = rng.choice([-1, 1])
                for pos in w_support:
                    W_try[pos, r] = rng.choice([-1, 1])
            
            # Solve for W
            D = np.zeros((n_dim * m_dim, target_rank))
            for a in range(n_dim):
                for b in range(m_dim):
                    D[a * m_dim + b, :] = U_try[a, :] * V_try[b, :]
            
            T_flat = target.reshape(n_dim * m_dim, p_dim)
            
            try:
                W_sol, residuals, rank_sys, sv = np.linalg.lstsq(D, T_flat.T, rcond=None)
                W_try = W_sol.T
            except:
                continue
            
            T_check = np.einsum('ar,br,cr->abc', U_try, V_try, W_try)
            if np.allclose(T_check, target, atol=1e-8):
                all_vals = np.concatenate([U_try.flatten(), V_try.flatten(), W_try.flatten()])
                for denom in [1, 2, 4, 8, 16, 32, 64, 128]:
                    scaled = all_vals * denom
                    if np.allclose(scaled, np.round(scaled), atol=1e-6):
                        U_exact = np.round(U_try * denom) / float(denom)
                        V_exact = np.round(V_try * denom) / float(denom)
                        W_exact = np.round(W_try * denom) / float(denom)
                        T_final = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                        if np.allclose(T_final, target, atol=1e-10):
                            actual_rank = sum(1 for r in range(target_rank) 
                                             if not (np.allclose(U_exact[:,r],0) and 
                                                     np.allclose(V_exact[:,r],0) and 
                                                     np.allclose(W_exact[:,r],0)))
                            if actual_rank < best_rank:
                                return U_exact, V_exact, W_exact, actual_rank
        
        return None
    
    result = try_alternative_strassen()
    if result is not None:
        best_U, best_V, best_W, best_rank = result
    
    # Final verification
    T_final = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
    if not np.allclose(T_final, target, atol=1e-9):
        # Fallback to base solution
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
