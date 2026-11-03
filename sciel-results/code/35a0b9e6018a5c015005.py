import time
import numpy as np
import random

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 4
    TIME_LIMIT = 120.0
    
    def time_up():
        return (time.time() - start_time) > TIME_LIMIT - 5.0
    
    def verify(U, V, W, rank):
        """Verify decomposition correctness."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
        
        T_target = np.ones((n, m, p), dtype=float)
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    def build_strassen():
        """Build the base rank-28 Strassen decomposition."""
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_steps = [
            {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
            {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
            {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
            {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
            {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
            {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
            {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}}
        ]
        
        def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
            for s in range(7):
                step = strassen_steps[s]
                col = col_offset + s
                
                for local_idx, coeff in step['u'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_i = l_row
                    global_j = a_col_start + l_col
                    global_idx = global_i * m + global_j
                    U[global_idx][col] += coeff
                    
                for local_idx, coeff in step['v'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_i = b_row_start + l_row
                    global_j = b_col_start + l_col
                    global_idx = global_i * p + global_j
                    V[global_idx][col] += coeff
                    
                for local_idx, coeff in step['w'].items():
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    global_k = c_col_start + l_col
                    global_i = l_row
                    global_idx = global_k * n + global_i
                    W[global_idx][col] += coeff

        add_block(0, 0, 0, 0, 0)
        add_block(0, 0, 2, 2, 7)
        add_block(2, 2, 0, 0, 14)
        add_block(2, 2, 2, 2, 21)
        
        return U, V, W, rank
    
    def try_reduce_rank(U, V, W, current_rank, target_rank):
        """
        Try to reduce rank by finding linear dependencies.
        We look for a linear combination of the last (current_rank - target_rank) terms
        that can be expressed in terms of the first target_rank terms.
        """
        if current_rank <= target_rank:
            return None
        
        # Convert to arrays
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # We want to eliminate terms from target_rank to current_rank-1
        # by expressing them as linear combinations of terms 0..target_rank-1
        # 
        # This is equivalent to finding matrices X, Y, Z such that:
        # U[:, target_rank:] = U[:, :target_rank] @ X
        # V[:, target_rank:] = V[:, :target_rank] @ Y  
        # W[:, target_rank:] = W[:, :target_rank] @ Z
        #
        # But we need X, Y, Z to be consistent (same transformation).
        # Actually, the condition is that the tensor T = sum U_r ⊗ V_r ⊗ W_r
        # remains unchanged.
        
        # Simpler approach: try to find a linear transformation that maps
        # the full set of rank terms to a smaller set while preserving the tensor.
        
        # For now, let's try a simple approach: check if any term is a linear
        # combination of others.
        
        # Try to eliminate one term at a time
        for elim_idx in range(current_rank - 1, target_rank - 1, -1):
            if time_up():
                break
            
            # Try to express term elim_idx as a linear combination of other terms
            # We need: U[:, elim_idx] = sum_{j != elim_idx} c_j * U[:, j]
            #          V[:, elim_idx] = sum_{j != elim_idx} c_j * V[:, j]
            #          W[:, elim_idx] = sum_{j != elim_idx} c_j * W[:, j]
            
            # This is a system of linear equations. Let's set it up.
            # Unknowns: c_j for j in 0..current_rank-1, j != elim_idx
            # Equations from U: U[:, elim_idx] = U[:, :elim_idx] @ c[:elim_idx] + U[:, elim_idx+1:] @ c[elim_idx+1:]
            
            # Build the system
            other_indices = [j for j in range(current_rank) if j != elim_idx]
            num_unknowns = len(other_indices)
            
            # From U
            U_target = U_arr[:, elim_idx]
            U_matrix = np.column_stack([U_arr[:, j] for j in other_indices])
            
            # From V
            V_target = V_arr[:, elim_idx]
            V_matrix = np.column_stack([V_arr[:, j] for j in other_indices])
            
            # From W
            W_target = W_arr[:, elim_idx]
            W_matrix = np.column_stack([W_arr[:, j] for j in other_indices])
            
            # Solve the combined system
            A_combined = np.vstack([U_matrix, V_matrix, W_matrix])
            b_combined = np.concatenate([U_target, V_target, W_target])
            
            try:
                # Use least squares
                coeffs, residuals, rank_sys, sv = np.linalg.lstsq(A_combined, b_combined, rcond=None)
                
                # Check if the solution is exact
                residual = np.linalg.norm(A_combined @ coeffs - b_combined)
                if residual < 1e-9:
                    # We found a valid elimination!
                    # Create new matrices with the term eliminated
                    new_U = [U[i][:] for i in range(len(U))]
                    new_V = [V[i][:] for i in range(len(V))]
                    new_W = [W[i][:] for i in range(len(W))]
                    
                    # Apply the linear combination
                    for new_idx in range(len(other_indices)):
                        j = other_indices[new_idx]
                        c = coeffs[new_idx]
                        if abs(c) > 1e-12:
                            for i in range(len(U)):
                                new_U[i][j] += c * U[i][elim_idx]
                                new_V[i][j] += c * V[i][elim_idx]
                                new_W[i][j] += c * W[i][elim_idx]
                    
                    # Remove the eliminated term
                    new_U = [[row[j] for j in range(current_rank) if j != elim_idx] for row in new_U]
                    new_V = [[row[j] for j in range(current_rank) if j != elim_idx] for row in new_V]
                    new_W = [[row[j] for j in range(current_rank) if j != elim_idx] for row in new_W]
                    
                    return new_U, new_V, new_W, current_rank - 1
            except:
                pass
        
        return None
    
    # Start with Strassen
    U, V, W, rank = build_strassen()
    
    # Verify base case
    if not verify(U, V, W, rank):
        # Should not happen, but return anyway
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to reduce rank
    target_rank = 24  # Try to get to rank 24
    
    while rank > target_rank and not time_up():
        result = try_reduce_rank(U, V, W, rank, target_rank)
        if result is None:
            break
        
        U, V, W, rank = result
        
        # Verify
        if verify(U, V, W, rank):
            best_U, best_V, best_W, best_rank = U, V, W, rank
        else:
            # Rollback
            U, V, W, rank = best_U, best_V, best_W, best_rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
