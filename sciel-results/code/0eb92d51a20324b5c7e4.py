"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(tensor_entries)  # should be 24
    
    # Generate schoolbook as fallback (rank 24)
    best_rank = n * m * p
    U_sb = [[0] * best_rank for _ in range(nm)]
    V_sb = [[0] * best_rank for _ in range(mp)]
    W_sb = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    np.random.seed(42)
    
    # Precompute entry index arrays for vectorized operations
    u_indices = np.array([e[0] for e in tensor_entries])
    v_indices = np.array([e[1] for e in tensor_entries])
    w_indices = np.array([e[2] for e in tensor_entries])
    
    def solve_for_W(U_curr, V_curr, target_rank):
        """Solve for W given U, V using vectorized construction."""
        A_mat = np.zeros((num_entries, pn * target_rank))
        for r in range(target_rank):
            uv_products = U_curr[u_indices, r] * V_curr[v_indices, r]
            for e in range(num_entries):
                A_mat[e, w_indices[e] * target_rank + r] = uv_products[e]
        
        b_vec = np.ones(num_entries)
        W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        W_curr = W_flat.reshape(pn, target_rank)
        # Round to half-integers
        W_curr = np.round(W_curr * 2) / 2
        return W_curr
    
    def solve_for_U(V_curr, W_curr, target_rank):
        """Solve for U given V, W using vectorized construction."""
        A_mat = np.zeros((num_entries, nm * target_rank))
        for r in range(target_rank):
            vw_products = V_curr[v_indices, r] * W_curr[w_indices, r]
            for e in range(num_entries):
                A_mat[e, u_indices[e] * target_rank + r] = vw_products[e]
        
        b_vec = np.ones(num_entries)
        U_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        U_curr = U_flat.reshape(nm, target_rank)
        U_curr = np.round(U_curr * 2) / 2
        return U_curr
    
    def solve_for_V(U_curr, W_curr, target_rank):
        """Solve for V given U, W using vectorized construction."""
        A_mat = np.zeros((num_entries, mp * target_rank))
        for r in range(target_rank):
            uw_products = U_curr[u_indices, r] * W_curr[w_indices, r]
            for e in range(num_entries):
                A_mat[e, v_indices[e] * target_rank + r] = uw_products[e]
        
        b_vec = np.ones(num_entries)
        V_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        V_curr = V_flat.reshape(mp, target_rank)
        V_curr = np.round(V_curr * 2) / 2
        return V_curr
    
    def compute_error(U_curr, V_curr, W_curr, target_rank):
        """Compute total absolute error."""
        total_error = 0.0
        for e in range(num_entries):
            u_idx, v_idx, w_idx = tensor_entries[e]
            val = sum(U_curr[u_idx, r] * V_curr[v_idx, r] * W_curr[w_idx, r] for r in range(target_rank))
            total_error += abs(val - 1.0)
        return total_error
    
    def is_valid(U_curr, V_curr, W_curr, target_rank):
        """Check if the decomposition is exact."""
        error = compute_error(U_curr, V_curr, W_curr, target_rank)
        return error < 1e-9

    # Try ranks from 15 to 24
    for target_rank in range(15, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Random restarts with coordinate descent
        num_restarts = 100 if target_rank < 20 else 50
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random half-integer values
            # Use a smaller range for initialization to help convergence
            init_vals = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
            U_curr = np.random.choice(init_vals, size=(nm, target_rank))
            V_curr = np.random.choice(init_vals, size=(mp, target_rank))
            W_curr = np.random.choice(init_vals, size=(pn, target_rank))
            
            # Iterative refinement
            prev_error = float('inf')
            for iteration in range(200):
                if time.time() > deadline - 3:
                    break
                
                # Solve for each factor in turn
                W_curr = solve_for_W(U_curr, V_curr, target_rank)
                U_curr = solve_for_U(V_curr, W_curr, target_rank)
                V_curr = solve_for_V(U_curr, W_curr, target_rank)
                
                # Check convergence
                error = compute_error(U_curr, V_curr, W_curr, target_rank)
                
                if error < 1e-9:
                    # Found exact solution!
                    best_rank = target_rank
                    U_list = [[float(x) for x in U_curr[i]] for i in range(nm)]
                    V_list = [[float(x) for x in V_curr[i]] for i in range(mp)]
                    W_list = [[float(x) for x in W_curr[i]] for i in range(pn)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
                
                # Early stop if no improvement
                if error >= prev_error and iteration > 20:
                    break
                prev_error = error
            
            if found:
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
